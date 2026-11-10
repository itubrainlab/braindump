"""The g.tec Unicorn Hybrid Black, through g.Pype.

The pipeline is built from g.Pype and from Rhino Recorder's g.Pype nodes
(the headset source with acquisition-time stamps, the simulated headset,
the Unicorn channel layout, swappable filters and gated loggers)::

    HeadsetSource ─┐                  ┌─ SweepTap "raw"   (quality, raw view)
    UDPReceiver ───┴─ Router ─ Layout ┼─ Filter ─ Notch ─ SweepTap "scope"
                                      └─ Recording filter ─ Router (kept
                                         channels) ─ BidsBdfLogger

The recording filter (high-pass, low-pass and notch, set in the settings;
by default a 0.1 Hz high-pass and no low-pass) is applied to the EEG as it
is recorded, and described in the sidecars. The Filter and Notch boxes
only shape the view.
"""
from __future__ import annotations

import threading
import time
from array import array
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import gpype as gp
import numpy as np
from gpype.common.constants import Constants
from rhinorecorder import layout, presets
from rhinorecorder.nodes import EegFilter, SweepTap, UnicornLayout
from rhinorecorder.recorder import GATE_MARGIN
from rhinorecorder.settings import trigger_bind_address
from rhinorecorder.sources import HeadsetSource, SimulatedSource

from .. import __version__, clock
from ..bids import NA, write_json, write_tsv
from .base import (IDLE, RUNNING, STARTING, STOPPING, Device, DeviceConfig,
                   Event, RecordedFiles, RecordingTarget, Timeline)
from .eeg import BidsBdfLogger, TimingFit, fit_timing  # noqa: F401

#: The Unicorn's default electrode positions (10-20 system).
DEFAULT_ELECTRODES = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]

#: Optional channel groups, as Rhino Recorder names them, and what they
#: hold.
GROUPS = {
    "accelerometer": "Accelerometer (ACC X/Y/Z)",
    "gyroscope": "Gyroscope (GYR X/Y/Z)",
    "counter": "Sample counter (CNT)",
    "battery": "Battery level (BAT)",
    "validation": "Validation indicator (VALID)",
    "timing": "Inter-sample time (DT)",
}

#: BIDS channel type and unit of each Unicorn layout channel.
_BIDS_TYPE = {"ACC": ("ACCEL", "g"), "GYR": ("GYRO", "deg/s"),
              "CNT": ("MISC", NA), "BAT": ("MISC", "%"),
              "VALID": ("MISC", NA), "DT": ("MISC", "ms"),
              "TRIG": ("TRIG", NA)}


#: The recording filter's choices: every cut-off of the Unicorn Recorder's
#: predefined filters (see :mod:`rhinorecorder.presets`). 0 is no
#: high-pass; half the sampling rate is no low-pass.
NYQUIST = layout.SAMPLING_RATE / 2


def _cutoffs(btypes: tuple[str, ...], index: int) -> list[float]:
    return sorted({float(spec[1][index if spec[0] == "bandpass" else 0])
                   for spec in presets.FILTERS.values()
                   if spec is not None and spec[0] in btypes})


HIGHPASS = [0.0] + _cutoffs(("highpass", "bandpass"), 0)
LOWPASS = _cutoffs(("lowpass", "bandpass"), 1) + [float(NYQUIST)]


@dataclass
class RecordingFilter:
    """The filter applied to the EEG channels as they are recorded: a
    4th-order Butterworth high-pass, low-pass or band-pass, then a notch,
    all causal (as Unicorn Recorder filters)."""
    highpass: float = 0.1
    lowpass: float = float(NYQUIST)
    notch: str = presets.NONE
    fs: float = float(layout.SAMPLING_RATE)

    @property
    def low_cutoff(self) -> Optional[float]:
        """The high-pass's cut-off, or None."""
        return float(self.highpass) if 0 < self.highpass < self.fs / 2 \
            else None

    @property
    def high_cutoff(self) -> Optional[float]:
        """The low-pass's cut-off, or None (up to half the sampling rate,
        which is no low-pass)."""
        return float(self.lowpass) if 0 < self.lowpass < self.fs / 2 \
            else None

    def notches(self) -> list[float]:
        return presets.notch_frequencies(self.notch, self.fs) \
            if self.notch in presets.NOTCHES else []

    def sos(self) -> Optional[np.ndarray]:
        """Second-order sections of the whole filter; None for none."""
        from scipy import signal
        low, high = self.low_cutoff, self.high_cutoff
        sections = []
        if low and high:
            if low >= high:
                raise ValueError(f"The recording filter's high-pass "
                                 f"({low:g} Hz) is above its low-pass "
                                 f"({high:g} Hz).")
            sections.append(signal.butter(presets.FILTER_ORDER, [low, high],
                                          btype="bandpass", fs=self.fs,
                                          output="sos"))
        elif low or high:
            sections.append(signal.butter(
                presets.FILTER_ORDER, low or high,
                btype="highpass" if low else "lowpass", fs=self.fs,
                output="sos"))
        if self.notches():
            sections.append(presets.notch_sos(self.notch, self.fs))
        return np.vstack(sections) if sections else None

    def software_filters(self):
        """The eeg.json's SoftwareFilters: what was applied, or n/a."""
        found = {}
        low, high = self.low_cutoff, self.high_cutoff
        common = {"Type": "Butterworth", "Order": presets.FILTER_ORDER,
                  "Causal": True, "Channels": "EEG"}
        if low and high:
            found["BandPass"] = {**common, "CutoffFrequencies": [low, high]}
        elif low:
            found["HighPass"] = {**common, "CutoffFrequency": low}
        elif high:
            found["LowPass"] = {**common, "CutoffFrequency": high}
        if self.notches():
            found["Notch"] = {"Type": "IIR notch",
                              "QualityFactor": presets.NOTCH_Q,
                              "Frequencies": self.notches(),
                              "Causal": True, "Channels": "EEG"}
        return found or NA


@dataclass
class UnicornConfig(DeviceConfig):
    #: Headset serial, e.g. UN-2024.10.18; empty for the first found.
    serial: str = ""
    simulate: bool = False
    electrodes: list[str] = field(
        default_factory=lambda: list(DEFAULT_ELECTRODES))
    accelerometer: bool = True
    gyroscope: bool = True
    counter: bool = True
    battery: bool = True
    validation: bool = True
    timing: bool = False
    #: The headset's internal square-wave test signal instead of the EEG.
    test_signal: bool = False
    #: Bluetooth delay taken off the time stamps, in ms.
    latency_ms: float = 40.0
    #: UDP port triggers arrive on (into TRIG); 0 is off.
    trigger_port: int = 0
    trigger_ip: str = "127.0.0.1"
    reference: str = "mastoids"
    ground: str = "mastoid"
    placement: str = "10-20"
    # The filter on the recorded EEG (see RecordingFilter): 0 Hz is no
    # high-pass, half the sampling rate no low-pass.
    record_highpass: float = 0.1
    record_lowpass: float = float(NYQUIST)
    record_notch: str = presets.NONE
    # The view (not recorded)
    view_filter: str = presets.DEFAULT_FILTER
    view_notch: str = presets.DEFAULT_NOTCH
    view_scaling: str = presets.DEFAULT_SCALING
    view_spectrum: bool = True

    def recording_filter(self) -> RecordingFilter:
        return RecordingFilter(float(self.record_highpass),
                               float(self.record_lowpass),
                               str(self.record_notch))

    def groups(self) -> dict[str, bool]:
        return {name: bool(getattr(self, name)) for name in GROUPS}

    def electrode_names(self) -> list[str]:
        names = [str(n).strip() for n in (self.electrodes or [])]
        names += [f"EEG {i + 1}" for i in range(len(names), layout.NUM_EEG)]
        return [n or f"EEG {i + 1}"
                for i, n in enumerate(names[:layout.NUM_EEG])]


#: Column of the sample counter in the headset's scan.
SCAN_COUNTER = 15
class _Stamped:
    """Keeps every scan's acquisition time, on brAIn dump's clock.

    Rhino Recorder's sources stamp each frame with when its scan was
    acquired, correcting for Bluetooth bursts and the link's delay; g.Pype
    uses the stamp to place triggers and then drops it before any sink.
    This keeps it, with the scan's counter, so a recording's timing can be
    measured: when its first sample was acquired, and how fast the
    headset's clock really ran."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stamps_lock = threading.Lock()
        self.acquired = array("d")
        self.counters = array("d")

    def start(self):
        with self._stamps_lock:
            self.acquired = array("d")
            self.counters = array("d")
        return super().start()

    def cycle(self):
        frame = self._current_frame
        result = super().cycle()
        width = getattr(self, "_total_channels", None)
        if frame is not None and width is not None and \
                frame.shape[1] > width:
            from gpype.common._private import channels
            now = channels.stamp_clock()
            to_clock = clock.from_stamp(now) - now
            with self._stamps_lock:
                for row in frame:
                    stamp = channels.unwrap_time(float(row[width]), now)
                    self.acquired.append(stamp + to_clock)
                    self.counters.append(float(row[SCAN_COUNTER]))
        return result

    def stamps(self) -> tuple[np.ndarray, np.ndarray]:
        """Acquisition times and counters of every scan since the start."""
        with self._stamps_lock:
            return (np.frombuffer(self.acquired, dtype=float).copy(),
                    np.frombuffer(self.counters, dtype=float).copy())


class StampedHeadsetSource(_Stamped, HeadsetSource):
    """The headset, keeping every scan's acquisition time."""


class StampedSimulatedSource(_Stamped, SimulatedSource):
    """The simulated headset, keeping every scan's acquisition time."""


@dataclass
class Graph:
    pipeline: gp.Pipeline
    source: object
    layout: UnicornLayout
    raw: SweepTap
    scope: SweepTap
    bandpass: EegFilter
    notch: EegFilter
    logger: BidsBdfLogger
    recorded: RecordingFilter


def find_headsets(rescan: bool = False) -> list[str]:
    """Serials of the headsets in range (or paired, on Windows)."""
    from rhinorecorder.ui.connect_dialog import find_headsets as find
    return find(rescan)


class UnicornDevice(Device):
    TYPE = "unicorn"
    NAME = "g.tec Unicorn"
    MANUFACTURER = "g.tec medical engineering GmbH"
    MODEL = "Unicorn Hybrid Black"
    DESCRIPTION = "8-channel EEG headset, 250 Hz, over Bluetooth (g.Pype)"
    MODALITY = "eeg"
    CONFIG = UnicornConfig

    def __init__(self, config: UnicornConfig, project_folder: Path,
                 simulate: bool = False):
        super().__init__(config, project_folder, simulate)
        self.graph: Optional[Graph] = None
        self._target: Optional[RecordingTarget] = None
        self._channels: list[int] = []

    # -- acquisition -----------------------------------------------------

    def _source(self):
        cfg = self.config
        common = dict(include_accel=True, include_gyro=True,
                      include_aux=True, frame_size=1, enable_oscar=False,
                      name="unicorn", latency_ms=cfg.latency_ms,
                      test_signal=cfg.test_signal)
        if self.simulated:
            return StampedSimulatedSource(**common)
        return StampedHeadsetSource(serial=cfg.serial or None, **common)

    def build(self) -> Graph:
        cfg = self.config
        p = gp.Pipeline()
        source = self._source()
        upstream = source
        if cfg.trigger_port:
            trigger = gp.UDPReceiver(
                ip=trigger_bind_address(cfg.trigger_ip, cfg.trigger_port),
                port=cfg.trigger_port, name="trigger")
            merge = gp.Router(input_channels=[gp.Router.ALL, gp.Router.ALL],
                              name="merge")
            p.connect(source, merge["in1"])
            p.connect(trigger, merge["in2"])
            upstream = merge
        arranged = UnicornLayout(raw=True, name="layout")
        p.connect(upstream, arranged)
        raw = SweepTap(seconds=10, name="raw")
        p.connect(arranged, raw)
        bandpass = EegFilter(presets.filter_sos(cfg.view_filter),
                             name="filter")
        notch = EegFilter(presets.notch_sos(cfg.view_notch), name="notch")
        p.connect(arranged, bandpass)
        p.connect(bandpass, notch)
        scope = SweepTap(seconds=10, name="scope")
        p.connect(notch, scope)

        self._channels = layout.selected(cfg.groups())
        logger = BidsBdfLogger(cfg.electrode_names(), cfg.latency_ms,
                               name="bids_bdf")
        recorded = cfg.recording_filter()
        upstream = arranged
        sos = recorded.sos()
        if sos is not None:
            # Fixed while connected: the sidecars describe one filter.
            record_filter = EegFilter(sos, name="record_filter")
            p.connect(arranged, record_filter)
            upstream = record_filter
        if self._channels != list(range(len(layout.CHANNELS))):
            select = gp.Router(input_channels={
                Constants.Defaults.PORT_IN: self._channels}, name="select")
            p.connect(upstream, select)
            upstream = select
        p.connect(upstream, logger)
        return Graph(p, source, arranged, raw, scope, bandpass, notch, logger,
                     recorded)

    def start(self) -> None:
        with self._lock:
            if self.state != IDLE:
                return
            self.error = None
            self._set_state(STARTING)
            graph = None
            try:
                graph = self.build()
                graph.pipeline.add_error_handler(self._pipeline_failed)
                graph.pipeline.start()
            except BaseException:
                if graph is not None:
                    try:
                        graph.pipeline.stop()
                    except Exception:  # noqa: BLE001
                        pass
                self._set_state(IDLE)
                raise
            self.graph = graph
            self._set_state(RUNNING)

    def stop(self) -> None:
        with self._lock:
            graph = self.graph
            if graph is None:
                return
            self._set_state(STOPPING)
            try:
                if self.recording:
                    self.end_recording()
            except Exception:  # noqa: BLE001
                pass
            try:
                graph.pipeline.stop()
            finally:
                self.graph = None
                self._set_state(IDLE)

    def _pipeline_failed(self, entry) -> None:
        try:
            message = str(entry["message"])
        except Exception:  # noqa: BLE001
            message = str(entry)
        self._fail(message)

    # -- live view settings ------------------------------------------------

    def set_view_filter(self, name: str) -> None:
        self.config.view_filter = name
        if self.graph is not None:
            self.graph.bandpass.set_sos(presets.filter_sos(name))

    def set_view_notch(self, name: str) -> None:
        self.config.view_notch = name
        if self.graph is not None:
            self.graph.notch.set_sos(presets.notch_sos(name))

    # -- recording -----------------------------------------------------------

    @property
    def recording(self) -> bool:
        graph = self.graph
        return bool(graph and graph.logger.recording)

    def begin_recording(self, target: RecordingTarget) -> list[Path]:
        with self._lock:
            graph = self.graph
            if graph is None:
                raise RuntimeError("The Unicorn is not acquiring.")
            path = target.dataset.path(target.entities, target.datatype,
                                       "eeg", ".bdf")
            logger = graph.logger
            logger.subject_id = f"sub-{target.entities.sub}"
            logger.recording_id = target.entities.stem()
            logger.device_serial = self.identity()
            logger.begin(str(path), logger.seen + GATE_MARGIN)
            self._target = target
            return [path]

    def end_recording(self) -> RecordedFiles:
        with self._lock:
            graph, target = self.graph, self._target
            if graph is None or target is None or not graph.logger.recording:
                return RecordedFiles()
            logger = graph.logger
            logger.end(logger.seen + GATE_MARGIN)
            deadline = time.monotonic() + 2.0
            while logger.recording and time.monotonic() < deadline:
                time.sleep(0.01)
            logger.end()
            self._target = None
            path = Path(logger.path)
            samples = logger.samples_written
            events = list(logger.events)
            labels = list(logger.labels)
            info = self.device_info
            timing = self._timing(graph.source, logger, samples)
            recorded = graph.recorded

        def finalize(reference: Optional[Timeline]) -> None:
            self._write_sidecars(target, path, samples, labels, info,
                                 timing, recorded)

        first = timing.timeline.t0
        moment = datetime.fromtimestamp(first)
        sidecars = [path.with_name(path.name.replace("_eeg.bdf", s))
                    for s in ("_eeg.json", "_channels.tsv")]
        # The triggers go into the run's events.tsv, which the session
        # writes with every device's events: placed at their sample.
        triggers = [Event(time=timing.timeline.clock(sample),
                          trial_type="trigger", value=f"{value:g}",
                          source="trigger")
                    for _, sample, value in events]
        return RecordedFiles(scans=[(path, moment)], files=[path, *sidecars],
                             first_sample=first, finalize=finalize,
                             samples=samples, events=triggers,
                             timeline=timing.timeline)

    def _timing(self, source, logger: BidsBdfLogger,
                samples: int) -> TimingFit:
        """The recording's timeline, from its samples' acquisition stamps
        (see :class:`_Stamped`), or, without them, from when its first
        sample arrived."""
        nominal = float(layout.SAMPLING_RATE)
        fallback = TimingFit(Timeline(logger.first_arrival or clock.now(),
                                      nominal, nominal), method="arrival")
        if logger.start_index is None or not hasattr(source, "stamps"):
            return fallback
        acquired, counters = source.stamps()
        start = logger.start_index
        if logger.first_counter is not None and start < len(counters) \
                and counters[start] != logger.first_counter:
            # The pipeline's index and the source's disagree: find the
            # sample by its counter.
            found = np.flatnonzero(counters == logger.first_counter)
            if not len(found):
                return fallback
            start = int(found[np.argmin(np.abs(found - start))])
        if start + samples > len(acquired) or samples == 0:
            return fallback
        return fit_timing(acquired[start:start + samples], nominal)

    def _write_sidecars(self, target: RecordingTarget, path: Path,
                        samples: int, labels: list[str],
                        info: dict, timing: TimingFit,
                        recorded: RecordingFilter) -> None:
        cfg = self.config
        fs = float(layout.SAMPLING_RATE)
        kinds = [self._bids_type(i) for i in self._channels]
        count = {k: sum(1 for t, _ in kinds if t == k)
                 for k in ("EEG", "MISC", "TRIG", "ACCEL", "GYRO")}
        software = [f"brAIn dump {__version__}", f"g.Pype {gp.__version__}"]
        if info.get("firmware_version"):
            software.append(f"firmware {info['firmware_version']}")
        write_json(path.with_name(path.name.replace("_eeg.bdf", "_eeg.json")), {
            "TaskName": target.entities.task,
            "SamplingFrequency": fs,
            "PowerLineFrequency": target.power_line_frequency,
            "SoftwareFilters": recorded.software_filters(),
            "HardwareFilters": NA,
            "EEGReference": cfg.reference,
            "EEGGround": cfg.ground,
            "EEGPlacementScheme": cfg.placement,
            "CapManufacturer": "g.tec",
            "CapManufacturersModelName": self.MODEL,
            "Manufacturer": self.MANUFACTURER,
            "ManufacturersModelName": self.MODEL,
            "DeviceSerialNumber": self.identity() or NA,
            "SoftwareVersions": ", ".join(software),
            "EEGChannelCount": count["EEG"],
            "EOGChannelCount": 0,
            "ECGChannelCount": 0,
            "EMGChannelCount": 0,
            "MiscChannelCount": count["MISC"],
            "MISCChannelCount": count["MISC"],
            "TriggerChannelCount": count["TRIG"],
            "RecordingDuration": round(samples / fs, 3),
            "RecordingType": "continuous",
            "SimulatedDevice": self.simulated,
            "TestSignal": cfg.test_signal,
            **target.common_metadata,
            **target.eeg_metadata,
            "RecordingTiming": timing.metadata(
                "corrected for Bluetooth bursts and a "
                f"{cfg.latency_ms:g} ms link delay"),
        })
        # The recording filter acts on the EEG channels only.
        low, high = recorded.low_cutoff, recorded.high_cutoff
        notches = recorded.notches()
        notch = NA
        if len(notches) == 1:
            notch = f"{notches[0]:g}"
        elif notches:
            notch = "[" + ", ".join(f"{f:g}" for f in notches) + "]"
        rows = []
        for label, (kind, unit) in zip(labels, kinds):
            eeg = kind == "EEG"
            rows.append([label, kind, unit,
                         f"{low:g}" if eeg and low else NA,
                         f"{high:g}" if eeg and high else NA,
                         notch if eeg else NA, fs, "good"])
        write_tsv(path.with_name(path.name.replace("_eeg.bdf",
                                                   "_channels.tsv")),
                  ["name", "type", "units", "low_cutoff", "high_cutoff",
                   "notch", "sampling_frequency", "status"], rows)

    @staticmethod
    def _bids_type(index: int) -> tuple[str, str]:
        label = layout.CHANNELS[index].label
        if index in layout.EEG:
            return "EEG", "uV"
        return _BIDS_TYPE.get(label.split(" ")[0], ("MISC", NA))

    # -- status ----------------------------------------------------------------

    @property
    def device_info(self) -> dict:
        graph = self.graph
        device = getattr(graph.source, "_device", None) if graph else None
        info = {}
        for key in ("serial_number", "firmware_version", "device_version"):
            try:
                info[key] = str(getattr(device, key))
            except Exception:  # noqa: BLE001
                pass
        return info

    def identity(self) -> str:
        if self.simulated:
            from rhinorecorder.simulator import SIMULATED_SERIAL
            return SIMULATED_SERIAL
        graph = self.graph
        if graph is not None:
            try:
                return str(graph.source.device_serial())
            except Exception:  # noqa: BLE001
                pass
        return self.config.serial or ""

    @property
    def lost_scans(self) -> int:
        graph = self.graph
        return int(graph.layout.invalid_scans) if graph else 0

    def battery(self) -> Optional[float]:
        graph = self.graph
        if graph is None:
            return None
        rows = graph.raw.latest(1)
        return float(rows[-1, layout.BAT]) if len(rows) else None

    def status(self) -> str:
        name = self.identity() or "Unicorn"
        if self.state == RUNNING:
            text = "recording" if self.recording else "acquiring"
            battery = self.battery()
            extra = f", battery {battery:.0f}%" if battery is not None else ""
            return f"{name}: {text}{extra}"
        return {IDLE: f"{name}: not connected",
                STARTING: f"{name}: connecting…",
                STOPPING: f"{name}: stopping…"}.get(self.state, self.state)
