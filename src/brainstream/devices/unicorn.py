"""The g.tec Unicorn Hybrid Black, as g.Pype's own HybridBlack source.

The headset's nodes in the session pipeline::

    HybridBlack ─┐                       ┌─ SweepTap "raw"  (quality, raw view)
    UDPReceiver ─┴─ Router ─ Labeler ────┼─ Filter ─ Notch ─ SweepTap "scope"
                                         └─ GenericFilter … ─ ChannelSelector
                                              ─ BdfWriter   (BIDS, recording)

g.Pype's HybridBlack drives the headset (and, simulated, a stand-in for
its driver: :mod:`.unicorn_sim`);
:class:`~brainstream.pipeline.AcquisitionStamps` keeps every scan's
acquisition time, corrected for Bluetooth bursts. The channels are named
with g.Pype's ChannelLabeler and chosen with its
ChannelSelector. The recording filter (high-pass, low-pass and notch, set
in the settings; by default a 0.1 Hz high-pass and no low-pass) is a chain
of g.Pype GenericFilters on the EEG channels, fixed while connected and
described in the sidecars. The Filter and Notch boxes only shape the view,
through filters that can be swapped while the pipeline runs
(:class:`~brainstream.pipeline.ViewFilter`: g.Pype's own filters can only be
bypassed).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import gpype as gp
import numpy as np
from gpype.backend.sources.hybrid_black import HybridBlack
from scipy import signal

from .. import __version__, clock, presets
from ..bids import NA, write_json, write_tsv
from ..pipeline import AcquisitionStamps, BdfWriter, SweepTap, ViewFilter
from .base import (IDLE, RUNNING, STARTING, STOPPING, Device, DeviceConfig,
                   Event, RecordedFiles, RecordingTarget, Timeline)
from .unicorn_sim import SIMULATED_SERIAL, SimulatedHybridBlack
from .eeg import (RecordingFilter, TimingFit,  # noqa: F401
                  channel_filter_columns, fit_timing, highpass_choices,
                  lowpass_choices)

SAMPLING_RATE = 250
NUM_EEG = 8

#: The Unicorn's default electrode positions (10-20 system).
DEFAULT_ELECTRODES = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]

#: The headset's channels, in the order HybridBlack delivers them (with
#: every group on): label, BIDS type, unit.
CHANNELS = ([(f"EEG {i + 1}", "EEG", "uV") for i in range(NUM_EEG)]
            + [(f"ACC {a}", "ACCEL", "g") for a in "XYZ"]
            + [(f"GYR {a}", "GYRO", "deg/s") for a in "XYZ"]
            + [("BAT", "MISC", "%"), ("CNT", "MISC", NA),
               ("VALID", "MISC", NA)])
EEG = list(range(NUM_EEG))
BAT, CNT, VALID = 14, 15, 16
#: TRIG follows the headset's channels, when a trigger port is set.
TRIGGER = len(CHANNELS)

#: Optional channel groups and what they hold.
GROUPS = {
    "accelerometer": "Accelerometer (ACC X/Y/Z)",
    "gyroscope": "Gyroscope (GYR X/Y/Z)",
    "counter": "Sample counter (CNT)",
    "battery": "Battery level (BAT)",
    "validation": "Validation indicator (VALID)",
}
_GROUP_CHANNELS = {"accelerometer": [8, 9, 10], "gyroscope": [11, 12, 13],
                   "battery": [BAT], "counter": [CNT],
                   "validation": [VALID]}

#: The recording filter's choices: every cut-off of the Unicorn Recorder's
#: predefined filters. 0 is no high-pass; half the sampling rate is no
#: low-pass.
NYQUIST = SAMPLING_RATE / 2
HIGHPASS = highpass_choices()
LOWPASS = lowpass_choices(SAMPLING_RATE)[1:] + [float(NYQUIST)]


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
    #: The headset's internal square-wave test signal instead of the EEG.
    test_signal: bool = False
    #: Bluetooth delay taken off the time stamps, in ms: by default
    #: g.tec's measurement, which g.Pype records but does not apply.
    latency_ms: float = float(HybridBlack.DEVICE_DELAY_MS)
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
                               str(self.record_notch), float(SAMPLING_RATE))

    def groups(self) -> dict[str, bool]:
        return {name: bool(getattr(self, name)) for name in GROUPS}

    def channels(self) -> list[int]:
        """The headset channels recorded: the EEG, then the groups that
        are on, in the headset's order."""
        kept = list(EEG)
        for name, on in self.groups().items():
            if on:
                kept += _GROUP_CHANNELS[name]
        return sorted(kept)

    def electrode_names(self) -> list[str]:
        names = [str(n).strip() for n in (self.electrodes or [])]
        names += [f"EEG {i + 1}" for i in range(len(names), NUM_EEG)]
        return [n or f"EEG {i + 1}" for i, n in enumerate(names[:NUM_EEG])]


class StampedHybridBlack(AcquisitionStamps, HybridBlack):
    """g.Pype's HybridBlack, keeping every scan's acquisition time."""
    counter_column = CNT
    burst_corrected = True


class StampedSimulatedHybridBlack(AcquisitionStamps, SimulatedHybridBlack):
    """g.Pype's HybridBlack on the simulated headset."""
    counter_column = CNT
    burst_corrected = True


def recording_filter_nodes(recorded: RecordingFilter,
                           prefix: str) -> list[gp.GenericFilter]:
    """The recording filter as g.Pype GenericFilters (on the EEG channels
    only): the Butterworth high-, low- or band-pass, then each notch."""
    nodes = []
    low, high = recorded.low_cutoff, recorded.high_cutoff
    if low and high:
        b, a = signal.butter(presets.FILTER_ORDER, [low, high], "bandpass",
                             fs=recorded.fs)
        nodes.append(gp.GenericFilter(b=b, a=a, name=f"{prefix}_bandpass"))
    elif low or high:
        b, a = signal.butter(presets.FILTER_ORDER, low or high,
                             "highpass" if low else "lowpass",
                             fs=recorded.fs)
        nodes.append(gp.GenericFilter(
            b=b, a=a, name=f"{prefix}_{'highpass' if low else 'lowpass'}"))
    for i, f in enumerate(recorded.notches()):
        b, a = signal.iirnotch(f, presets.NOTCH_Q, fs=recorded.fs)
        nodes.append(gp.GenericFilter(b=b, a=a,
                                      name=f"{prefix}_notch{i}"))
    return nodes


@dataclass
class Graph:
    """The headset's nodes in the session pipeline."""
    source: AcquisitionStamps
    raw: SweepTap
    scope: SweepTap
    bandpass: ViewFilter
    notch: ViewFilter
    writer: BdfWriter
    recorded: RecordingFilter
    #: Columns of the recorded channels, in the labelled stream.
    channels: list[int]
    #: Column of TRIG in the labelled stream; None without a trigger.
    trigger: Optional[int] = None


def find_headsets(rescan: bool = False) -> list[str]:
    """Serials of the headsets gtec-unicorn can open: the paired ones on
    Windows; on Linux and macOS, the devices named UN-... in range, paired
    or not, which takes several seconds (the previous search's result is
    reused unless ``rescan``).

    The driver is asked directly, as g.Pype's HybridBlack does: its
    ``get_available_devices`` can neither search again nor say why it
    found nothing.

    Raises:
        RuntimeError: If gtec-unicorn cannot be loaded on this system.
    """
    try:
        import gtec_unicorn
    except (ImportError, OSError) as error:
        raise RuntimeError(f"gtec-unicorn could not be loaded: {error}") \
            from error
    return [str(s) for s in
            gtec_unicorn.Amplifier.get_connected_devices(rescan=rescan)]


#: Addresses that mean this computer only.
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def trigger_bind_address(ip: str, port: int, platform: str = sys.platform
                         ) -> str:
    """The address the trigger input listens on. macOS lets ordinary
    programs listen on ports below 1024 only on all interfaces, not on
    127.0.0.1; there, a loopback address with such a port listens on all
    interfaces instead (still receiving what is sent to 127.0.0.1)."""
    if platform == "darwin" and port < 1024 and ip.strip() in _LOOPBACK:
        return "0.0.0.0"
    return ip


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

    # -- the session pipeline ------------------------------------------------

    def _source(self):
        cfg = self.config
        common = dict(include_accel=True, include_gyro=True,
                      include_aux=True, frame_size=1,
                      latency_ms=cfg.latency_ms,
                      test_signal=cfg.test_signal,
                      name=self.node_name("source"))
        if self.simulated:
            return StampedSimulatedHybridBlack(**common)
        return StampedHybridBlack(serial=cfg.serial or None, **common)

    def build(self, p: gp.Pipeline) -> None:
        cfg = self.config
        source = self._source()
        upstream = source
        labels = [label for label, _, _ in CHANNELS]
        labels[:NUM_EEG] = cfg.electrode_names()
        trigger = None
        if cfg.trigger_port:
            udp = gp.UDPReceiver(
                ip=trigger_bind_address(cfg.trigger_ip, cfg.trigger_port),
                port=cfg.trigger_port, name=self.node_name("trigger"))
            merge = gp.Router(input_channels=[gp.Router.ALL, gp.Router.ALL],
                              name=self.node_name("merge"))
            p.connect(source, merge["in1"])
            p.connect(udp, merge["in2"])
            upstream = merge
            labels.append("TRIG")
            trigger = TRIGGER
        named = gp.ChannelLabeler(labels=labels,
                                  name=self.node_name("labels"))
        p.connect(upstream, named)
        raw = SweepTap(seconds=10, name=self.node_name("raw"))
        p.connect(named, raw)
        bandpass = ViewFilter(presets.filter_sos(cfg.view_filter),
                             name=self.node_name("view_filter"))
        notch = ViewFilter(presets.notch_sos(cfg.view_notch),
                          name=self.node_name("view_notch"))
        p.connect(named, bandpass)
        p.connect(bandpass, notch)
        scope = SweepTap(seconds=10, name=self.node_name("scope"))
        p.connect(notch, scope)

        # Recorded: through the recording filter, the chosen channels.
        recorded = cfg.recording_filter()
        upstream = named
        for node in recording_filter_nodes(recorded,
                                           self.node_name("record")):
            p.connect(upstream, node)
            upstream = node
        channels = cfg.channels() + ([TRIGGER] if trigger is not None
                                     else [])
        if len(channels) != len(labels):
            select = gp.ChannelSelector(indices=channels,
                                        name=self.node_name("channels"))
            p.connect(upstream, select)
            upstream = select
        writer = BdfWriter(self.node_name("bids"))
        p.connect(upstream, writer)
        self.graph = Graph(source, raw, scope, bandpass, notch, writer,
                           recorded, channels, trigger)

    def closed(self) -> None:
        self.graph = None
        self._target = None

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
        return bool(graph and graph.writer.recording)

    def begin_recording(self, target: RecordingTarget) -> list[Path]:
        with self._lock:
            graph = self.graph
            if graph is None:
                raise RuntimeError("The Unicorn is not acquiring.")
            path = target.dataset.path(target.entities, target.datatype,
                                       "eeg", ".bdf")
            self.invoke(graph.writer, "begin", {
                "path": str(path),
                "meta": {"electrodes": self.config.electrode_names(),
                         "subject": f"sub-{target.entities.sub}",
                         "recording": target.entities.stem(),
                         "serial": self.identity()}})
            self._target = target
            return [path]

    def end_recording(self) -> RecordedFiles:
        with self._lock:
            graph, target = self.graph, self._target
            if graph is None or target is None or not graph.writer.recording:
                return RecordedFiles()
            writer = graph.writer
            summary = self.invoke(writer, "end")
            self._target = None
            path = Path(writer.path)
            samples = int(summary.get("samples") or 0)
            events = list(writer.events)
            labels = list(writer.file_labels)
            info = self.device_info
            timing = self._timing(graph.source, writer, samples)
            recorded = graph.recorded
            columns = list(graph.channels)

        def finalize(reference: Optional[Timeline]) -> None:
            self._write_sidecars(target, path, samples, labels, columns,
                                 info, timing, recorded)

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

    def _timing(self, source, writer: BdfWriter,
                samples: int) -> TimingFit:
        """The recording's timeline, from its samples' acquisition stamps
        (see :class:`~brainstream.pipeline.AcquisitionStamps`), or, without
        them, the nominal rate from now."""
        nominal = float(SAMPLING_RATE)
        fallback = TimingFit(Timeline(clock.now() - samples / nominal,
                                      nominal, nominal), method="arrival")
        if writer.start_index is None or not hasattr(source, "stamps"):
            return fallback
        acquired, counters = source.stamps()
        start = writer.start_index
        first_counter = writer.first_counter
        if first_counter is not None and start < len(counters) \
                and counters[start] != first_counter:
            # The pipeline's index and the source's disagree: find the
            # sample by its counter.
            found = np.flatnonzero(counters == first_counter)
            if not len(found):
                return fallback
            start = int(found[np.argmin(np.abs(found - start))])
        if start + samples > len(acquired) or samples == 0:
            return fallback
        return fit_timing(acquired[start:start + samples], nominal)

    def _write_sidecars(self, target: RecordingTarget, path: Path,
                        samples: int, labels: list[str], columns: list[int],
                        info: dict, timing: TimingFit,
                        recorded: RecordingFilter) -> None:
        cfg = self.config
        fs = float(SAMPLING_RATE)
        kinds = [self._bids_type(i) for i in columns]
        count = {k: sum(1 for t, _ in kinds if t == k)
                 for k in ("EEG", "MISC", "TRIG", "ACCEL", "GYRO")}
        software = [f"brAIn stream {__version__}", f"g.Pype {gp.__version__}"]
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
        filtered = channel_filter_columns(recorded.low_cutoff,
                                          recorded.high_cutoff,
                                          recorded.notches())
        rows = [[label, kind, unit,
                 *(filtered if kind == "EEG" else (NA, NA, NA)), fs, "good"]
                for label, (kind, unit) in zip(labels, kinds)]
        write_tsv(path.with_name(path.name.replace("_eeg.bdf",
                                                   "_channels.tsv")),
                  ["name", "type", "units", "low_cutoff", "high_cutoff",
                   "notch", "sampling_frequency", "status"], rows)

    @staticmethod
    def _bids_type(index: int) -> tuple[str, str]:
        if index == TRIGGER:
            return "TRIG", NA
        _, kind, unit = CHANNELS[index]
        return kind, unit

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
        return int(graph.source.lost_scans) if graph else 0

    def battery(self) -> Optional[float]:
        graph = self.graph
        if graph is None:
            return None
        rows = graph.raw.latest(1)
        return float(rows[-1, BAT]) if len(rows) else None

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
