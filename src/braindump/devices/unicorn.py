"""The g.tec Unicorn Hybrid Black, through g.Pype.

The pipeline is built from g.Pype and from Rhino Recorder's g.Pype nodes
(the headset source with acquisition-time stamps, the simulated headset,
the Unicorn channel layout, swappable filters and gated loggers)::

    HeadsetSource ─┐                  ┌─ SweepTap "raw"   (quality, raw view)
    UDPReceiver ───┴─ Router ─ Layout ┼─ Filter ─ Notch ─ SweepTap "scope"
                                      └─ Router (kept channels) ─ BidsBdfLogger

Recordings are raw (BIDS asks for raw data); the filters only shape the
view.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import gpype as gp
import numpy as np
from gpype.common.constants import Constants
from rhinorecorder import layout, presets
from rhinorecorder.nodes import BdfLogger, EegFilter, SweepTap, UnicornLayout
from rhinorecorder.recorder import GATE_MARGIN
from rhinorecorder.settings import trigger_bind_address

from .. import __version__
from ..bids import NA, write_json, write_tsv
from .base import (IDLE, RUNNING, STARTING, STOPPING, Device, DeviceConfig,
                   Event, RecordedFiles, RecordingTarget)

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
    # The view
    view_filter: str = presets.DEFAULT_FILTER
    view_notch: str = presets.DEFAULT_NOTCH
    view_scaling: str = presets.DEFAULT_SCALING
    view_spectrum: bool = True

    def groups(self) -> dict[str, bool]:
        return {name: bool(getattr(self, name)) for name in GROUPS}

    def electrode_names(self) -> list[str]:
        names = [str(n).strip() for n in (self.electrodes or [])]
        names += [f"EEG {i + 1}" for i in range(len(names), layout.NUM_EEG)]
        return [n or f"EEG {i + 1}"
                for i, n in enumerate(names[:layout.NUM_EEG])]


class BidsBdfLogger(BdfLogger):
    """Rhino Recorder's BDF+ logger, with the electrodes named as the
    project says, and keeping what BIDS needs beside the file: when its
    first sample was acquired, and the trigger events."""

    def __init__(self, electrodes: list[str], latency_ms: float = 40.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.electrodes = electrodes
        self.latency_s = latency_ms / 1000.0
        self.first_sample: Optional[float] = None
        self.events: list[tuple[float, int, float]] = []

    def setup(self, data, port_context_in):
        out = super().setup(data, port_context_in)
        names = dict(zip([f"EEG {i + 1}" for i in range(layout.NUM_EEG)],
                         self.electrodes))
        self.labels = [names.get(label, label) for label in self.labels]
        return out

    def _open_file(self, path):
        self.first_sample = None
        self.events = []
        super()._open_file(path)

    def _write(self, block):
        if self.first_sample is None:
            # The newest of these rows has just arrived; it was acquired
            # the link's delay before.
            self.first_sample = (time.time() - self.latency_s
                                 - (block.shape[0] - 1) / self.sampling_rate)
        if self._status is not None:
            for row in np.flatnonzero(block[:, self._status]):
                sample = self.samples_written + int(row)
                self.events.append((sample / self.sampling_rate, sample,
                                    float(block[row, self._status])))
        super()._write(block)


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
        from rhinorecorder.sources import HeadsetSource, SimulatedSource
        cfg = self.config
        common = dict(include_accel=True, include_gyro=True,
                      include_aux=True, frame_size=1, enable_oscar=False,
                      name="unicorn", latency_ms=cfg.latency_ms,
                      test_signal=cfg.test_signal)
        if self.simulated:
            return SimulatedSource(**common)
        return HeadsetSource(serial=cfg.serial or None, **common)

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
        upstream = arranged
        if self._channels != list(range(len(layout.CHANNELS))):
            select = gp.Router(input_channels={
                Constants.Defaults.PORT_IN: self._channels}, name="select")
            p.connect(arranged, select)
            upstream = select
        p.connect(upstream, logger)
        return Graph(p, source, arranged, raw, scope, bandpass, notch, logger)

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
            first = logger.first_sample
            samples = logger.samples_written
            events = list(logger.events)
            labels = list(logger.labels)
            info = self.device_info

        def finalize(reference: Optional[float]) -> None:
            self._write_sidecars(target, path, samples, labels, info)

        moment = datetime.fromtimestamp(first or target.started)
        sidecars = [path.with_name(path.name.replace("_eeg.bdf", s))
                    for s in ("_eeg.json", "_channels.tsv")]
        # The triggers go into the run's events.tsv, which the session
        # writes with every device's events.
        start = first or target.started
        triggers = [Event(time=start + onset, trial_type="trigger",
                          value=f"{value:g}", source="trigger")
                    for onset, _, value in events]
        return RecordedFiles(scans=[(path, moment)], files=[path, *sidecars],
                             first_sample=first, finalize=finalize,
                             samples=samples, events=triggers)

    def _write_sidecars(self, target: RecordingTarget, path: Path,
                        samples: int, labels: list[str],
                        info: dict) -> None:
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
            "SoftwareFilters": NA,
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
        })
        rows = []
        for label, (kind, unit) in zip(labels, kinds):
            rows.append([label, kind, unit, NA, NA, fs, "good"])
        write_tsv(path.with_name(path.name.replace("_eeg.bdf",
                                                   "_channels.tsv")),
                  ["name", "type", "units", "low_cutoff", "high_cutoff",
                   "sampling_frequency", "status"], rows)

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
