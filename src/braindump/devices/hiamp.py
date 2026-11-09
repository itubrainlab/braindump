"""The g.tec g.HIamp, through g.Pype.

g.Pype's :class:`~gpype.backend.sources.g_hiamp.GHIamp` drives the
amplifier with g.tec's gtec-gds driver (Windows, Linux and macOS)::

    GHIamp ─┬─ SweepTap "raw"                       (raw view)
            ├─ Filter ─ Notch ─ SweepTap "scope"    (filtered view)
            └─ BidsBdfLogger                        (BIDS, while recording)

The EEG channels are recorded raw, with the digital input (TRIG) beside
them: its rising edges are the run's trigger events. Each block of
samples is stamped as it arrives, so the recording's timing is measured
as for the Unicorn: its first sample's time and the amplifier's real
rate, the time base the other devices are given in.

A simulated amplifier (``simulate``) runs g.Pype's real GHIamp source on
a stand-in for the driver, as Rhino Recorder simulates the Unicorn.
"""
from __future__ import annotations

import sys
import threading
import time
from array import array
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import gpype as gp
import numpy as np
from gpype.backend.sources.g_hiamp import GHIamp
from rhinorecorder import presets
from rhinorecorder.nodes import EegFilter, SweepTap

from .. import __version__, clock
from ..bids import NA, write_json, write_tsv
from .base import (IDLE, RUNNING, STARTING, STOPPING, Device, DeviceConfig,
                   Event, RecordedFiles, RecordingTarget, Timeline)
from .eeg import SIGNAL, TRIGGER, BidsBdfLogger, TimingFit, fit_timing

#: Sampling rates offered, and the block size (scans) the driver
#: recommends for each (the simulated amplifier uses them too).
RATES = {256: 8, 512: 16, 600: 20, 1200: 40, 2400: 80, 4800: 160}
SIMULATED_SERIAL = "HA-SIMULATED"
SIMULATED_CHANNELS = 64


@dataclass
class HIampConfig(DeviceConfig):
    #: Serial of the amplifier, e.g. HA-2019.06.12; empty for the first.
    serial: str = ""
    simulate: bool = False
    sampling_rate: int = 256
    #: Channels to record, from the first; 0 is every channel the
    #: amplifier has.
    channel_count: int = 64
    #: Names of the channels, in order; the rest are Ch<n>.
    electrodes: list[str] = field(default_factory=list)
    #: Record the digital input (TRIG).
    digital_input: bool = True
    #: Delay of the USB link, in ms, taken off the time stamps.
    latency_ms: float = 0.0
    reference: str = ""
    ground: str = ""
    placement: str = "10-20"
    cap_manufacturer: str = "g.tec"
    cap_model: str = ""
    # The view
    view_filter: str = presets.DEFAULT_FILTER
    view_notch: str = presets.DEFAULT_NOTCH
    view_scaling: str = presets.DEFAULT_SCALING


# -- the source ------------------------------------------------------------------------

class StampedGHIamp(GHIamp):
    """g.Pype's GHIamp, keeping every sample's acquisition time.

    The driver hands over a block when its last sample is in; the samples
    before it were acquired one period apart before that. g.Pype stamps
    the block for its own synchronisation and drops the stamp before any
    sink; this keeps the times."""

    def __init__(self, *args, latency_ms: float = 0.0, **kwargs):
        self._stamps_lock = threading.Lock()
        self.acquired = array("d")
        self._latency_s = latency_ms / 1000.0
        self._rate = float(kwargs.get("sampling_rate") or 256)
        super().__init__(*args, **kwargs)
        rate = getattr(getattr(self, "_device", None), "sampling_rate", None)
        if rate:
            self._rate = float(rate)

    def start(self):
        with self._stamps_lock:
            self.acquired = array("d")
        return super().start()

    def _data_callback(self, data: np.ndarray):
        arrival = clock.now() - self._latency_s
        n = data.shape[0]
        with self._stamps_lock:
            self.acquired.extend(arrival - (n - 1 - np.arange(n))
                                 / self._rate)
        super()._data_callback(data)

    def stamps(self) -> np.ndarray:
        with self._stamps_lock:
            return np.frombuffer(self.acquired, dtype=float).copy()


class _SimulatedAmplifier:
    """Stands in for gtec_gds's GHIamp: EEG with alpha on the posterior
    channels, background noise and mains, and a digital input pulse every
    two seconds (values 1-4, held for 100 ms), delivered in blocks at the
    amplifier's pace."""

    def __init__(self, serial=None, sampling_rate=256, channel_count=None,
                 frame_size=None, enable_di=None, enable_counter=None,
                 **kwargs):
        self.serial_number = serial or SIMULATED_SERIAL
        self.sampling_rate = int(sampling_rate or 256)
        self.channel_count = int(channel_count or SIMULATED_CHANNELS)
        self.frame_size = int(frame_size or RATES.get(self.sampling_rate, 8))
        self.enable_di = bool(enable_di)
        self.enable_counter = bool(enable_counter)
        self.stream_error = None
        self._callback = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def set_data_callback(self, callback):
        self._callback = callback

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="hiamp-sim")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def close(self):
        self.stop()

    def _run(self):
        rng = np.random.default_rng()
        fs, block = self.sampling_rate, self.frame_size
        ch = self.channel_count
        posterior = np.linspace(0, 1, ch) ** 2
        t0, n = clock.now(), 0
        while self._running:
            due = t0 + (n + block) / fs
            wait = due - clock.now()
            if wait > 0:
                time.sleep(min(wait, 0.05))
                continue
            t = (n + np.arange(block)) / fs
            eeg = (rng.normal(0, 8, (block, ch))
                   + np.outer(15 * np.sin(2 * np.pi * 10 * t), posterior)
                   + np.outer(4 * np.sin(2 * np.pi * 50 * t), np.ones(ch)))
            columns = [eeg]
            if self.enable_di:
                phase = t % 2.0
                value = (np.floor(t / 2.0) % 4 + 1) * (phase < 0.1)
                columns.append(value[:, None])
            n += block
            if self._callback is not None:
                self._callback(np.hstack(columns).astype(np.float32))


class SimulatedGHIamp(StampedGHIamp):
    """g.Pype's GHIamp on the simulated amplifier. GHIamp imports
    gtec_gds when it opens the device; the stand-in is in its place only
    for that, so a real amplifier in the same process is unaffected."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("serial", SIMULATED_SERIAL)
        stand_in = SimpleNamespace(GHIamp=_SimulatedAmplifier)
        real = sys.modules.get("gtec_gds")
        sys.modules["gtec_gds"] = stand_in
        try:
            super().__init__(*args, **kwargs)
        finally:
            if real is None:
                sys.modules.pop("gtec_gds", None)
            else:
                sys.modules["gtec_gds"] = real


# -- the device ------------------------------------------------------------------------

@dataclass
class Graph:
    pipeline: gp.Pipeline
    source: StampedGHIamp
    raw: SweepTap
    scope: SweepTap
    bandpass: EegFilter
    notch: EegFilter
    logger: BidsBdfLogger


class HIampDevice(Device):
    TYPE = "hiamp"
    NAME = "g.tec g.HIamp"
    MANUFACTURER = "g.tec medical engineering GmbH"
    MODEL = "g.HIamp"
    DESCRIPTION = "Wired high-density EEG amplifier, up to 256 channels " \
                  "(g.Pype)"
    MODALITY = "eeg"
    CONFIG = HIampConfig

    def __init__(self, config: HIampConfig, project_folder: Path,
                 simulate: bool = False):
        super().__init__(config, project_folder, simulate)
        self.graph: Optional[Graph] = None
        self._target: Optional[RecordingTarget] = None

    @property
    def rate(self) -> float:
        return float(self.config.sampling_rate)

    # -- acquisition -------------------------------------------------------

    def build(self) -> Graph:
        cfg = self.config
        cls = SimulatedGHIamp if self.simulated else StampedGHIamp
        rate = int(cfg.sampling_rate)
        source = cls(serial=cfg.serial or None, sampling_rate=rate,
                     channel_count=cfg.channel_count or None,
                     frame_size=RATES.get(rate),
                     enable_di=bool(cfg.digital_input),
                     latency_ms=cfg.latency_ms, name="hiamp")
        p = gp.Pipeline()
        raw = SweepTap(seconds=10, name="raw")
        p.connect(source, raw)
        bandpass = EegFilter(presets.filter_sos(cfg.view_filter, rate),
                             name="filter")
        notch = EegFilter(presets.notch_sos(cfg.view_notch, rate),
                          name="notch")
        p.connect(source, bandpass)
        p.connect(bandpass, notch)
        scope = SweepTap(seconds=10, name="scope")
        p.connect(notch, scope)
        logger = BidsBdfLogger(cfg.electrodes, cfg.latency_ms,
                               name="bids_bdf")
        p.connect(source, logger)
        return Graph(p, source, raw, scope, bandpass, notch, logger)

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
            self.graph.bandpass.set_sos(presets.filter_sos(name, self.rate))

    def set_view_notch(self, name: str) -> None:
        self.config.view_notch = name
        if self.graph is not None:
            self.graph.notch.set_sos(presets.notch_sos(name, self.rate))

    def signal_columns(self) -> list[int]:
        graph = self.graph
        if graph is None:
            return []
        return [i for i, r in enumerate(graph.raw.roles) if r == SIGNAL]

    def trigger_column(self) -> Optional[int]:
        graph = self.graph
        if graph is None:
            return None
        roles = graph.raw.roles
        return roles.index(TRIGGER) if TRIGGER in roles else None

    def channel_names(self) -> list[str]:
        names = list(self.config.electrodes or [])
        count = len(self.signal_columns()) or self.config.channel_count \
            or SIMULATED_CHANNELS
        return [names[i] if i < len(names) and names[i] else f"Ch{i + 1}"
                for i in range(count)]

    # -- recording -----------------------------------------------------------

    @property
    def recording(self) -> bool:
        graph = self.graph
        return bool(graph and graph.logger.recording)

    def begin_recording(self, target: RecordingTarget) -> list[Path]:
        with self._lock:
            graph = self.graph
            if graph is None:
                raise RuntimeError("The g.HIamp is not acquiring.")
            path = target.dataset.path(target.entities, target.datatype,
                                       "eeg", ".bdf")
            logger = graph.logger
            logger.subject_id = f"sub-{target.entities.sub}"
            logger.recording_id = target.entities.stem()
            logger.device_serial = self.identity()
            logger.begin(str(path), logger.seen + max(8, int(self.rate / 10)))
            self._target = target
            return [path]

    def end_recording(self) -> RecordedFiles:
        with self._lock:
            graph, target = self.graph, self._target
            if graph is None or target is None or not graph.logger.recording:
                return RecordedFiles()
            logger = graph.logger
            logger.end(logger.seen + max(8, int(self.rate / 10)))
            deadline = time.monotonic() + 2.0
            while logger.recording and time.monotonic() < deadline:
                time.sleep(0.01)
            logger.end()
            self._target = None
            path = Path(logger.path)
            samples = logger.samples_written
            events = list(logger.events)
            labels, roles = list(logger.labels), list(logger.roles)
            timing = self._timing(graph.source, logger, samples)

        def finalize(reference: Optional[Timeline]) -> None:
            self._write_sidecars(target, path, samples, labels, roles,
                                 timing)

        first = timing.timeline.t0
        sidecars = [path.with_name(path.name.replace("_eeg.bdf", s))
                    for s in ("_eeg.json", "_channels.tsv")]
        triggers = [Event(time=timing.timeline.clock(sample),
                          trial_type="trigger", value=f"{value:g}",
                          source="digital input")
                    for _, sample, value in events]
        return RecordedFiles(
            scans=[(path, datetime.fromtimestamp(first))],
            files=[path, *sidecars], first_sample=first, finalize=finalize,
            samples=samples, events=triggers, timeline=timing.timeline)

    def _timing(self, source, logger: BidsBdfLogger,
                samples: int) -> TimingFit:
        nominal = self.rate
        fallback = TimingFit(Timeline(logger.first_arrival or clock.now(),
                                      nominal, nominal), method="arrival")
        start = logger.start_index
        if start is None or samples == 0 or not hasattr(source, "stamps"):
            return fallback
        acquired = source.stamps()
        if start + samples > len(acquired):
            return fallback
        return fit_timing(acquired[start:start + samples], nominal)

    def _write_sidecars(self, target: RecordingTarget, path: Path,
                        samples: int, labels: list[str], roles: list[str],
                        timing: TimingFit) -> None:
        cfg = self.config
        fs = self.rate
        kinds = [("EEG", "uV") if r == SIGNAL else
                 ("TRIG", NA) if r == TRIGGER else ("MISC", NA)
                 for r in roles]
        count = {k: sum(1 for t, _ in kinds if t == k)
                 for k in ("EEG", "MISC", "TRIG")}
        write_json(path.with_name(path.name.replace("_eeg.bdf", "_eeg.json")), {
            "TaskName": target.entities.task,
            "SamplingFrequency": fs,
            "PowerLineFrequency": target.power_line_frequency,
            "SoftwareFilters": NA,
            "HardwareFilters": NA,
            "EEGReference": cfg.reference or NA,
            "EEGGround": cfg.ground or NA,
            "EEGPlacementScheme": cfg.placement or NA,
            "CapManufacturer": cfg.cap_manufacturer or NA,
            "CapManufacturersModelName": cfg.cap_model or NA,
            "Manufacturer": self.MANUFACTURER,
            "ManufacturersModelName": self.MODEL,
            "DeviceSerialNumber": self.identity() or NA,
            "SoftwareVersions": f"brAIn dump {__version__}, "
                                f"g.Pype {gp.__version__}",
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
            **target.common_metadata,
            **target.eeg_metadata,
            "RecordingTiming": timing.metadata(
                "each block stamped as it arrived over USB, less a "
                f"{cfg.latency_ms:g} ms link delay, its samples one period "
                "apart"),
        })
        write_tsv(path.with_name(path.name.replace("_eeg.bdf",
                                                   "_channels.tsv")),
                  ["name", "type", "units", "low_cutoff", "high_cutoff",
                   "sampling_frequency", "status"],
                  [[label, kind, unit, NA, NA, fs, "good"]
                   for label, (kind, unit) in zip(labels, kinds)])

    # -- status ------------------------------------------------------------------

    def identity(self) -> str:
        graph = self.graph
        device = getattr(graph.source, "_device", None) if graph else None
        serial = getattr(device, "serial_number", None)
        if serial:
            return str(serial)
        if self.simulated:
            return SIMULATED_SERIAL
        return self.config.serial or ""

    def status(self) -> str:
        name = self.identity() or "g.HIamp"
        if self.state == RUNNING:
            text = "recording" if self.recording else "acquiring"
            n = len(self.signal_columns())
            return f"{name}: {text}, {n} channels at {self.rate:g} Hz"
        return {IDLE: f"{name}: not connected",
                STARTING: f"{name}: connecting…",
                STOPPING: f"{name}: stopping…"}.get(self.state, self.state)
