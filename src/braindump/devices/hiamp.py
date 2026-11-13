"""The g.tec g.HIamp, through g.Pype.

g.Pype's :class:`~gpype.backend.sources.g_hiamp.GHIamp` drives the
amplifier with g.tec's gtec-gds driver (Windows, Linux and macOS)::

    GHIamp ─┬─ SweepTap "raw"                       (raw view)
            ├─ Filter ─ Notch ─ SweepTap "scope"    (filtered view)
            └─ BdfWriter                            (BIDS, while recording)

The EEG is recorded through the **amplifier's own filters**: a band-pass
and a notch, chosen in the settings from those the driver lists for the
sampling rate (loaded from the amplifier, or listed as it connects),
applied in the amplifier and described in the sidecars
(``HardwareFilters``). The Filter and Notch boxes only shape the view. The digital input (TRIG)
is recorded beside the EEG, unfiltered: its rising edges are the run's
trigger events. Each block of
samples is stamped as it arrives, so the recording's timing is measured
as for the Unicorn: its first sample's time and the amplifier's real
rate, the time base the other devices are given in.

A simulated amplifier (``simulate``) runs g.Pype's real GHIamp source on
a stand-in for the driver, as the Unicorn is simulated.
"""
from __future__ import annotations

import functools
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

from .. import __version__, clock
from ..bids import NA, write_json, write_tsv
from .base import (IDLE, RUNNING, STARTING, STOPPING, Device, DeviceConfig,
                   Event, RecordedFiles, RecordingTarget, Timeline)
from .. import presets
from ..pipeline import BdfWriter, SweepTap, ViewFilter
from .eeg import (SIGNAL, TRIGGER, TimingFit,
                  channel_filter_columns, fit_timing)

#: Sampling rates offered, and the block size (scans) the driver
#: recommends for each (the simulated amplifier uses them too).
RATES = {256: 8, 512: 16, 600: 20, 1200: 40, 2400: 80, 4800: 160}
SIMULATED_SERIAL = "HA-SIMULATED"
SIMULATED_CHANNELS = 64
#: No amplifier filter (the driver's index for none).
NO_FILTER = -1
#: The BIDS types a channel other than EEG may have.
CHANNEL_TYPES = ("EOG", "ECG", "EMG", "MISC")


def _field(entry: dict, *words: str, default=None):
    """The value of the first key of a driver filter entry that holds
    every one of ``words`` (the driver's keys, e.g. LowerCutoffFrequency,
    are matched loosely)."""
    for key, value in entry.items():
        if all(w.lower() in str(key).lower() for w in words):
            return value
    return default


def amplifier_filter(entry: dict, kind: str) -> dict:
    """One of the driver's filters, as brAIn dump keeps it: its index,
    cut-offs (0 for none), order and type."""
    index = _field(entry, kind, "index")
    if index is None:
        index = _field(entry, "index", default=NO_FILTER)
    return {"index": int(index),
            "low": float(_field(entry, "lower", default=0) or 0),
            "high": float(_field(entry, "upper", default=0) or 0),
            "order": int(_field(entry, "order", default=0) or 0),
            "type": str(_field(entry, "type", default="") or "")}


def default_bandpass(offered: list[dict], fs: float) -> Optional[dict]:
    """The default band-pass among the amplifier's: a 0.1 Hz high-pass
    with no low-pass (or one at half the sampling rate), else the
    band-pass from 0.1 Hz with the highest low-pass; None if neither is
    offered."""
    from_01 = [f for f in offered if abs(f["low"] - 0.1) < 1e-6]
    open_top = [f for f in from_01 if not f["high"] or f["high"] >= fs / 2]
    if open_top:
        return open_top[0]
    return max(from_01, key=lambda f: f["high"], default=None)


def describe_amplifier_filter(f: dict, notch: bool = False) -> str:
    """How a settings list shows an amplifier filter."""
    low, high = f["low"], f["high"]
    if notch:
        text = f"{(low + high) / 2:g} Hz ({low:g} – {high:g} Hz)"
    elif low and high:
        text = f"{low:g} – {high:g} Hz"
    elif low:
        text = f"> {low:g} Hz"
    elif high:
        text = f"< {high:g} Hz"
    else:
        text = "no cut-off"
    if f["order"]:
        text += f", order {f['order']}"
    return text


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
    #: Channels that are not EEG, as "<channel number>=<BIDS type>", e.g.
    #: "64=EOG" for a bipolar EOG lead on channel 64. Types: EOG, ECG,
    #: EMG, MISC.
    channel_types: list[str] = field(default_factory=list)
    #: Record the digital input (TRIG).
    digital_input: bool = True
    #: Delay of the USB link, in ms, taken off the time stamps.
    latency_ms: float = 0.0
    reference: str = ""
    ground: str = ""
    placement: str = "10-20"
    cap_manufacturer: str = "g.tec"
    cap_model: str = ""
    #: The amplifier's own band-pass and notch: the driver's filter
    #: indices, NO_FILTER for none. Until chosen, loading the filters
    #: picks the default (see default_bandpass).
    amp_bandpass: int = NO_FILTER
    amp_notch: int = NO_FILTER
    #: The filters the amplifier offers, as it listed them when last
    #: connected, by sampling rate: {"256": {"bandpass": [..], "notch":
    #: [..]}} (see amplifier_filter).
    amp_filters: dict = field(default_factory=dict)
    # The view (not recorded)
    view_filter: str = presets.DEFAULT_FILTER
    view_notch: str = presets.DEFAULT_NOTCH
    view_scaling: str = presets.DEFAULT_SCALING

    def types_by_channel(self) -> dict[int, str]:
        """The channels that are not EEG: 0-based number -> BIDS type."""
        found = {}
        for entry in self.channel_types or []:
            number, _, kind = str(entry).partition("=")
            kind = kind.strip().upper()
            try:
                index = int(number.strip()) - 1
            except ValueError:
                continue
            if index >= 0 and kind in CHANNEL_TYPES:
                found[index] = kind
        return found

    def known_filters(self, kind: str) -> list[dict]:
        """The amplifier's filters of ``kind`` (bandpass or notch) at the
        sampling rate, as last listed; empty when never connected."""
        return list((self.amp_filters or {})
                    .get(str(int(self.sampling_rate)), {}).get(kind, []))

    def learn_filters(self, found: dict[str, list[dict]]) -> None:
        """Keeps the filters the amplifier offers at the sampling rate.
        The first time a rate's filters are known, the band-pass is set
        to the default (see :func:`default_bandpass`)."""
        if not any(found.values()):
            return
        rate = str(int(self.sampling_rate))
        known = dict(self.amp_filters or {})
        first = rate not in known
        known[rate] = found
        self.amp_filters = known
        if first and int(self.amp_bandpass) == NO_FILTER:
            default = default_bandpass(found.get("bandpass", []),
                                       float(self.sampling_rate))
            if default is not None:
                self.amp_bandpass = default["index"]

    def chosen_filter(self, kind: str) -> Optional[dict]:
        """The amplifier filter chosen of ``kind``; None for none."""
        index = int(self.amp_bandpass if kind == "bandpass"
                    else self.amp_notch)
        if index == NO_FILTER:
            return None
        for f in self.known_filters(kind):
            if f["index"] == index:
                return f
        raise ValueError(
            f"The g.HIamp's {'band-pass' if kind == 'bandpass' else 'notch'}"
            f" filter #{index} is not one it offers at "
            f"{int(self.sampling_rate)} Hz: choose another in the "
            "settings.")


# -- the source ------------------------------------------------------------------------

class StampedGHIamp(GHIamp):
    """g.Pype's GHIamp, keeping every sample's acquisition time.

    The driver hands over a block when its last sample is in; the samples
    before it were acquired one period apart before that. g.Pype stamps
    the block for its own synchronisation and drops the stamp before any
    sink; this keeps the times."""

    def __init__(self, *args, latency_ms: float = 0.0,
                 bandpass_index: int = NO_FILTER,
                 notch_index: int = NO_FILTER, **kwargs):
        self._stamps_lock = threading.Lock()
        self.acquired = array("d")
        self._latency_s = latency_ms / 1000.0
        self._rate = float(kwargs.get("sampling_rate") or 256)
        filters = {k: v for k, v in (("bandpass_filter_index", bandpass_index),
                                     ("notch_filter_index", notch_index))
                   if v is not None and v != NO_FILTER}
        with _driver_filters(filters):
            super().__init__(*args, **kwargs)
        rate = getattr(getattr(self, "_device", None), "sampling_rate", None)
        if rate:
            self._rate = float(rate)

    def amplifier_filters(self) -> dict[str, list[dict]]:
        """The band-pass and notch filters the amplifier offers at its
        sampling rate."""
        device = getattr(self, "_device", None)
        found = {}
        for kind, method in (("bandpass", "get_bandpass_filters"),
                             ("notch", "get_notch_filters")):
            try:
                entries = getattr(device, method)() or []
            except Exception:  # noqa: BLE001
                entries = []
            found[kind] = sorted(
                (amplifier_filter(dict(e), kind) for e in entries),
                key=lambda f: (f["low"], f["high"], f["index"]))
        return found

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


class _driver_filters:
    """While g.Pype opens the amplifier, the driver's GHIamp is given the
    filter indices: g.Pype's GHIamp does not pass them on, and the driver
    sets them on every channel as it opens."""

    def __init__(self, filters: dict):
        self.filters = filters
        self.module = None

    def __enter__(self):
        if not self.filters:
            return self
        module = sys.modules.get("gtec_gds")
        if module is None:
            try:
                import gtec_gds as module
            except ImportError:
                return self  # g.Pype reports the missing driver
        self.module, self.real = module, module.GHIamp
        module.GHIamp = functools.partial(self.real, **self.filters)
        return self

    def __exit__(self, *exc):
        if self.module is not None:
            self.module.GHIamp = self.real


#: The simulated amplifier's filters, numbered across sampling rates as
#: the driver numbers them: 4th-order Butterworth high-passes and
#: band-passes, and notches for 50 and 60 Hz.
_SIMULATED_BANDPASS = ((0.01, 0), (0.1, 0), (0.5, 0), (1, 0), (2, 0),
                       (0.1, 30), (0.5, 30), (1, 30), (0.1, 100), (1, 100),
                       (0.1, 200), (0.1, 1000), (0.01, 100))
_SIMULATED_NOTCH = ((48, 52), (58, 62))


def _simulated_filters(kind: str, rate: int) -> list[dict]:
    table = _SIMULATED_BANDPASS if kind == "bandpass" else _SIMULATED_NOTCH
    key = "BandpassFilterIndex" if kind == "bandpass" else "NotchFilterIndex"
    found = []
    for r, rate_ in enumerate(RATES):
        for i, (low, high) in enumerate(table):
            if rate_ == rate and high < rate / 2:
                found.append({key: r * len(table) + i,
                              "Order": 4 if kind == "bandpass" else 2,
                              "LowerCutoffFrequency": float(low),
                              "UpperCutoffFrequency": float(high),
                              "SamplingRate": float(rate),
                              "TypeId": "Butterworth" if kind == "bandpass"
                              else "Notch"})
    return found


class _SimulatedAmplifier:
    """Stands in for gtec_gds's GHIamp: EEG with alpha on the posterior
    channels, background noise and mains, and a digital input pulse every
    two seconds (values 1-4, held for 100 ms), delivered in blocks at the
    amplifier's pace."""

    def __init__(self, serial=None, sampling_rate=256, channel_count=None,
                 frame_size=None, enable_di=None, enable_counter=None,
                 bandpass_filter_index=None, notch_filter_index=None,
                 channel_selection=None, **kwargs):
        self.serial_number = serial or SIMULATED_SERIAL
        self.sampling_rate = int(sampling_rate or 256)
        # g.Pype 4.2 asks for channels by selection, and reads it back;
        # 4.1 by count.
        if channel_selection is None:
            channel_selection = range(int(channel_count
                                          or SIMULATED_CHANNELS))
        self.channel_selection = [int(c) for c in channel_selection]
        self.channel_count = len(self.channel_selection)
        self.frame_size = int(frame_size or RATES.get(self.sampling_rate, 8))
        self.enable_di = bool(enable_di)
        self.enable_counter = bool(enable_counter)
        self.stream_error = None
        self._sos = self._filter_sos(bandpass_filter_index,
                                     notch_filter_index)
        self._callback = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def get_bandpass_filters(self):
        return _simulated_filters("bandpass", self.sampling_rate)

    def get_notch_filters(self):
        return _simulated_filters("notch", self.sampling_rate)

    def _filter_sos(self, bandpass, notch):
        """The amplifier's filters, applied to the simulated EEG."""
        from scipy import signal
        fs, sections = self.sampling_rate, []
        for kind, index in (("bandpass", bandpass), ("notch", notch)):
            if index is None or index == NO_FILTER:
                continue
            entry = next((amplifier_filter(e, kind)
                          for e in _simulated_filters(kind, fs)
                          if amplifier_filter(e, kind)["index"] == index),
                         None)
            if entry is None:
                raise ValueError(f"No {kind} filter {index} at {fs} Hz")
            low, high = entry["low"], entry["high"]
            if kind == "notch":
                b, a = signal.iirnotch((low + high) / 2, 30, fs=fs)
                sections.append(signal.tf2sos(b, a))
            elif low and high:
                sections.append(signal.butter(4, [low, high], "bandpass",
                                              fs=fs, output="sos"))
            else:
                sections.append(signal.butter(4, low or high,
                                              "highpass" if low else
                                              "lowpass", fs=fs,
                                              output="sos"))
        return np.vstack(sections) if sections else None

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
        state = None
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
            if self._sos is not None:
                from scipy import signal
                if state is None:
                    state = np.zeros((self._sos.shape[0], 2, ch))
                eeg, state = signal.sosfilt(self._sos, eeg, axis=0, zi=state)
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
        # Its own serial even when the settings give none: without one,
        # g.Pype (4.2) asks g.tec's device service which unit to open.
        kwargs["serial"] = kwargs.get("serial") or SIMULATED_SERIAL
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

    def _remember_serial(self, serial) -> None:
        """Not g.Pype's favourite g.HIamp: that is for real units."""


def _close(source) -> None:
    """Releases the amplifier a source opened, without a pipeline: g.Pype's
    own release, which the GDS sources have, else the driver's close."""
    release = getattr(source, "_release_device", None)
    if callable(release):
        release()
        return
    device = getattr(source, "_device", None)
    source._device = None
    if device is not None:
        device.close()


# -- the device ------------------------------------------------------------------------

@dataclass
class Graph:
    """The amplifier's nodes in the session pipeline."""
    source: StampedGHIamp
    raw: SweepTap
    scope: SweepTap
    bandpass: ViewFilter
    notch: ViewFilter
    writer: BdfWriter
    #: The amplifier's band-pass and notch in use (None for none).
    amp_bandpass: Optional[dict] = None
    amp_notch: Optional[dict] = None


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

    # -- the session pipeline ---------------------------------------------

    def build(self, p: gp.Pipeline) -> None:
        """g.Pype's GHIamp opens the amplifier as it is built."""
        cfg = self.config
        rate = int(cfg.sampling_rate)
        # Checked before the amplifier opens: a filter that cannot apply
        # stops the connection rather than record something else.
        amp_bandpass = cfg.chosen_filter("bandpass")
        amp_notch = cfg.chosen_filter("notch")
        source = self._open(
            bandpass_index=amp_bandpass["index"] if amp_bandpass
            else NO_FILTER,
            notch_index=amp_notch["index"] if amp_notch else NO_FILTER)
        before = cfg.amp_bandpass
        cfg.learn_filters(source.amplifier_filters())
        if cfg.amp_bandpass != before:
            # The filters were first listed as it opened, and the default
            # band-pass chosen: open it again with that.
            _close(source)
            amp_bandpass = cfg.chosen_filter("bandpass")
            source = self._open(
                bandpass_index=amp_bandpass["index"],
                notch_index=amp_notch["index"] if amp_notch else NO_FILTER)
        raw = SweepTap(seconds=10, name=self.node_name("raw"))
        p.connect(source, raw)
        bandpass = ViewFilter(presets.filter_sos(cfg.view_filter, rate),
                             name=self.node_name("view_filter"))
        notch = ViewFilter(presets.notch_sos(cfg.view_notch, rate),
                          name=self.node_name("view_notch"))
        p.connect(source, bandpass)
        p.connect(bandpass, notch)
        scope = SweepTap(seconds=10, name=self.node_name("scope"))
        p.connect(notch, scope)
        writer = BdfWriter(self.node_name("bids"))
        p.connect(source, writer)
        self.graph = Graph(source, raw, scope, bandpass, notch, writer,
                           amp_bandpass, amp_notch)

    def closed(self) -> None:
        self.graph = None
        self._target = None

    def _open(self, **filters) -> StampedGHIamp:
        """g.Pype's GHIamp source on the amplifier (or the simulated
        one), opened with the settings' rate and channels."""
        cfg = self.config
        cls = SimulatedGHIamp if self.simulated else StampedGHIamp
        rate = int(cfg.sampling_rate)
        return cls(serial=cfg.serial or None, sampling_rate=rate,
                   channel_selection=(range(int(cfg.channel_count))
                                      if cfg.channel_count else None),
                   frame_size=RATES.get(rate),
                   enable_di=bool(cfg.digital_input),
                   latency_ms=cfg.latency_ms,
                   name=self.node_name("source"), **filters)

    def load_filters(self) -> dict[str, list[dict]]:
        """Asks the amplifier which filters it offers at the sampling
        rate, and keeps them in the configuration: from the running
        source when connected, else by opening the amplifier briefly
        (without acquiring)."""
        with self._lock:
            graph = self.graph
            if graph is not None:
                found = graph.source.amplifier_filters()
            elif self.state != IDLE:
                raise RuntimeError("The g.HIamp is connecting or stopping.")
            else:
                source = self._open()
                try:
                    found = source.amplifier_filters()
                finally:
                    _close(source)
            if not any(found.values()):
                raise RuntimeError("The amplifier listed no filters at "
                                   f"{int(self.config.sampling_rate)} Hz.")
            self.config.learn_filters(found)
            return found

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
        return bool(graph and graph.writer.recording)

    def begin_recording(self, target: RecordingTarget) -> list[Path]:
        with self._lock:
            graph = self.graph
            if graph is None:
                raise RuntimeError("The g.HIamp is not acquiring.")
            path = target.dataset.path(target.entities, target.datatype,
                                       "eeg", ".bdf")
            self.invoke(graph.writer, "begin", {
                "path": str(path),
                "meta": {"electrodes": list(self.config.electrodes or []),
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
            labels, roles = list(writer.file_labels), list(writer.roles)
            timing = self._timing(graph.source, writer, samples)
            filters = (graph.amp_bandpass, graph.amp_notch)

        def finalize(reference: Optional[Timeline]) -> None:
            self._write_sidecars(target, path, samples, labels, roles,
                                 timing, *filters)

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

    def _timing(self, source, writer: BdfWriter,
                samples: int) -> TimingFit:
        nominal = self.rate
        fallback = TimingFit(Timeline(clock.now() - samples / nominal,
                                      nominal, nominal), method="arrival")
        start = writer.start_index
        if start is None or samples == 0 or not hasattr(source, "stamps"):
            return fallback
        acquired = source.stamps()
        if start + samples > len(acquired):
            return fallback
        return fit_timing(acquired[start:start + samples], nominal)

    def _write_sidecars(self, target: RecordingTarget, path: Path,
                        samples: int, labels: list[str], roles: list[str],
                        timing: TimingFit, amp_bandpass: Optional[dict],
                        amp_notch: Optional[dict]) -> None:
        cfg = self.config
        fs = self.rate
        # The signal channels in order: EEG, unless typed otherwise
        # (e.g. a bipolar EOG lead).
        typed = cfg.types_by_channel()
        kinds, signal_n = [], 0
        for r in roles:
            if r == SIGNAL:
                kinds.append((typed.get(signal_n, "EEG"), "uV"))
                signal_n += 1
            else:
                kinds.append(("TRIG", NA) if r == TRIGGER else ("MISC", NA))
        count = {k: sum(1 for t, _ in kinds if t == k)
                 for k in ("EEG", "EOG", "ECG", "EMG", "MISC", "TRIG")}
        write_json(path.with_name(path.name.replace("_eeg.bdf", "_eeg.json")), {
            "TaskName": target.entities.task,
            "SamplingFrequency": fs,
            "PowerLineFrequency": target.power_line_frequency,
            "SoftwareFilters": NA,
            "HardwareFilters": self._hardware_filters(amp_bandpass,
                                                      amp_notch),
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
            "EOGChannelCount": count["EOG"],
            "ECGChannelCount": count["ECG"],
            "EMGChannelCount": count["EMG"],
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
        # The amplifier's filters act on the EEG channels.
        filtered = channel_filter_columns(
            amp_bandpass["low"] if amp_bandpass else None,
            amp_bandpass["high"] if amp_bandpass else None,
            [(amp_notch["low"] + amp_notch["high"]) / 2] if amp_notch
            else [])
        write_tsv(path.with_name(path.name.replace("_eeg.bdf",
                                                   "_channels.tsv")),
                  ["name", "type", "units", "low_cutoff", "high_cutoff",
                   "notch", "sampling_frequency", "status"],
                  [[label, kind, unit,
                    *(filtered if unit == "uV" else (NA, NA, NA)), fs,
                    "good"]
                   for label, (kind, unit) in zip(labels, kinds)])

    @staticmethod
    def _hardware_filters(bandpass: Optional[dict], notch: Optional[dict]):
        """The eeg.json's HardwareFilters: the amplifier's own filters in
        use, or n/a."""
        found = {}
        for name, f in (("Amplifier band-pass", bandpass),
                        ("Amplifier notch", notch)):
            if f:
                found[name] = {
                    "FilterIndex": f["index"],
                    "LowerCutoffFrequency": f["low"] or NA,
                    "UpperCutoffFrequency": f["high"] or NA,
                    "Order": f["order"] or NA, "Type": f["type"] or NA,
                    "Channels": "every signal channel"}
        return found or NA

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
