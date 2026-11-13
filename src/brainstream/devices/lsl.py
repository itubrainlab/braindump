"""Lab Streaming Layer streams, recorded as BIDS events.

Each configured stream is matched by name, type, and/or source id (an
empty field matches anything), and found on the network whenever it
appears: a stimulus program may start after brAIn stream. Every sample a
stream sends while recording becomes an event: onset from the sample's
LSL time stamp (corrected for the sender's clock, then moved onto this
computer's clock), ``trial_type`` the stream's name, ``value`` the
sample. The recording session merges them with the other devices'
events into the run's ``_events.tsv``.

The samples enter the session pipeline through a :class:`MarkerSource`::

    MarkerSource ── EventLog   (the events, while recording)

g.Pype's own LslReceiver does not fit marker streams: g.Pype carries
float32, and markers are usually strings; it drops LSL's time stamps; and
it gives up when a stream is not on the network within seconds, while a
stimulus program may start long after brAIn stream. So the streams are
pulled with pylsl, and each sample enters the pipeline as a row: its time
on brAIn stream's clock, and a code standing for its stream and text
(:attr:`LslDevice.codes`).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import gpype as gp

from .. import clock
from ..pipeline import EventLog, PushSource
from .base import (IDLE, RUNNING, STARTING, STOPPING, Device, DeviceConfig,
                   Event, RecordedFiles, RecordingTarget)

#: How often streams not yet found are looked for, in seconds.
RESOLVE_EVERY_S = 2.0
#: How often a stream's clock offset is measured again, in seconds.
CLOCK_EVERY_S = 5.0
SIMULATED_STREAM = "brAIn stream test markers"


@dataclass
class LslConfig(DeviceConfig):
    #: Streams to record: dicts with "name", "type" and "source_id";
    #: empty fields match any stream.
    streams: list[dict] = field(default_factory=lambda: [
        {"name": "", "type": "Markers", "source_id": ""}])
    simulate: bool = False
    #: Seconds a resolve waits for answers.
    resolve_timeout: float = 1.0


def describe(selector: dict) -> str:
    parts = [f"{k}={selector[k]}" for k in ("name", "type", "source_id")
             if selector.get(k)]
    return ", ".join(parts) or "any stream"


def matches(selector: dict, info) -> bool:
    for key, value in (("name", info.name()), ("type", info.type()),
                       ("source_id", info.source_id())):
        wanted = (selector.get(key) or "").strip()
        if wanted and wanted != value:
            return False
    return True


def find_streams(timeout: float = 1.0) -> list[dict]:
    """The streams on the network, described for the user."""
    import pylsl
    out = []
    for info in pylsl.resolve_streams(timeout):
        out.append({"name": info.name(), "type": info.type(),
                    "source_id": info.source_id(),
                    "channels": info.channel_count(),
                    "rate": info.nominal_srate(),
                    "format": _FORMATS.get(info.channel_format(), "?"),
                    "host": info.hostname(), "uid": info.uid()})
    return out


_FORMATS = {1: "float32", 2: "double64", 3: "string", 4: "int32",
            5: "int16", 6: "int8", 7: "int64"}


class MarkerSource(PushSource):
    """The LSL samples, one row each: ``[time, code]``. Irregular: the
    declared rate (one a second) only describes the stream to g.Pype."""

    def __init__(self, **kwargs):
        super().__init__(["CODE"], [""], 1.0, **kwargs)


@dataclass
class Graph:
    """The LSL device's nodes in the session pipeline."""
    source: MarkerSource
    log: EventLog


class _Inlet:
    """One connected stream, pulled on its own thread.

    An inlet keeps to the one outlet it was opened on. LSL's recovery
    would re-attach it, once its outlet is gone, to any new outlet with
    the same name and source_id (a procedure run again, say), while the
    resolver opens an inlet on that new outlet too: every marker would
    then be recorded twice. So when its outlet is lost, the inlet closes,
    and the resolver opens one on whatever stream comes next."""

    def __init__(self, device: "LslDevice", info):
        import pylsl
        self.device = device
        self.info = info
        self.name = info.name()
        self.uid = info.uid()
        self.inlet = pylsl.StreamInlet(info, max_buflen=360,
                                       recover=False)
        self.correction = 0.0
        self._corrected_at = 0.0
        self.samples = 0
        self.running = True
        self.thread = threading.Thread(target=self._pull, daemon=True,
                                       name=f"lsl-{self.name}")
        self.thread.start()

    def _pull(self):
        # pylsl 1.16 exported LostError at the top; 1.17 and later keep it
        # in pylsl.util only.
        try:
            from pylsl.util import LostError
        except ImportError:  # pragma: no cover - older pylsl
            from pylsl import LostError
        while self.running:
            now = time.monotonic()
            if now - self._corrected_at > CLOCK_EVERY_S:
                try:
                    self.correction = self.inlet.time_correction(timeout=1.0)
                    self._corrected_at = now
                except Exception:  # noqa: BLE001 - keep the last one
                    pass
            try:
                chunk, stamps = self.inlet.pull_chunk(timeout=0.2)
            except LostError:
                # The outlet is gone: the resolver connects again when a
                # stream (this one back, or a new one) appears.
                if self.running:
                    self.device._inlet_failed(self, "stream lost")
                self._close_inlet()
                return
            except Exception as error:  # noqa: BLE001
                if self.running:
                    self.device._inlet_failed(self, str(error))
                return
            if not stamps:
                continue
            offset = self.device.clock_offset()
            for sample, stamp in zip(chunk, stamps):
                self.samples += 1
                value = sample[0] if len(sample) == 1 else \
                    ",".join(str(v) for v in sample)
                if isinstance(value, float) and value.is_integer():
                    value = int(value)
                self.device._event(Event(
                    time=stamp + self.correction + offset,
                    trial_type=self.name, value=str(value),
                    source=f"lsl:{self.name}"))

    def close(self):
        self.running = False
        if self.thread is not threading.current_thread():
            self.thread.join(timeout=1.0)
        self._close_inlet()

    def _close_inlet(self):
        try:
            self.inlet.close_stream()
        except Exception:  # noqa: BLE001
            pass


class LslDevice(Device):
    TYPE = "lsl"
    NAME = "LSL streams"
    MANUFACTURER = "Lab Streaming Layer"
    DESCRIPTION = "Markers (or any stream) from the network, recorded as " \
                  "BIDS events"
    MODALITY = "events"
    CONFIG = LslConfig

    def __init__(self, config: LslConfig, project_folder: Path,
                 simulate: bool = False):
        super().__init__(config, project_folder, simulate)
        self.inlets: dict[str, _Inlet] = {}
        self.graph: Optional[Graph] = None
        #: The latest events, newest last, for the view.
        self.log: deque[Event] = deque(maxlen=500)
        self.received = 0
        #: What each code in the pipeline stands for: (trial_type, value,
        #: source), in the order first seen.
        self.codes: list[tuple[str, str, str]] = []
        self._code_of: dict[tuple[str, str, str], int] = {}
        self._target: Optional[RecordingTarget] = None
        self._events_lock = threading.Lock()
        self._running = False
        self._resolver: Optional[threading.Thread] = None
        #: Counts the connections: a resolver from an earlier one that
        #: outlived its join stops instead of opening inlets of its own.
        self._generation = 0
        self._inlets_lock = threading.Lock()
        self._outlet = None
        self._sim_thread: Optional[threading.Thread] = None
        self._offset = 0.0
        self._offset_at = 0.0

    # -- clocks ------------------------------------------------------------

    def clock_offset(self) -> float:
        """brAIn stream's clock minus LSL's. Both are steady (LSL reads the
        system's monotonic clock), so the difference holds; it is still
        measured again every few seconds, between two readings of
        brAIn stream's clock, to stay within a few microseconds."""
        import pylsl
        now = time.monotonic()
        if now - self._offset_at > CLOCK_EVERY_S:
            before = clock.now()
            lsl = pylsl.local_clock()
            after = clock.now()
            self._offset = (before + after) / 2 - lsl
            self._offset_at = now
        return self._offset

    # -- the session pipeline ----------------------------------------------------

    def build(self, pipeline: gp.Pipeline) -> None:
        import pylsl  # noqa: F401 - fail here if liblsl is missing
        source = MarkerSource(name=self.node_name("source"))
        log = EventLog(self.node_name("events"))
        pipeline.connect(source, log)
        self.graph = Graph(source, log)

    def opened(self) -> None:
        self._offset_at = 0.0
        self.clock_offset()
        if self.simulated:
            self._start_simulation()
        self._running = True
        self._generation += 1
        self._resolver = threading.Thread(target=self._resolve,
                                          args=(self._generation,),
                                          daemon=True, name="lsl-resolve")
        self._resolver.start()

    def closing(self) -> None:
        self._running = False
        if self._resolver:
            self._resolver.join(timeout=3.0)
            self._resolver = None
        with self._inlets_lock:
            inlets, self.inlets = list(self.inlets.values()), {}
        for inlet in inlets:
            inlet.close()
        self._outlet = None
        if self._sim_thread:
            self._sim_thread.join(timeout=1.0)
            self._sim_thread = None

    def closed(self) -> None:
        self.graph = None
        self._target = None

    def _selectors(self) -> list[dict]:
        selectors = [s for s in self.config.streams if isinstance(s, dict)]
        if self.simulated:
            selectors = [{"name": SIMULATED_STREAM}]
        return selectors

    def _resolve(self, generation: int):
        import pylsl

        def current() -> bool:
            return self._running and generation == self._generation
        while current():
            try:
                found = pylsl.resolve_streams(self.config.resolve_timeout)
            except Exception:  # noqa: BLE001
                found = []
            for info in found:
                if not current():
                    break
                if not any(matches(s, info) for s in self._selectors()):
                    continue
                # Checked and opened under the lock: one inlet a stream.
                with self._inlets_lock:
                    if not current() or info.uid() in self.inlets:
                        continue
                    try:
                        self.inlets[info.uid()] = _Inlet(self, info)
                    except Exception:  # noqa: BLE001
                        pass
            deadline = time.monotonic() + RESOLVE_EVERY_S
            while current() and time.monotonic() < deadline:
                time.sleep(0.1)

    def _inlet_failed(self, inlet: _Inlet, message: str) -> None:
        with self._inlets_lock:
            if self.inlets.get(inlet.uid) is inlet:
                del self.inlets[inlet.uid]

    def _event(self, event: Event) -> None:
        with self._events_lock:
            self.received += 1
            self.log.append(event)
            key = (event.trial_type, event.value, event.source)
            code = self._code_of.get(key)
            if code is None:
                code = self._code_of[key] = len(self.codes)
                self.codes.append(key)
        graph = self.graph
        if graph is not None:
            graph.source.push([event.time, code])

    # -- simulation ----------------------------------------------------------------

    def _start_simulation(self):
        import pylsl
        info = pylsl.StreamInfo(SIMULATED_STREAM, "Markers", 1, 0,
                                "string", "brainstream-sim-markers")
        self._outlet = pylsl.StreamOutlet(info)

        def send():
            n = 0
            labels = ["fixation", "stimulus", "response"]
            while self._outlet is not None:
                time.sleep(1.0)
                outlet = self._outlet
                if outlet is not None:
                    outlet.push_sample([f"{labels[n % 3]}"])
                    n += 1
        self._sim_thread = threading.Thread(target=send, daemon=True)
        self._sim_thread.start()

    # -- recording ---------------------------------------------------------------------

    @property
    def recording(self) -> bool:
        graph = self.graph
        return bool(graph and graph.log.recording)

    def begin_recording(self, target: RecordingTarget) -> list[Path]:
        with self._lock:
            graph = self.graph
            if graph is None:
                raise RuntimeError("LSL is not started.")
            self.invoke(graph.log, "begin",
                        {"meta": {"epoch": graph.source.epoch}})
            self._target = target
        return []

    def end_recording(self) -> RecordedFiles:
        with self._lock:
            graph, target = self.graph, self._target
            if graph is None or target is None:
                return RecordedFiles()
            # Samples sent just before now may still be on their way.
            time.sleep(0.3)
            self.invoke(graph.log, "end")
            self._target = None
            rows = list(graph.log.rows)
        with self._events_lock:
            codes = list(self.codes)
        events = []
        for t, values in rows:
            # Sent before Record was pressed, but arrived after: not this
            # recording's.
            if t < target.started:
                continue
            trial_type, value, source = codes[int(round(values[0]))]
            events.append(Event(time=t, trial_type=trial_type, value=value,
                                source=source))
        return RecordedFiles(events=sorted(events, key=lambda e: e.time),
                             samples=len(events))

    # -- status -----------------------------------------------------------------------------

    def identity(self) -> str:
        n = len(self.inlets)
        if self.state != RUNNING:
            return ""
        return f"{n} stream{'s' if n != 1 else ''}"

    def status(self) -> str:
        if self.state == RUNNING:
            names = ", ".join(sorted(i.name for i in self.inlets.values()))
            text = "recording" if self.recording else "listening"
            return f"LSL: {text}, {names or 'no stream found yet'}"
        return {IDLE: "LSL: not started", STARTING: "LSL: starting…",
                STOPPING: "LSL: stopping…"}.get(self.state, self.state)
