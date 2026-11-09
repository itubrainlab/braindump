"""Lab Streaming Layer streams, recorded as BIDS events.

Each configured stream is matched by name, type, and/or source id (an
empty field matches anything), and found on the network whenever it
appears: a stimulus program may start after brAIn dump. Every sample a
stream sends while recording becomes an event: onset from the sample's
LSL time stamp (corrected for the sender's clock, then moved onto this
computer's clock), ``trial_type`` the stream's name, ``value`` the
sample. The recording session merges them with the other devices'
events into the run's ``_events.tsv``.

pylsl is used directly rather than g.Pype's LslReceiver, because g.Pype
carries float32 samples and marker streams are usually strings.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .. import clock
from .base import (IDLE, RUNNING, STARTING, STOPPING, Device, DeviceConfig,
                   Event, RecordedFiles, RecordingTarget)

#: How often streams not yet found are looked for, in seconds.
RESOLVE_EVERY_S = 2.0
#: How often a stream's clock offset is measured again, in seconds.
CLOCK_EVERY_S = 5.0
SIMULATED_STREAM = "brAIn dump test markers"


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


class _Inlet:
    """One connected stream, pulled on its own thread."""

    def __init__(self, device: "LslDevice", info):
        import pylsl
        self.device = device
        self.info = info
        self.name = info.name()
        self.uid = info.uid()
        self.inlet = pylsl.StreamInlet(info, max_buflen=360,
                                       recover=True)
        self.correction = 0.0
        self._corrected_at = 0.0
        self.samples = 0
        self.running = True
        self.thread = threading.Thread(target=self._pull, daemon=True,
                                       name=f"lsl-{self.name}")
        self.thread.start()

    def _pull(self):
        import pylsl
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
            except pylsl.LostError:
                continue
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
        self.thread.join(timeout=1.0)
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
        #: The latest events, newest last, for the view.
        self.log: deque[Event] = deque(maxlen=500)
        self.received = 0
        self._recorded: list[Event] = []
        self._record_from: Optional[float] = None
        self._events_lock = threading.Lock()
        self._running = False
        self._resolver: Optional[threading.Thread] = None
        self._outlet = None
        self._sim_thread: Optional[threading.Thread] = None
        self._offset = 0.0
        self._offset_at = 0.0

    # -- clocks ------------------------------------------------------------

    def clock_offset(self) -> float:
        """brAIn dump's clock minus LSL's. Both are steady (LSL reads the
        system's monotonic clock), so the difference holds; it is still
        measured again every few seconds, between two readings of
        brAIn dump's clock, to stay within a few microseconds."""
        import pylsl
        now = time.monotonic()
        if now - self._offset_at > CLOCK_EVERY_S:
            before = clock.now()
            lsl = pylsl.local_clock()
            after = clock.now()
            self._offset = (before + after) / 2 - lsl
            self._offset_at = now
        return self._offset

    # -- acquisition -----------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self.state != IDLE:
                return
            self.error = None
            self._set_state(STARTING)
            try:
                import pylsl  # noqa: F401 - fail here if liblsl is missing
                self._offset_at = 0.0
                self.clock_offset()
                if self.simulated:
                    self._start_simulation()
            except BaseException:
                self._set_state(IDLE)
                raise
            self._running = True
            self._resolver = threading.Thread(target=self._resolve,
                                              daemon=True, name="lsl-resolve")
            self._resolver.start()
            self._set_state(RUNNING)

    def stop(self) -> None:
        with self._lock:
            if self.state == IDLE:
                return
            self._set_state(STOPPING)
            self._record_from = None
            self._running = False
            if self._resolver:
                self._resolver.join(timeout=3.0)
                self._resolver = None
            for inlet in list(self.inlets.values()):
                inlet.close()
            self.inlets = {}
            self._outlet = None
            if self._sim_thread:
                self._sim_thread.join(timeout=1.0)
                self._sim_thread = None
            self._set_state(IDLE)

    def _selectors(self) -> list[dict]:
        selectors = [s for s in self.config.streams if isinstance(s, dict)]
        if self.simulated:
            selectors = [{"name": SIMULATED_STREAM}]
        return selectors

    def _resolve(self):
        import pylsl
        while self._running:
            try:
                found = pylsl.resolve_streams(self.config.resolve_timeout)
            except Exception:  # noqa: BLE001
                found = []
            for info in found:
                if not self._running:
                    break
                if info.uid() in self.inlets:
                    continue
                if any(matches(s, info) for s in self._selectors()):
                    try:
                        self.inlets[info.uid()] = _Inlet(self, info)
                    except Exception:  # noqa: BLE001
                        pass
            deadline = time.monotonic() + RESOLVE_EVERY_S
            while self._running and time.monotonic() < deadline:
                time.sleep(0.1)

    def _inlet_failed(self, inlet: _Inlet, message: str) -> None:
        self.inlets.pop(inlet.uid, None)

    def _event(self, event: Event) -> None:
        with self._events_lock:
            self.received += 1
            self.log.append(event)
            if self._record_from is not None and \
                    event.time >= self._record_from:
                self._recorded.append(event)

    # -- simulation ----------------------------------------------------------------

    def _start_simulation(self):
        import pylsl
        info = pylsl.StreamInfo(SIMULATED_STREAM, "Markers", 1, 0,
                                "string", "braindump-sim-markers")
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
        return self._record_from is not None

    def begin_recording(self, target: RecordingTarget) -> list[Path]:
        with self._events_lock:
            self._recorded = []
            self._record_from = target.started
        return []

    def end_recording(self) -> RecordedFiles:
        # Samples sent just before now may still be on their way.
        time.sleep(0.3)
        with self._events_lock:
            events, self._recorded = self._recorded, []
            self._record_from = None
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
