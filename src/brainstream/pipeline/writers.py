"""brAIn stream's BIDS writers: g.Pype file writers that write only while a
recording is open.

g.Pype's :class:`~gpype.backend.sinks.base.file_writer.FileWriter` writes
one file for each run of a pipeline, from its start to its stop. brAIn
dump's session pipeline runs from Connect to Disconnect, and records many
runs in between, so :class:`BidsWriter` keeps FileWriter's queue and
background thread and opens and closes its files on two g.Pype actions,
``begin`` and ``end``, which the session invokes through
``Pipeline.invoke_action``. Every writer counts the samples it is handed,
so a recording starts and ends at a sample: the first one to arrive after
``begin``, and the last one that had arrived when ``end`` was invoked.

The format writers:

- :class:`BdfWriter`: EEG as BDF+ (24-bit EDF+), with the trigger
  channel's onsets as events and annotations;
- :class:`PhysioWriter`: the rows of a :class:`~.sources.PushSource`, with
  their clock time, as a headerless gzipped TSV (BIDS physio);
- :class:`EventLog`: rows kept in memory, for sources whose recording is
  events rather than a file (LSL markers).
"""
from __future__ import annotations

import queue
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import gpype as gp
from gpype.backend.sinks.base.file_writer import FileWriter
from gpype.backend.sinks.base.sink import Sink
from gpype.common.constants import Constants

from ..bids import GzipTsvWriter

PORT_IN = Constants.Defaults.PORT_IN
Keys = Constants.Keys
SIGNAL = Constants.ChannelRoles.SIGNAL
TRIGGER = Constants.ChannelRoles.TRIGGER

#: How long ``end`` waits for the file to be written and closed.
END_TIMEOUT_S = 10.0

_OPEN, _ROWS, _CLOSE = "open", "rows", "close"


class BidsWriter(FileWriter):
    """A g.Pype file writer whose files open and close on its ``begin``
    and ``end`` actions, while the pipeline keeps running.

    Subclasses write the format: :meth:`_open` (in the writer's thread,
    with the path and the action's ``meta``), :meth:`_write` (each block,
    with the index of its first sample since the pipeline started) and
    :meth:`_close` (which returns a JSON-safe summary, the ``end``
    action's result). What the writer learned beyond that (events, say)
    it keeps in attributes, which the device reads after ``end``.
    """

    #: The file extension, ".tsv.gz" say; "" for a writer without files.
    extension = ".dat"

    def __init__(self, name: str, **kwargs):
        # FileWriter wants a file name with its extension; the real paths
        # come with each begin.
        FileWriter.__init__(self, file_name=f"{name}{self.extension}",
                            name=name, **kwargs)
        self._gate = threading.Lock()
        self._seen = 0
        self._is_open = False
        self._start_at = 0
        self._end_at: Optional[int] = None
        self._closed = threading.Event()
        self._closed.set()
        self._summary: dict = {}
        self._failure: Optional[BaseException] = None
        self.path: Optional[str] = None
        self.samples_written = 0
        #: Index (since the pipeline started) of the first sample written.
        self.start_index: Optional[int] = None
        self.labels: list[str] = []
        self.units: list[str] = []
        self.roles: list[str] = []
        self.sampling_rate: Optional[float] = None
        self.device_serial = ""

    # -- g.Pype ---------------------------------------------------------------

    @property
    def file_extension(self) -> str:
        return self.extension or ".dat"

    def _check_extension(self, file_name: str) -> None:
        # Paths are whole BIDS names (".tsv.gz"), given with begin.
        return

    def start(self):
        self._sample_counter = 0
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._file_worker,
                                               daemon=True,
                                               name=f"{self.name}-writer")
        self._worker_thread.start()
        Sink.start(self)

    def stop(self):
        try:
            self.end({})
        except Exception:  # noqa: BLE001 - stopping regardless
            pass
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=END_TIMEOUT_S)
            self._worker_thread = None
        Sink.stop(self)

    def setup(self, data, port_context_in):
        context = port_context_in.get(PORT_IN, {})
        width = int(context.get(Keys.CHANNEL_COUNT) or 0)
        self.labels = list(context.get(Keys.CHANNEL_LABELS)
                           or [f"Ch{i + 1}" for i in range(width)])
        self.units = [u or "" for u in (context.get(Keys.CHANNEL_UNITS)
                                        or [""] * width)]
        self.roles = list(context.get(Keys.CHANNEL_ROLES)
                          or [SIGNAL] * width)
        rate = context.get(Keys.SAMPLING_RATE)
        self.sampling_rate = float(rate) if rate else None
        self._sampling_rate = self.sampling_rate or 1.0
        self.device_serial = str(context.get(Keys.DEVICE_SERIAL) or "")
        with self._gate:
            self._seen = 0
        return {}

    def step(self, data):
        block = data.get(PORT_IN)
        if block is None:
            return {}
        with self._gate:
            first = self._seen
            self._seen += block.shape[0]
            if self._is_open:
                lo = max(self._start_at - first, 0)
                hi = block.shape[0]
                if self._end_at is not None:
                    hi = min(self._end_at - first, hi)
                if hi > lo:
                    self._file_queue.put((_ROWS, block[lo:hi].copy(),
                                          first + lo))
                    self.samples_written += hi - lo
                    if self.start_index is None:
                        self.start_index = first + lo
                if self._end_at is not None and self._seen >= self._end_at:
                    self._queue_close()
        return {}

    def _file_worker(self):
        while not self._stop_event.is_set() or not self._file_queue.empty():
            try:
                item = self._file_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            kind = item[0]
            try:
                if kind == _OPEN:
                    _, path, meta = item
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    self._open_file(path, meta)
                elif kind == _ROWS and self._failure is None:
                    self._write(item[1], item[2])
                elif kind == _CLOSE:
                    if self._failure is None:
                        self._summary = self._close() or {}
                    self._closed.set()
            except BaseException as error:  # noqa: BLE001 - reported at end
                self._failure = error
                if kind == _CLOSE:
                    self._closed.set()

    # -- actions --------------------------------------------------------------

    @property
    def recording(self) -> bool:
        return self._is_open

    @property
    def seen(self) -> int:
        return self._seen

    @gp.action
    def begin(self, arg: Optional[dict] = None) -> dict:
        """Open a file at ``arg["path"]`` and write into it every sample
        that arrives from now on. ``arg["meta"]`` goes to the format."""
        arg = arg or {}
        path = str(arg["path"]) if arg.get("path") else ""
        meta = dict(arg.get("meta") or {})
        with self._gate:
            if self._is_open:
                raise RuntimeError(f"{self.name} is already recording.")
            self.path = path or None
            self.samples_written = 0
            self.start_index = None
            self._start_at = self._seen
            self._end_at = None
            self._summary = {}
            self._failure = None
            self._closed.clear()
            self._file_queue.put((_OPEN, path, meta))
            self._is_open = True
        return {"path": path}

    @gp.action
    def end(self, arg: Optional[dict] = None) -> dict:
        """End the recording with the last sample that has arrived, wait
        for the file to be written, and return its summary."""
        with self._gate:
            if not self._is_open:
                return {}
            self._end_at = self._seen
            self._queue_close()
        if not self._closed.wait(END_TIMEOUT_S):
            raise RuntimeError(f"{self.name} did not finish writing "
                               f"{self.path} in {END_TIMEOUT_S:g} s.")
        if self._failure is not None:
            failure, self._failure = self._failure, None
            raise RuntimeError(f"{self.name}: {failure}") from failure
        return dict(self._summary, samples=self.samples_written,
                    start_index=self.start_index)

    def _queue_close(self) -> None:
        # Under the gate: no more rows are queued after this.
        if self._is_open:
            self._is_open = False
            self._file_queue.put((_CLOSE,))

    # -- the format ---------------------------------------------------------------

    def _open_file(self, file_path, meta) -> None:  # FileWriter's name
        self._open(file_path, meta)

    def _write_block(self, block, timestamps) -> None:  # unused
        return

    def _close_file(self) -> None:  # unused: closing is _close
        return

    def _open(self, path: str, meta: dict) -> None:
        raise NotImplementedError

    def _write(self, block: np.ndarray, first_index: int) -> None:
        raise NotImplementedError

    def _close(self) -> dict:
        raise NotImplementedError


# -- physio ------------------------------------------------------------------------

class PhysioWriter(BidsWriter):
    """The rows of a push source, ``[TIME, *values]``, into a headerless
    gzipped TSV: the clock time (``meta["epoch"]`` plus TIME) and each
    value times ``scale``. The summary gives the first and last sample's
    clock time."""

    extension = ".tsv.gz"

    def __init__(self, name: str, scale: float = 1.0, **kwargs):
        super().__init__(name, **kwargs)
        self.scale = scale
        self.epoch = 0.0
        self.first_sample: Optional[float] = None
        self.last_sample: Optional[float] = None
        self._writer: Optional[GzipTsvWriter] = None

    def _open(self, path, meta):
        self.epoch = float(meta.get("epoch", 0.0))
        self.first_sample = self.last_sample = None
        self._writer = GzipTsvWriter(Path(path))

    def _write(self, block, first_index):
        for row in block:
            t = self.epoch + float(row[0])
            if self.first_sample is None:
                self.first_sample = t
            self.last_sample = t
            self._writer.write([t, *(float(v) * self.scale
                                     for v in row[1:])])

    def _close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        return {"first": self.first_sample, "last": self.last_sample}


class EventLog(BidsWriter):
    """No file: keeps the rows a push source sends while recording,
    ``(clock time, values)``, in :attr:`rows`."""

    extension = ""

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.epoch = 0.0
        self.rows: list[tuple[float, np.ndarray]] = []

    def _open(self, path, meta):
        self.epoch = float(meta.get("epoch", 0.0))
        self.rows = []

    def _write(self, block, first_index):
        for row in block:
            self.rows.append((self.epoch + float(row[0]),
                              np.array(row[1:], dtype=float)))

    def _close(self):
        return {"events": len(self.rows)}


# -- EEG -----------------------------------------------------------------------------

#: Physical range of a BDF channel, by unit. EEG spans the 24-bit range of
#: the amplifiers' converters, so the raw values survive.
_BDF_RANGE = {
    "uV": (-750000.0, 750000.0),
    "µV": (-750000.0, 750000.0),
    "g": (-8.0, 8.0),
    "deg/s": (-1000.0, 1000.0),
    "%": (0.0, 100.0),
    "ms": (0.0, 60000.0),
}
_BDF_DIGITAL = (-8388608, 8388607)


def _bdf_range(label: str, unit: str, role: str) -> tuple[float, float]:
    if label in ("VALID", "Validation"):
        return (0.0, 1.0)
    if role == TRIGGER or label in ("CNT", "TRIG", "Counter"):
        # One digital step per count: a counter fits for 18 hours.
        return (0.0, float(_BDF_DIGITAL[1] - _BDF_DIGITAL[0]))
    return _BDF_RANGE.get(unit, (-750000.0, 750000.0))


def _header_number(value: float):
    """A whole number as an int, so the 8-character EDF header field does
    not lose its last digit to a trailing ".0"."""
    return int(value) if float(value).is_integer() else value


class BdfWriter(BidsWriter):
    """EEG as BDF+ (24-bit EDF+), with one-second data records.

    The EEG channels (signal role) are named as ``meta["electrodes"]``
    says, in order, and the trigger channel TRIG. A trigger is an onset:
    the sample where TRIG turns to a new non-zero value (an amplifier's
    digital input holds its level while the line is high). Each is kept
    in :attr:`events` as ``(seconds, sample, value)`` and written as an
    annotation. A last, partial data record is padded by repeating the
    last sample.
    """

    extension = ".bdf"
    #: Annotation signals reserved; each holds a few annotations per
    #: one-second record.
    ANNOTATION_SIGNALS = 8

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.events: list[tuple[float, int, float]] = []
        self.file_labels: list[str] = []
        #: The scan counter of the first sample written (where there is a
        #: counter channel, "CNT" or "Counter").
        self.first_counter: Optional[float] = None
        self._writer = None

    def _names(self, electrodes: list[str]) -> list[str]:
        labels, signal = list(self.labels), 0
        for i, role in enumerate(self.roles):
            if role == SIGNAL:
                if signal < len(electrodes) and electrodes[signal]:
                    labels[i] = str(electrodes[signal])
                signal += 1
            elif role == TRIGGER:
                labels[i] = "TRIG"
        return labels

    def _open(self, path, meta):
        import pyedflib
        self.file_labels = self._names(list(meta.get("electrodes") or []))
        rate = float(self.sampling_rate or 1.0)
        n = len(self.file_labels)
        self._ranges = np.array([
            _bdf_range(lab, unit, role) for lab, unit, role
            in zip(self.file_labels, self.units, self.roles)])
        # pyedflib warns about spaces in header fields, which subject and
        # recording identifications may well contain.
        warnings.filterwarnings("ignore", "Invalid char", UserWarning,
                                "pyedflib")
        writer = pyedflib.EdfWriter(path, n,
                                    file_type=pyedflib.FILETYPE_BDFPLUS)
        writer.set_number_of_annotation_signals(self.ANNOTATION_SIGNALS)
        writer.setSignalHeaders([{
            "label": self.file_labels[i][:16],
            "dimension": "uV" if self.units[i] == "µV"
            else (self.units[i] if self.units[i] != "-" else ""),
            "sample_frequency": rate,
            "physical_min": _header_number(self._ranges[i, 0]),
            "physical_max": _header_number(self._ranges[i, 1]),
            "digital_min": _BDF_DIGITAL[0],
            "digital_max": _BDF_DIGITAL[1],
            "transducer": "",
            "prefilter": "",
        } for i in range(n)])
        writer.setStartdatetime(datetime.now())
        if meta.get("subject"):
            writer.setPatientCode(str(meta["subject"]))
        serial = meta.get("serial") or self.device_serial
        if serial:
            writer.setEquipment(str(serial))
        if self._marked:
            from gpype.common._private.entitlement import MARK
            writer.setRecordingAdditional(MARK)
        elif meta.get("recording"):
            writer.setRecordingAdditional(str(meta["recording"]))
        self._writer = writer
        self._record = int(round(rate * writer.record_duration))
        self._buffer = np.empty((0, n))
        self._trigger = (self.file_labels.index("TRIG")
                         if "TRIG" in self.file_labels else None)
        counter = [i for i, lab in enumerate(self.labels)
                   if lab in ("CNT", "Counter")]
        self._counter = counter[0] if counter else None
        self.events = []
        self.first_counter = None
        self._last_trigger = 0.0
        self._written = 0

    def _write(self, block, first_index):
        rate = float(self.sampling_rate or 1.0)
        if self._written == 0 and self._counter is not None and len(block):
            self.first_counter = float(block[0, self._counter])
        if self._trigger is not None:
            values = block[:, self._trigger]
            previous = np.concatenate(([self._last_trigger], values[:-1]))
            for row in np.flatnonzero((values != 0) & (values != previous)):
                sample = self._written + int(row)
                value = float(values[row])
                self.events.append((sample / rate, sample, value))
                self._writer.writeAnnotation(sample / rate, -1, f"{value:g}")
            if len(values):
                self._last_trigger = float(values[-1])
        self._written += block.shape[0]
        block = np.clip(block, self._ranges[:, 0], self._ranges[:, 1])
        self._buffer = np.vstack((self._buffer, block))
        while self._buffer.shape[0] >= self._record:
            self._flush(self._buffer[:self._record])
            self._buffer = self._buffer[self._record:]

    def _flush(self, record):
        self._writer.writeSamples(
            [np.ascontiguousarray(record[:, i], dtype=float)
             for i in range(record.shape[1])])

    def _close(self):
        if self._writer is None:
            return {}
        if len(self._buffer):
            pad = np.repeat(self._buffer[-1:],
                            self._record - len(self._buffer), axis=0)
            self._flush(np.vstack((self._buffer, pad)))
        self._buffer = np.empty((0, len(self.file_labels)))
        self._writer.close()
        self._writer = None
        return {"triggers": len(self.events)}


def wait_until(predicate, timeout: float) -> bool:
    """Polls ``predicate`` until it holds or ``timeout`` passes."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


__all__ = ["BidsWriter", "PhysioWriter", "EventLog", "BdfWriter",
           "END_TIMEOUT_S", "wait_until"]
