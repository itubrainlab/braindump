"""A g.Pype source fed from outside: a device's own thread pushes rows,
one at a time, as g.Pype's LslReceiver does with its stream."""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional, Sequence

import numpy as np
from gpype.backend.core.o_port import OPort
from gpype.backend.sources.base.source import Source
from gpype.common.constants import Constants
from pathlib import Path
from rhinorecorder.nodes import GatedLogger

from .. import clock
from ..bids import GzipTsvWriter

PORT_OUT = Constants.Defaults.PORT_OUT


class PushSource(Source):
    """Rows of ``[clock time, *values]`` pushed by :meth:`push`
    (:mod:`braindump.clock`).

    g.Pype carries float32, too coarse for clock times, so the first
    channel leaves as ``TIME``: seconds since :attr:`epoch` (the source's
    start). Loggers add the epoch back."""

    def __init__(self, labels: Sequence[str], units: Sequence[str],
                 rate: float, **kwargs):
        self.labels = ["TIME", *labels]
        self.units = ["s", *units]
        self.rate = float(rate)
        kwargs.setdefault("frame_size", 1)
        kwargs.setdefault("output_ports", [OPort.Configuration()])
        kwargs.setdefault("channel_count", len(self.labels))
        super().__init__(sampling_rate=self.rate, **kwargs)
        self.epoch = clock.now()
        self._pending: Optional[np.ndarray] = None
        self._push_lock = threading.Lock()
        self.samples = 0

    def setup(self, data, port_context_in):
        out = super().setup(data, port_context_in)
        ctx = out[PORT_OUT]
        ctx[Constants.Keys.SAMPLING_RATE] = self.rate
        ctx[Constants.Keys.FRAME_RATE] = self.rate
        ctx[Constants.Keys.CHANNEL_LABELS] = list(self.labels)
        ctx[Constants.Keys.CHANNEL_UNITS] = list(self.units)
        return out

    def start(self):
        self.epoch = clock.now()
        Source.start(self)

    def push(self, row) -> None:
        values = np.empty((1, len(self.labels)))
        values[0, 0] = row[0] - self.epoch
        values[0, 1:] = row[1:]
        with self._push_lock:
            self._pending = values
            self.samples += 1
            self.cycle()

    def step(self, data):
        pending, self._pending = self._pending, None
        return None if pending is None else {PORT_OUT: pending}


class ClockMap:
    """A device's own time stamps onto brAIn dump's clock.

    Each reading arrives some delay after it was stamped: never less than
    the link's fastest, often more (Bluetooth bursts, scheduling). So the
    smallest ``received - sensor`` seen is the best estimate of the
    offset between the clocks, and it is taken over a window of
    ``window_s`` seconds: long enough that every burst pattern has
    delivered a fast reading, short enough to follow the slow drift
    between the device's crystal and this computer's (tens of ppm: a few
    milliseconds a minute).

    What the data cannot tell is the link's minimum delay itself: the
    fastest reading still arrived that long after it was taken. Give it
    as ``latency_s`` when it is known (measured, or the manufacturer's
    figure); otherwise samples are placed that much late."""

    def __init__(self, window_s: float = 30.0, latency_s: float = 0.0):
        self.window_s = window_s
        self.latency_s = latency_s
        #: (received, offset), offsets increasing: a sliding minimum.
        self._window: deque[tuple[float, float]] = deque()

    def __call__(self, sensor: float, received: float) -> float:
        offset = received - sensor
        window = self._window
        while window and window[-1][1] >= offset:
            window.pop()
        window.append((received, offset))
        while window[0][0] < received - self.window_s:
            window.popleft()
        return sensor + window[0][1] - self.latency_s

    @property
    def offset(self) -> Optional[float]:
        return self._window[0][1] if self._window else None


class PhysioLogger(GatedLogger):
    """Rows of a push source, with their clock time, into a headerless
    gzipped TSV, while recording."""

    extension = ".tsv.gz"

    def __init__(self, scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.epoch = 0.0
        self.first_sample: Optional[float] = None
        self.last_sample: Optional[float] = None
        self._writer: Optional[GzipTsvWriter] = None

    def _open_file(self, path):
        self._writer = GzipTsvWriter(Path(path))
        self.first_sample = self.last_sample = None

    def _write(self, block):
        for row in block:
            t = self.epoch + float(row[0])
            if self.first_sample is None:
                self.first_sample = t
            self.last_sample = t
            self._writer.write([t, *(float(v) * self.scale
                                     for v in row[1:])])

    def _close_file(self):
        self._writer.close()
        self._writer = None
