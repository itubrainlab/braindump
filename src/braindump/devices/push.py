"""A g.Pype source fed from outside: a device's own thread pushes rows,
one at a time, as g.Pype's LslReceiver does with its stream."""
from __future__ import annotations

import threading
import time
from typing import Optional, Sequence

import numpy as np
from gpype.backend.core.o_port import OPort
from gpype.backend.sources.base.source import Source
from gpype.common.constants import Constants

PORT_OUT = Constants.Defaults.PORT_OUT


class PushSource(Source):
    """Rows of ``[time.time(), *values]`` pushed by :meth:`push`.

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
        self.epoch = time.time()
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
        self.epoch = time.time()
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
