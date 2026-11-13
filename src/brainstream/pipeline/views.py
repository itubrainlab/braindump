"""The nodes the device pages draw from: g.Pype nodes of brAIn stream's own.

- :class:`ViewFilter` filters the EEG channels with a filter that can be
  swapped while the pipeline runs, so a page's Filter and Notch boxes
  work without restarting it. g.Pype's own filters can be bypassed while
  running, but not retuned; they are what the recording filters use.
- :class:`SweepTap` keeps the last seconds of a stream for a sweeping
  display.

(From Rhino Recorder, the ITU brAIn lab's Unicorn recorder, MIT.)
"""
from __future__ import annotations

import threading
from typing import Optional

import gpype as gp
import numpy as np
from gpype.common.constants import Constants
from scipy.signal import sosfilt, sosfilt_zi

PORT_IN = Constants.Defaults.PORT_IN
PORT_OUT = Constants.Defaults.PORT_OUT
Keys = Constants.Keys
SIGNAL = Constants.ChannelRoles.SIGNAL


class ViewFilter(gp.IONode):
    """An IIR filter on the EEG (signal) channels that can be replaced
    while the pipeline runs. The other channels pass through untouched.
    It starts, and restarts after each swap, from the steady state of the
    sample it starts on: an amplifier's electrode offsets are hundreds of
    millivolts, and a high-pass started from zero would ring for seconds.

    Args:
        sos: Second-order sections, or None to pass the EEG through.
    """

    def __init__(self, sos: Optional[np.ndarray] = None, **kwargs):
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._pending = (sos,)
        self._sos: Optional[np.ndarray] = None
        self._z: Optional[np.ndarray] = None
        self._signal: list[int] = []
        self._primed = False

    def set_sos(self, sos: Optional[np.ndarray]) -> None:
        """Use these sections from the next frame on."""
        with self._lock:
            self._pending = (sos,)

    def setup(self, data, port_context_in):
        context = port_context_in[PORT_IN]
        roles = context.get(Keys.CHANNEL_ROLES)
        width = int(context[Keys.CHANNEL_COUNT])
        self._signal = ([i for i, r in enumerate(roles) if r == SIGNAL]
                        if roles else list(range(width)))
        with self._lock:
            if self._pending is None:
                self._pending = (self._sos,)
        self._primed = False
        return super().setup(data, port_context_in)

    def step(self, data):
        block = data[PORT_IN]
        with self._lock:
            pending, self._pending = self._pending, None
        if pending is not None:
            self._sos = pending[0]
            self._primed = False
        if self._sos is None or not len(block):
            return {PORT_OUT: block}
        x = block[:, self._signal].astype(np.float64)
        if not self._primed:
            zi = sosfilt_zi(self._sos)
            self._z = zi[:, :, None] * x[0][None, None, :]
            self._primed = True
        out = block.copy()
        y, self._z = sosfilt(self._sos, x, axis=0, zi=self._z)
        out[:, self._signal] = y
        return {PORT_OUT: out}


class SweepTap(gp.Sink):
    """The last ``seconds`` of samples, laid out for a sweeping display:
    sample *k* lives in slot ``k % capacity``. Knows its input's channel
    labels, units and roles."""

    def __init__(self, seconds: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.seconds = float(seconds)
        self._lock = threading.Lock()
        self.labels: list[str] = []
        self.units: list[str] = []
        self.roles: list[str] = []
        self.sampling_rate = 1.0
        self.capacity = 1
        self._buffer = np.zeros((1, 1))
        self.total = 0

    def setup(self, data, port_context_in):
        context = port_context_in[PORT_IN]
        width = int(context[Keys.CHANNEL_COUNT])
        self.labels = list(context.get(Keys.CHANNEL_LABELS)
                           or [f"Ch{i + 1}" for i in range(width)])
        self.units = [u or "" for u in (context.get(Keys.CHANNEL_UNITS)
                                        or [""] * width)]
        self.roles = list(context.get(Keys.CHANNEL_ROLES) or [SIGNAL] * width)
        self.sampling_rate = float(context.get(Keys.SAMPLING_RATE) or 1.0)
        with self._lock:
            self.capacity = max(int(self.seconds * self.sampling_rate), 1)
            self._buffer = np.zeros((self.capacity, width))
            self.total = 0
        return super().setup(data, port_context_in)

    def step(self, data):
        block = data.get(PORT_IN)
        if block is None:
            return None
        with self._lock:
            n = block.shape[0]
            slots = (self.total + np.arange(n)) % self.capacity
            self._buffer[slots] = block
            self.total += n
        return None

    def sweep(self) -> tuple[np.ndarray, int]:
        """A copy of the buffer and the number of samples seen so far."""
        with self._lock:
            return self._buffer.copy(), self.total

    def latest(self, n: int) -> np.ndarray:
        """The last ``n`` samples (fewer at the start), oldest first."""
        with self._lock:
            n = min(n, self.total, self.capacity)
            if n == 0:
                return np.empty((0, self._buffer.shape[1]))
            slots = (self.total - n + np.arange(n)) % self.capacity
            return self._buffer[slots].copy()
