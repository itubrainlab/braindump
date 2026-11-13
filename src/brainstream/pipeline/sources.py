"""brAIn stream's g.Pype sources: what every input device feeds into the
session pipeline.

- **Amplifiers** (Unicorn, g.HIamp) are g.Pype's own sources, with
  :class:`AcquisitionStamps` keeping every scan's acquisition time, which
  g.Pype uses to place triggers and then drops before any sink.
- **Every other device** (eye tracker, heart-rate strap, GSR, LSL markers,
  video) is a :class:`PushSource`: the device's own thread pushes rows,
  one at a time, as g.Pype's LslReceiver does with its stream. The first
  channel of every row is the time the device says the row was acquired,
  on brAIn stream's clock (:mod:`brainstream.clock`).

g.Pype gives each stream its own sample grid. How the grids line up --
which EEG sample a gaze sample or a marker happened during -- is measured
on top of it, from these times (see :class:`ClockMap` and
:func:`brainstream.devices.eeg.fit_timing`).
"""
from __future__ import annotations

import threading
from array import array
from collections import deque
from typing import Optional, Sequence

import numpy as np
from gpype.backend.core.o_port import OPort
from gpype.backend.sources.base.source import Source
from gpype.common.constants import Constants

from .. import clock

PORT_OUT = Constants.Defaults.PORT_OUT
Keys = Constants.Keys


class PushSource(Source):
    """Rows of ``[clock time, *values]`` pushed by :meth:`push`.

    g.Pype carries float32, too coarse for clock times, so the first
    channel leaves as ``TIME``: seconds since :attr:`epoch` (when the
    pipeline started this source). Writers add the epoch back.

    A push source never claims the pipeline's master timeline: an
    amplifier, when there is one, owns it.
    """

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
        self._started = False
        self.samples = 0

    def setup(self, data, port_context_in):
        out = super().setup(data, port_context_in)
        ctx = out[PORT_OUT]
        ctx[Keys.SAMPLING_RATE] = self.rate
        ctx[Keys.FRAME_RATE] = self.rate
        ctx[Keys.CHANNEL_LABELS] = list(self.labels)
        ctx[Keys.CHANNEL_UNITS] = list(self.units)
        return out

    def start(self):
        self.epoch = clock.now()
        Source.start(self)
        self._started = True

    def stop(self):
        self._started = False
        Source.stop(self)

    def push(self, row) -> None:
        """One row, ``[clock time, *values]``, into the pipeline. Ignored
        unless the pipeline is running."""
        if not self._started:
            return
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
    """A device's own time stamps onto brAIn stream's clock.

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


class AcquisitionStamps:
    """Keeps every scan's acquisition time, on brAIn stream's clock, for an
    amplifier source g.Pype stamps.

    g.Pype's amplifier sources append the time each frame reached this
    computer, use it to place triggers, and drop it before any sink. This
    keeps it, frame by frame, with the scan counter where the device has
    one (``counter_column``), so a recording's timing can be measured:
    when its first sample was acquired, and how fast the amplifier's
    clock really ran (:func:`brainstream.devices.eeg.fit_timing`).

    With ``burst_corrected``, a frame's stamp is moved back by the scans
    still waiting behind it in the driver (a scan early in a Bluetooth
    burst waits for the rest of it). Either way ``latency_s``, the link's
    fixed delay, is taken off.
    """

    #: Column of the scan counter in the device's frame; None for none.
    counter_column: Optional[int] = None
    burst_corrected: bool = False

    def __init__(self, *args, latency_ms: float = 0.0, **kwargs):
        self._stamps_lock = threading.Lock()
        self.acquired = array("d")
        self.counters = array("d")
        #: Scans the counter skipped (lost on the way), since the start.
        self.lost_scans = 0
        self._latency_s = max(float(latency_ms), 0.0) / 1000.0
        super().__init__(*args, **kwargs)

    def start(self):
        with self._stamps_lock:
            self.acquired = array("d")
            self.counters = array("d")
            self.lost_scans = 0
        return super().start()

    def _waiting_scans(self) -> Optional[int]:
        device = getattr(self, "_device", None)
        try:
            return int(device.get_stream_status().available_scans)
        except Exception:  # noqa: BLE001 - no such status: arrival time
            return None

    def cycle(self):
        frame = getattr(self, "_current_frame", None)
        width = getattr(self, "_total_channels", None)
        if frame is not None and width is not None and \
                frame.shape[1] > width:
            from gpype.common._private import channels
            now = channels.stamp_clock()
            to_clock = clock.from_stamp(now) - now
            rate = float(getattr(self, "SAMPLING_RATE", 0) or
                         self.config.get(Keys.SAMPLING_RATE, [0])[0] or 1)
            waiting = self._waiting_scans() if self.burst_corrected \
                else None
            with self._stamps_lock:
                n = frame.shape[0]
                if waiting is not None:
                    last = now - waiting / rate
                    stamps = last - (n - 1 - np.arange(n)) / rate
                else:
                    stamps = [channels.unwrap_time(float(row[width]), now)
                              for row in frame]
                for i, stamp in enumerate(stamps):
                    acquired = float(stamp) - self._latency_s
                    # g.Pype places triggers by this stamp: it must say
                    # when the scan was acquired, not when it arrived.
                    frame[i, width:] = channels.wrap_time(acquired)
                    self.acquired.append(acquired + to_clock)
                    if self.counter_column is not None:
                        count = float(frame[i, self.counter_column])
                        if len(self.counters) and \
                                count > self.counters[-1] + 1:
                            self.lost_scans += int(count
                                                   - self.counters[-1] - 1)
                        self.counters.append(count)
        return super().cycle()

    def stamps(self) -> tuple[np.ndarray, np.ndarray]:
        """Acquisition times and counters of every scan since the start."""
        with self._stamps_lock:
            return (np.frombuffer(self.acquired, dtype=float).copy(),
                    np.frombuffer(self.counters, dtype=float).copy())
