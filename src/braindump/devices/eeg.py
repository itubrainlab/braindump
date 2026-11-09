"""What brAIn dump's EEG amplifiers share: the BIDS BDF logger and the
measurement of a recording's timing."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
from gpype.common.constants import Constants
from rhinorecorder.nodes import BdfLogger

from .. import clock
from .base import Timeline

SIGNAL = Constants.ChannelRoles.SIGNAL
TRIGGER = Constants.ChannelRoles.TRIGGER
#: Shortest recording, in seconds, whose rate is measured rather than
#: taken as nominal.
MIN_FIT_S = 2.0


@dataclass
class TimingFit:
    timeline: Timeline
    #: Spread of the acquisition stamps around the fitted line, in s.
    residual_sd: float = 0.0
    residual_max: float = 0.0
    method: str = "fit"

    def metadata(self, how: str) -> dict:
        """``RecordingTiming`` for ``_eeg.json``; ``how`` says how the
        acquisition times were taken."""
        fs = self.timeline.nominal_rate
        return {
            "Description": "When the first sample was acquired, on the "
                           "recording computer's clock, and the rate the "
                           "samples really came at by that clock (fitted to "
                           f"every sample's acquisition time: {how}). Event "
                           "onsets and the other recordings' StartTime are "
                           "in this file's time: sample / "
                           "SamplingFrequency.",
            "FirstSampleTime": datetime.fromtimestamp(
                self.timeline.t0).isoformat(timespec="microseconds"),
            "MeasuredSamplingFrequency": round(self.timeline.rate, 6),
            "ClockDeviationPPM": round((self.timeline.rate / fs - 1) * 1e6,
                                       1),
            "TimestampResidualSD": round(self.residual_sd, 6),
            "TimestampResidualMax": round(self.residual_max, 6),
            "Method": self.method,
        }


def fit_timing(acquired: np.ndarray, nominal: float) -> TimingFit:
    """When sample 0 was acquired, and the rate the samples really came
    at, from every sample's acquisition stamp: a straight line through
    them (refitted without stamps more than 3 SD off, as a delayed burst
    can be). Short recordings, or a rate more than 1 % off nominal (stamps
    that cannot be right), keep the nominal rate."""
    n = len(acquired)
    k = np.arange(n, dtype=float)
    if n < 2:
        t0 = float(acquired[0]) if n else 0.0
        return TimingFit(Timeline(t0, nominal, nominal), method="nominal")
    if n >= MIN_FIT_S * nominal:
        slope, intercept = np.polyfit(k, acquired, 1)
        res = acquired - (intercept + slope * k)
        keep = np.abs(res) <= max(3 * res.std(), 0.002)
        if keep.sum() > n // 2:
            slope, intercept = np.polyfit(k[keep], acquired[keep], 1)
        rate = 1.0 / slope if slope > 0 else nominal
        if abs(rate / nominal - 1) <= 0.01:
            res = acquired - (intercept + slope * k)
            return TimingFit(Timeline(float(intercept), float(rate),
                                      nominal),
                             float(res.std()), float(np.abs(res).max()))
    # The nominal rate, placed where the stamps are (their median).
    t0 = float(np.median(acquired - k / nominal))
    res = acquired - (t0 + k / nominal)
    return TimingFit(Timeline(t0, nominal, nominal), float(res.std()),
                     float(np.abs(res).max()), "nominal")


class BidsBdfLogger(BdfLogger):
    """Rhino Recorder's BDF+ logger, for BIDS: the EEG channels named as
    the project says (in order of their signal role), the trigger channel
    named TRIG, and kept beside the file: which sample it started at (and
    that sample's counter, where there is one), and the trigger events.

    A trigger is an onset: the sample where the trigger channel turns to a
    new non-zero value. An amplifier's digital input holds its level for
    as long as the line is high; only its rising edges are events."""

    def __init__(self, electrodes: list[str], latency_ms: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.electrodes = list(electrodes)
        self.latency_s = latency_ms / 1000.0
        #: A fallback first-sample time, from when it arrived.
        self.first_arrival: Optional[float] = None
        #: Pipeline index and counter of the first sample written.
        self.start_index: Optional[int] = None
        self.first_counter: Optional[float] = None
        self.events: list[tuple[float, int, float]] = []
        self._last_trigger = 0.0

    def setup(self, data, port_context_in):
        out = super().setup(data, port_context_in)
        labels, signal = list(self.labels), 0
        for i, role in enumerate(self.roles):
            if role == SIGNAL:
                if signal < len(self.electrodes) and self.electrodes[signal]:
                    labels[i] = self.electrodes[signal]
                signal += 1
            elif role == TRIGGER:
                labels[i] = "TRIG"
        self.labels = labels
        return out

    def _open_file(self, path):
        self.first_arrival = None
        self.start_index = self.first_counter = None
        self.events = []
        self._last_trigger = 0.0
        super()._open_file(path)

    def _write(self, block):
        if self.first_arrival is None:
            # The newest of these rows has just arrived; it was acquired
            # the link's delay before.
            self.first_arrival = (clock.now() - self.latency_s
                                  - (block.shape[0] - 1) / self.sampling_rate)
            self.start_index = self._start_at
            if "CNT" in self.labels:
                self.first_counter = float(block[0, self.labels.index("CNT")])
        status = self._status
        if status is not None:
            values = block[:, status]
            previous = np.concatenate(([self._last_trigger], values[:-1]))
            for row in np.flatnonzero((values != 0) & (values != previous)):
                sample = self.samples_written + int(row)
                value = float(values[row])
                self.events.append((sample / self.sampling_rate, sample,
                                    value))
                self._writer.writeAnnotation(sample / self.sampling_rate, -1,
                                             f"{value:g}")
            self._last_trigger = float(values[-1]) if len(values) else \
                self._last_trigger
        # The onsets are annotated above; BdfLogger would annotate every
        # non-zero sample.
        self._status = None
        try:
            super()._write(block)
        finally:
            self._status = status
