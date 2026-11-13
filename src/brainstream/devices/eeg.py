"""What brAIn stream's EEG amplifiers share: the recording filter's
description, and the measurement of a recording's timing. (Their BIDS BDF
writer is :class:`brainstream.pipeline.BdfWriter`.)"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
from gpype.common.constants import Constants
from .. import presets


from ..bids import NA
from .base import Timeline

SIGNAL = Constants.ChannelRoles.SIGNAL
TRIGGER = Constants.ChannelRoles.TRIGGER
#: Shortest recording, in seconds, whose rate is measured rather than
#: taken as nominal.
MIN_FIT_S = 2.0


# -- recording filters ------------------------------------------------------------------

#: Low-pass cut-offs offered beyond Unicorn Recorder's, for amplifiers that
#: sample faster than the Unicorn's 250 Hz.
FAST_LOWPASS = (100.0, 200.0, 500.0, 1000.0, 2000.0)


def _cutoffs(btypes: tuple[str, ...], index: int) -> list[float]:
    return sorted({float(spec[1][index if spec[0] == "bandpass" else 0])
                   for spec in presets.FILTERS.values()
                   if spec is not None and spec[0] in btypes})


def highpass_choices() -> list[float]:
    """High-pass cut-offs: none (0), then every one of Unicorn Recorder's
    predefined filters (see :mod:`brainstream.presets`)."""
    return [0.0] + _cutoffs(("highpass", "bandpass"), 0)


def lowpass_choices(fs: float) -> list[float]:
    """Low-pass cut-offs below half of ``fs``: none (0, i.e. half the
    sampling rate) first, then Unicorn Recorder's, then faster ones."""
    found = set(_cutoffs(("lowpass", "bandpass"), 1)) | set(FAST_LOWPASS)
    return [0.0] + sorted(f for f in found if f < fs / 2)


def channel_filter_columns(low: Optional[float], high: Optional[float],
                           notches: list[float]) -> tuple[str, str, str]:
    """channels.tsv's low_cutoff, high_cutoff and notch for a filtered
    channel: n/a for no filter, several notch frequencies as a list."""
    if not notches:
        notch = NA
    elif len(notches) == 1:
        notch = f"{notches[0]:g}"
    else:
        notch = "[" + ", ".join(f"{f:g}" for f in notches) + "]"
    return (f"{low:g}" if low else NA, f"{high:g}" if high else NA, notch)


@dataclass
class RecordingFilter:
    """The filter applied to the EEG channels as they are recorded: a
    4th-order Butterworth high-pass, low-pass or band-pass, then a notch,
    all causal (as Unicorn Recorder filters)."""
    highpass: float = 0.1
    #: Half the sampling rate or more, or 0, is no low-pass.
    lowpass: float = 0.0
    notch: str = presets.NONE
    fs: float = 250.0

    @property
    def low_cutoff(self) -> Optional[float]:
        """The high-pass's cut-off, or None."""
        return float(self.highpass) if 0 < self.highpass < self.fs / 2 \
            else None

    @property
    def high_cutoff(self) -> Optional[float]:
        """The low-pass's cut-off, or None (0, or half the sampling rate
        and above, is no low-pass)."""
        return float(self.lowpass) if 0 < self.lowpass < self.fs / 2 \
            else None

    def notches(self) -> list[float]:
        return presets.notch_frequencies(self.notch, self.fs) \
            if self.notch in presets.NOTCHES else []

    def sos(self) -> Optional[np.ndarray]:
        """Second-order sections of the whole filter; None for none."""
        from scipy import signal
        low, high = self.low_cutoff, self.high_cutoff
        sections = []
        if low and high:
            if low >= high:
                raise ValueError(f"The recording filter's high-pass "
                                 f"({low:g} Hz) is above its low-pass "
                                 f"({high:g} Hz).")
            sections.append(signal.butter(presets.FILTER_ORDER, [low, high],
                                          btype="bandpass", fs=self.fs,
                                          output="sos"))
        elif low or high:
            sections.append(signal.butter(
                presets.FILTER_ORDER, low or high,
                btype="highpass" if low else "lowpass", fs=self.fs,
                output="sos"))
        if self.notches():
            sections.append(presets.notch_sos(self.notch, self.fs))
        return np.vstack(sections) if sections else None

    def software_filters(self):
        """The eeg.json's SoftwareFilters: what was applied, or n/a."""
        found = {}
        low, high = self.low_cutoff, self.high_cutoff
        common = {"Type": "Butterworth", "Order": presets.FILTER_ORDER,
                  "Causal": True, "Channels": "EEG"}
        if low and high:
            found["BandPass"] = {**common, "CutoffFrequencies": [low, high]}
        elif low:
            found["HighPass"] = {**common, "CutoffFrequency": low}
        elif high:
            found["LowPass"] = {**common, "CutoffFrequency": high}
        if self.notches():
            found["Notch"] = {"Type": "IIR notch",
                              "QualityFactor": presets.NOTCH_Q,
                              "Frequencies": self.notches(),
                              "Causal": True, "Channels": "EEG"}
        return found or NA




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
