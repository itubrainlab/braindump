"""The Unicorn Recorder's predefined filters, notches and display ranges.

Every filter is an IIR filter in second-order sections, applied causally
to the EEG channels only. Bandpass, highpass and lowpass are 4th-order
Butterworth designs; each notch is a second-order IIR notch with a
quality factor of 30. A cascading notch removes the line frequency and
every harmonic of it below the Nyquist frequency.

(From Rhino Recorder, the ITU brAIn lab's Unicorn recorder, MIT.)
"""
from __future__ import annotations

import numpy as np
from scipy import signal

#: The Unicorn's rate, which the presets default to.
SAMPLING_RATE = 250

FILTER_ORDER = 4
NOTCH_Q = 30.0

NONE = "None"

#: Name shown in the Filter box -> (btype, cutoff frequencies).
FILTERS: dict[str, tuple[str, list[float]] | None] = {
    "0.1 – 30 Hz": ("bandpass", [0.1, 30]),
    "0.1 – 50 Hz": ("bandpass", [0.1, 50]),
    "0.5 – 30 Hz": ("bandpass", [0.5, 30]),
    "0.5 – 50 Hz": ("bandpass", [0.5, 50]),
    "1 – 30 Hz": ("bandpass", [1, 30]),
    "1 – 50 Hz": ("bandpass", [1, 50]),
    "2 – 30 Hz": ("bandpass", [2, 30]),
    "2 – 50 Hz": ("bandpass", [2, 50]),
    "> 0.1 Hz": ("highpass", [0.1]),
    "> 0.5 Hz": ("highpass", [0.5]),
    "> 1 Hz": ("highpass", [1]),
    "> 2 Hz": ("highpass", [2]),
    "< 20 Hz": ("lowpass", [20]),
    "< 50 Hz": ("lowpass", [50]),
    NONE: None,
}

#: Name shown in the Notch box -> (line frequencies, cascading).
NOTCHES: dict[str, tuple[list[float], bool] | None] = {
    "50 Hz": ([50], False),
    "60 Hz": ([60], False),
    "50 & 60 Hz": ([50, 60], False),
    "Cascading 50 Hz": ([50], True),
    "Cascading 60 Hz": ([60], True),
    "Cascading 50 & 60 Hz": ([50, 60], True),
    NONE: None,
}

#: Name shown in the Scaling box -> half the displayed range, in uV.
SCALINGS: dict[str, float] = {
    f"±{v:g}µV": float(v)
    for v in (5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
}

DEFAULT_FILTER = "0.1 – 30 Hz"
DEFAULT_NOTCH = "50 Hz"
DEFAULT_SCALING = "±50µV"


def filter_sos(name: str, fs: float = SAMPLING_RATE) -> np.ndarray | None:
    """Second-order sections of the named filter; None for no filter."""
    spec = FILTERS[name]
    if spec is None:
        return None
    btype, cutoffs = spec
    wn = cutoffs if btype == "bandpass" else cutoffs[0]
    return signal.butter(FILTER_ORDER, wn, btype=btype, fs=fs, output="sos")


def notch_frequencies(name: str, fs: float = SAMPLING_RATE) -> list[float]:
    """Every frequency the named notch removes, in ascending order."""
    spec = NOTCHES[name]
    if spec is None:
        return []
    lines, cascading = spec
    nyquist = fs / 2
    found = set()
    for line in lines:
        harmonics = range(1, int(nyquist // line) + 1) if cascading else [1]
        for k in harmonics:
            f = line * k
            # A notch right at Nyquist cannot be designed.
            if f < nyquist - 1:
                found.add(f)
    return sorted(found)


def notch_sos(name: str, fs: float = SAMPLING_RATE) -> np.ndarray | None:
    """Second-order sections of the named notch; None for no notch."""
    sections = []
    for f in notch_frequencies(name, fs):
        b, a = signal.iirnotch(f, NOTCH_Q, fs=fs)
        sections.append(signal.tf2sos(b, a))
    if not sections:
        return None
    return np.vstack(sections)
