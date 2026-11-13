"""The power spectrum shown beside the data viewer: Welch's estimate of
each electrode's power spectral density over the last few seconds of the
EEG, filtered or raw, averaged across the electrodes.

It covers the EEG frequency bands, up to the top of the gamma band
(:data:`MAX_FREQUENCY`), below the 125 Hz the headset can measure at its
250 Hz sampling rate. The window
chooses the EEG: after OSCAR (when on) and the display filter and notch,
or raw, as acquired.

(From Rhino Recorder, the ITU brAIn lab's Unicorn recorder, MIT.)
"""
from __future__ import annotations

import numpy as np
from scipy import signal

SAMPLING_RATE = 250

#: Seconds of EEG each estimate uses.
WINDOW_S = 8.0
#: Seconds per Welch segment: 1 Hz resolution, segments overlapping by
#: half, so eight seconds give fifteen averaged segments.
SEGMENT_S = 1.0
#: The EEG frequency bands highlighted in the spectrum:
#: (symbol, name, low Hz, high Hz).
BANDS = (("δ", "delta", 0.5, 4.0),
         ("θ", "theta", 4.0, 8.0),
         ("α", "alpha", 8.0, 13.0),
         ("β", "beta", 13.0, 30.0),
         ("γ", "gamma", 30.0, 100.0))
#: Highest frequency shown, in Hz: the top of the gamma band.
MAX_FREQUENCY = BANDS[-1][3]


def decibels(psd: np.ndarray) -> np.ndarray:
    """Power density in dB relative to 1 µV²/Hz; a flat signal's zeros
    sit at a floor instead of minus infinity."""
    return 10 * np.log10(np.maximum(psd, 1e-12))


def average_psd(eeg: np.ndarray, fs: float = SAMPLING_RATE,
                max_frequency: float = MAX_FREQUENCY
                ) -> tuple[np.ndarray, np.ndarray]:
    """The power spectral density averaged across channels.

    Args:
        eeg: (samples, channels) in microvolts, oldest first.
        fs: Sampling rate in Hz.
        max_frequency: Highest frequency kept, capped at fs / 2.

    Returns:
        (frequencies in Hz, density in µV²/Hz). Both are empty until
        there is a whole segment of EEG.
    """
    nperseg = int(SEGMENT_S * fs)
    if eeg.ndim != 2 or eeg.shape[0] < nperseg or eeg.shape[1] == 0:
        return np.empty(0), np.empty(0)
    # Linear detrending keeps the electrodes' offsets and drifts out of
    # the lowest bins.
    freqs, psd = signal.welch(eeg, fs=fs, window="hann", nperseg=nperseg,
                              noverlap=nperseg // 2, detrend="linear",
                              axis=0)
    keep = freqs <= min(max_frequency, fs / 2)
    return freqs[keep], psd[keep].mean(axis=1)
