"""Signal quality per electrode, as the signal quality scope shows it.

Like Unicorn Recorder's, it filters the raw EEG to the band where EEG
lives and judges each electrode by how much the filtered signal varies:
an electrode is good while the variation stays in the range EEG has.
Too little means a flat or saturated channel, too much means a loose
electrode, movement or mains pickup. The filters need time to settle, so
no judgement is made during the first seconds.

The band alone is fooled by an electrode that touches no skin: its input
floats, and the raw signal drifts by tens of millivolts a second, all of
it far below the band, so what is left after filtering looks like quiet
EEG. Two things give it away. An electrode in contact holds a steady
offset, so a channel whose raw signal swings by more than
``RAW_SWING_LIMIT_UV`` within the window is bad. And a floating input
picks up only what is common to all of them, so a channel whose filtered
signal follows the average of the others almost exactly (correlation
above ``COMMON_MODE_LIMIT``) is bad too.

Measured on a Unicorn Hybrid Black (UN-2024.10.18), in 2 s windows:

=================  =============  ==============  ==========================
                   filtered SD    raw swing       correlation with others
=================  =============  ==============  ==========================
worn, 60 s         5-54 uV        0.4-16 mV       at most 0.88
                                  (mostly < 1)
lying on a desk    2.7-4.2 uV     15-18 mV        1.00 on every channel
=================  =============  ==============  ==========================

(From Rhino Recorder, the ITU brAIn lab's Unicorn recorder, MIT.)
"""
from __future__ import annotations

import numpy as np
from scipy import signal

NUM_EEG, SAMPLING_RATE = 8, 250

BAND = (1.0, 30.0)
#: Seconds of filtered signal judged.
WINDOW_S = 2.0
#: Seconds after the start before any judgement.
SETTLE_S = 5.0
#: Standard deviation of good EEG, in microvolts.
GOOD_RANGE_UV = (0.5, 40.0)
#: Largest swing of the raw signal within the window, in microvolts.
RAW_SWING_LIMIT_UV = 10000.0
#: Largest correlation of a channel with the average of the others.
COMMON_MODE_LIMIT = 0.98

UNKNOWN, GOOD, BAD = "unknown", "good", "bad"


class SignalQuality:
    """Feed raw EEG with :meth:`update`, read the verdicts from
    :meth:`verdicts`."""

    def __init__(self, fs: float = SAMPLING_RATE, channels: int = NUM_EEG):
        self.fs = fs
        self._sos = np.vstack((
            signal.butter(4, BAND, btype="bandpass", fs=fs, output="sos"),
            signal.tf2sos(*signal.iirnotch(50, 30, fs=fs)),
            signal.tf2sos(*signal.iirnotch(60, 30, fs=fs)),
        ))
        self.channels = channels
        self.reset()

    def reset(self) -> None:
        self._z = None
        self._window = np.zeros((0, self.channels))
        self._raw = np.zeros((0, self.channels))
        self._seen = 0

    def update(self, eeg: np.ndarray) -> None:
        """Add raw EEG samples, shape (n, channels), oldest first."""
        if eeg.size == 0:
            return
        if self._z is None:
            self._z = (signal.sosfilt_zi(self._sos)[:, :, None]
                       * eeg[0][None, None, :])
        filtered, self._z = signal.sosfilt(self._sos, eeg, axis=0,
                                           zi=self._z)
        keep = int(WINDOW_S * self.fs)
        self._window = np.vstack((self._window, filtered))[-keep:]
        self._raw = np.vstack((self._raw, eeg))[-keep:]
        self._seen += eeg.shape[0]

    def deviation(self) -> np.ndarray:
        """Standard deviation of each filtered channel, in microvolts."""
        if len(self._window) == 0:
            return np.full(self.channels, np.nan)
        return self._window.std(axis=0)

    def swing(self) -> np.ndarray:
        """Peak-to-peak of each raw channel over the window, in uV."""
        if len(self._raw) == 0:
            return np.full(self.channels, np.nan)
        return np.ptp(self._raw, axis=0)

    def common_mode(self) -> np.ndarray:
        """Correlation of each filtered channel with the average of the
        others over the window; NaN where a channel is flat."""
        x = self._window
        result = np.full(self.channels, np.nan)
        if len(x) < 2 or self.channels < 2:
            return result
        total = x.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            for c in range(self.channels):
                others = (total - x[:, c]) / (self.channels - 1)
                result[c] = np.corrcoef(x[:, c], others)[0, 1]
        return result

    def verdicts(self) -> list[str]:
        if self._seen < SETTLE_S * self.fs:
            return [UNKNOWN] * self.channels
        low, high = GOOD_RANGE_UV
        verdicts = []
        for sd, swing, common in zip(self.deviation(), self.swing(),
                                     self.common_mode()):
            good = (low <= sd <= high and swing <= RAW_SWING_LIMIT_UV
                    and not common > COMMON_MODE_LIMIT)
            verdicts.append(GOOD if good else BAD)
        return verdicts
