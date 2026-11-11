"""A simulated Unicorn Hybrid Black, for trying brAIn dump without a
headset.

:class:`SimulatedAmplifier` stands in for gtec-unicorn's ``Amplifier``:
the same calls, the same 17 channels in the same order, and scans that
reach the host in bursts of 8, 40 ms after they were acquired, as the
radio delivers them. A scan the radio loses is filled in, with its
Validation Indicator at 0, as the driver does. Everything above it --
g.Pype's own HybridBlack source and the rest of the pipeline -- is the
real code: :class:`SimulatedHybridBlack` is HybridBlack, opened on the
simulated amplifier.

The signal: alpha (10 Hz) waxing and waning over background noise, a
large DC offset per electrode as on real dry electrodes, 50 Hz line
noise, a blink on the frontal channels now and then, gravity on the
accelerometer, a slowly draining battery, and a lost scan every so
often. Electrode 8 is noisy, so the signal quality scope has something
to show. With the test signal on, every electrode carries a square wave
instead, as the headset's internal test signal is rectangular.

(From Rhino Recorder, the ITU brAIn lab's Unicorn recorder, MIT.)
"""
from __future__ import annotations

import threading
import time
from collections import deque
from types import SimpleNamespace
from typing import NamedTuple, Optional

import numpy as np
from gpype.backend.sources import hybrid_black
from gpype.backend.sources.hybrid_black import HybridBlack

SIMULATED_SERIAL = "UN-SIMULATED"

SAMPLING_RATE = 250
NUM_EEG = 8
#: gtec-unicorn's channels, in configuration and scan order.
CHANNEL_NAMES = (
    "EEG 1", "EEG 2", "EEG 3", "EEG 4", "EEG 5", "EEG 6", "EEG 7", "EEG 8",
    "Accelerometer X", "Accelerometer Y", "Accelerometer Z",
    "Gyroscope X", "Gyroscope Y", "Gyroscope Z",
    "Battery Level", "Counter", "Validation Indicator",
)
#: Floats the driver's ring holds; a read may ask for this over the width.
RING_CAPACITY_FLOATS = 8499

#: Fraction of scans the simulated radio loses.
LOSS_RATE = 1 / 4000
#: Scans per Bluetooth burst, and how long after its last scan was acquired
#: a burst reaches the host; 40 ms is g.tec's figure for this headset.
BURST_SCANS = 8
LINK_DELAY_S = 0.040
#: Up to this much more delay, at random, per burst.
LINK_JITTER_S = 0.005
#: The test signal, as UN-2024.08.41 delivers it: a square wave between
#: these two levels (uV), the same on every electrode, with a period of
#: 128 scans (0.512 s, 1.95 Hz) and half of it at each level.
TEST_SIGNAL_HIGH_UV = 3633.0
TEST_SIGNAL_LOW_UV = -3846.5
TEST_SIGNAL_PERIOD_SCANS = 128


class SignalModel:
    """Synthetic EEG and motion, one scan at a time."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        # Dry electrodes sit at a large offset, never near zero.
        self.offsets = (self.rng.choice([-1, 1], NUM_EEG)
                        * self.rng.uniform(5000, 30000, NUM_EEG))
        self.alpha_phase = self.rng.uniform(0, 2 * np.pi, NUM_EEG)
        self.alpha_gain = np.array([0.5, 0.7, 0.8, 0.7, 1.0, 1.2, 1.4, 1.2])
        self.noise_sd = np.full(NUM_EEG, 3.0)
        self.noise_sd[7] = 90.0
        self.drift = np.zeros(NUM_EEG)
        self.blink_left = 0

    def scan(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = n / SAMPLING_RATE
        envelope = 6 + 5 * np.sin(2 * np.pi * 0.1 * t)
        eeg = (self.offsets
               + envelope * self.alpha_gain
               * np.sin(2 * np.pi * 10 * t + self.alpha_phase)
               + 15 * np.sin(2 * np.pi * 50 * t)
               + self.rng.normal(0, self.noise_sd))
        self.drift = 0.999 * self.drift + self.rng.normal(0, 0.5, NUM_EEG)
        eeg += self.drift
        if self.blink_left == 0 and self.rng.random() < 1 / 1500:
            self.blink_left = 100
        if self.blink_left:
            phase = (100 - self.blink_left) / 100
            eeg[:3] += 120 * np.sin(np.pi * phase) * np.array([1, 0.6, 0.6])
            self.blink_left -= 1
        accel = np.array([0.02, -0.05, 1.0]) + self.rng.normal(0, 0.003, 3)
        gyro = self.rng.normal(0, 0.4, 3)
        return eeg, accel, gyro


class StreamStatus(NamedTuple):
    available_scans: int
    dropped_scans: int


class AttestationUnavailable(RuntimeError):
    """A simulated headset cannot attest: g.Pype runs it unattested."""


class SimulatedAmplifier:
    """Stands in for ``gtec_unicorn.Amplifier``."""

    SAMPLING_RATE = SAMPLING_RATE
    EEG_CHANNELS = NUM_EEG
    TOTAL_CHANNELS = len(CHANNEL_NAMES)
    CHANNEL_NAMES = CHANNEL_NAMES
    #: How long get_data waits for scans before giving up, in seconds.
    READ_TIMEOUT_S = 2.0

    @staticmethod
    def get_connected_devices(rescan: bool = False) -> list:
        return [SIMULATED_SERIAL]

    def __init__(self, serial: Optional[str] = None,
                 seed: Optional[int] = None):
        self.serial_number = serial or SIMULATED_SERIAL
        self.firmware_version = "simulated"
        self.device_version = "simulated"
        self.model = SignalModel(seed)
        self._enabled = [True] * len(CHANNEL_NAMES)
        self._scans: deque = deque()
        self._ready = threading.Condition()
        self._streaming = False
        self._thread: Optional[threading.Thread] = None
        self._counter = 0
        self.battery = 100.0
        self.test_signal = False
        #: perf_counter() at which scan 0 was acquired; scan k was
        #: acquired k / SAMPLING_RATE later. None until streaming starts.
        self.started_at: Optional[float] = None

    # -- gtec_unicorn.Amplifier's calls ---------------------------------

    def configure(self, eeg_channels=NUM_EEG, accelerometer=True,
                  gyroscope=True, battery=True, counter=True,
                  validation=True) -> list:
        eeg = (set(range(eeg_channels)) if isinstance(eeg_channels, int)
               else {c - 1 for c in eeg_channels})
        self._enabled = ([i in eeg for i in range(NUM_EEG)]
                         + [bool(accelerometer)] * 3 + [bool(gyroscope)] * 3
                         + [bool(battery), bool(counter), bool(validation)])
        return self.channel_names

    @property
    def channel_names(self) -> list:
        return [n for n, on in zip(CHANNEL_NAMES, self._enabled) if on]

    @property
    def no_of_acquired_channels(self) -> int:
        return sum(self._enabled)

    @property
    def max_scans_per_read(self) -> int:
        return RING_CAPACITY_FLOATS // max(1, self.no_of_acquired_channels)

    @property
    def is_streaming(self) -> bool:
        return self._streaming

    def start(self, test_signal: bool = False) -> None:
        self.test_signal = bool(test_signal)
        if not self._streaming:
            self._streaming = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._streaming = False
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1)
        with self._ready:
            self._ready.notify_all()

    def close(self) -> None:
        self.stop()

    def get_data(self, num_scans: int, out=None):
        deadline = time.monotonic() + self.READ_TIMEOUT_S
        with self._ready:
            while len(self._scans) < num_scans:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._streaming:
                    raise ValueError("The simulated Unicorn delivered no "
                                     "data.")
                self._ready.wait(remaining)
            scans = np.array([self._scans.popleft()
                              for _ in range(num_scans)])
        block = scans[:, self._enabled].astype(np.float32)
        if out is None:
            return block
        out[...] = block
        return out

    def get_stream_status(self) -> StreamStatus:
        with self._ready:
            return StreamStatus(len(self._scans), 0)

    def attest(self, nonce: bytes) -> bytes:
        raise AttestationUnavailable("a simulated Unicorn cannot attest")

    # -- the simulated headset ------------------------------------------------

    def _scan(self, previous: Optional[np.ndarray]) -> np.ndarray:
        """The next scan, all 17 channels; a lost one filled in from the
        previous scan, as the driver does, with validation 0."""
        counter = self._counter
        self._counter += 1
        if counter and counter % (SAMPLING_RATE * 120) == 0:
            self.battery = max(1.0, self.battery - 1.0)
        eeg, accel, gyro = self.model.scan(counter)
        if previous is not None and self.model.rng.random() < LOSS_RATE:
            scan = previous.copy()
            scan[15], scan[16] = counter, 0.0
            return scan
        if self.test_signal:
            high = counter % TEST_SIGNAL_PERIOD_SCANS < TEST_SIGNAL_PERIOD_SCANS // 2
            eeg = np.full(NUM_EEG, TEST_SIGNAL_HIGH_UV if high
                          else TEST_SIGNAL_LOW_UV)
        return np.concatenate((eeg, accel, gyro,
                               [self.battery, counter, 1.0]))

    def _run(self) -> None:
        start = self.started_at = time.perf_counter()
        sent = 0
        rng = self.model.rng
        previous = None
        while self._streaming:
            # The next burst: BURST_SCANS scans, delivered LINK_DELAY_S
            # (plus jitter) after the last of them was acquired.
            last = sent + BURST_SCANS - 1
            due = (start + last / SAMPLING_RATE + LINK_DELAY_S
                   + rng.uniform(0, LINK_JITTER_S))
            while self._streaming and time.perf_counter() < due:
                time.sleep(min(0.002, max(0.0, due - time.perf_counter())))
            burst = []
            for _ in range(BURST_SCANS):
                previous = self._scan(previous)
                burst.append(previous)
                sent += 1
            if self._streaming:
                with self._ready:
                    self._scans.extend(burst)
                    self._ready.notify_all()


class SimulatedHybridBlack(HybridBlack):
    """g.Pype's HybridBlack, opened on a simulated amplifier."""

    def __init__(self, *args, seed: Optional[int] = None, **kwargs):
        kwargs.setdefault("serial", SIMULATED_SERIAL)
        super().__init__(*args, **kwargs)
        self._seed = seed

    def _connect(self) -> None:
        # HybridBlack opens the device through its module's `unicorn`, a
        # name it leaves alone once filled ("a test that has replaced
        # unicorn with a stand-in keeps it"); swap in the simulation only
        # for the open, so a real headset in the same process is unaffected.
        seed = self._seed
        stand_in = SimpleNamespace(Amplifier=type(
            "Amplifier", (SimulatedAmplifier,),
            {"__init__": lambda s, serial=None:
                SimulatedAmplifier.__init__(s, serial, seed)}))
        real = hybrid_black.unicorn
        hybrid_black.unicorn = stand_in
        try:
            super()._connect()
        finally:
            hybrid_black.unicorn = real
