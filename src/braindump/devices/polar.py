"""The Polar H10 chest strap, over Bluetooth Low Energy (bleak).

Two of the strap's services are read:

- **Polar Measurement Data (PMD)**: the raw single-lead ECG at 130 Hz
  (microvolts) and, optionally, the accelerometer (25-200 Hz). Started
  as PolarBand2lsl does, by writing the stream's settings to the PMD
  control point.
- **Heart Rate (0x180D)**: the strap's heart rate, its skin contact, and
  the RR intervals of every beat, about once a second.

ECG and acceleration enter g.Pype pipelines through push sources, so the
view and the recording work as for the other devices. Each PMD frame
carries the sensor's time stamp of its last sample: samples are spaced by
the sensor's clock and moved onto this computer's clock by the smallest
delay seen. Beats (from the RR intervals) are recorded as physioevents
of the ECG.

BIDS: ``recording-ecg_physio`` (``cardiac`` in µV), optionally
``recording-accel_physio`` (g), and ``recording-ecg_physioevents`` with
one ``beat`` per RR interval.
"""
from __future__ import annotations

import asyncio
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import gpype as gp
import numpy as np

from .. import __version__, clock
from ..bids import NA, write_json
from .base import (IDLE, RUNNING, STARTING, STOPPING, Device, DeviceConfig,
                   RecordedFiles, RecordingTarget, Timeline,
                   start_time)
from ..pipeline import ClockMap, PhysioWriter, PushSource, SweepTap

# -- the protocol ----------------------------------------------------------------

PMD_CONTROL = "fb005c81-02e7-f387-1cad-8acd2d8df0c8"
PMD_DATA = "fb005c82-02e7-f387-1cad-8acd2d8df0c8"
HR_MEASUREMENT = "00002a37-0000-1000-8000-00805f9b34fb"
BATTERY_LEVEL = "00002a19-0000-1000-8000-00805f9b34fb"
FIRMWARE = "00002a26-0000-1000-8000-00805f9b34fb"
SERIAL = "00002a25-0000-1000-8000-00805f9b34fb"

ECG, ACC = 0x00, 0x02
ECG_RATE = 130.0
ACC_RATES = (25, 50, 100, 200)
ACC_RANGES = (2, 4, 8)


def ecg_start() -> bytes:
    """Start ECG: 130 Hz, 14 bits (as PolarBand2lsl writes it)."""
    return bytes([0x02, ECG, 0x00, 0x01, 0x82, 0x00, 0x01, 0x01, 0x0E,
                  0x00])


def acc_start(rate: int, range_g: int) -> bytes:
    """Start acceleration: ``rate`` Hz, 16 bits, ±``range_g`` g."""
    return bytes([0x02, ACC, 0x00, 0x01, rate & 0xFF, rate >> 8, 0x01, 0x01,
                  0x10, 0x00, 0x02, 0x01, range_g, 0x00])


def pmd_stop(kind: int) -> bytes:
    return bytes([0x03, kind])


@dataclass
class PmdFrame:
    kind: int
    #: Sensor time of the frame's last sample, in seconds.
    sensor_time: float
    #: (samples, channels): µV for ECG, mG for acceleration.
    samples: np.ndarray


def parse_pmd(data: bytes) -> Optional[PmdFrame]:
    """A PMD data notification, or None for one that cannot be read
    (delta-compressed frames, which the H10 does not send by default)."""
    if len(data) < 10:
        return None
    kind, frame_type = data[0], data[9]
    stamp = int.from_bytes(data[1:9], "little") / 1e9
    payload = bytes(data[10:])
    if frame_type & 0x80:
        return None
    if kind == ECG and frame_type == 0x00:
        width, channels = 3, 1
    elif kind == ACC and frame_type in (0x00, 0x01, 0x02):
        width, channels = frame_type + 1, 3
    else:
        return None
    step = width * channels
    n = len(payload) // step
    values = np.empty((n, channels))
    for i in range(n):
        for c in range(channels):
            at = i * step + c * width
            values[i, c] = int.from_bytes(payload[at:at + width], "little",
                                          signed=True)
    return PmdFrame(kind, stamp, values)


@dataclass
class HeartRate:
    bpm: int
    #: None when the strap cannot tell.
    contact: Optional[bool]
    #: RR intervals in this notification, in seconds.
    rr: list[float]


def parse_heart_rate(data: bytes) -> HeartRate:
    """A Heart Rate Measurement (0x2A37) notification."""
    flags = data[0]
    i = 1
    if flags & 0x01:
        bpm = int.from_bytes(data[i:i + 2], "little")
        i += 2
    else:
        bpm = data[i]
        i += 1
    contact = bool(flags & 0x02) if flags & 0x04 else None
    if flags & 0x08:
        i += 2  # energy expended
    rr = []
    if flags & 0x10:
        while i + 1 < len(data):
            rr.append(int.from_bytes(data[i:i + 2], "little") / 1024.0)
            i += 2
    return HeartRate(bpm, contact, rr)


class BeatClock:
    """Times of beats from RR intervals: each notification's intervals
    continue from the previous beat while that stays within a beat of
    when they arrived, and are anchored back from their arrival otherwise
    (at the start, or after a gap)."""

    def __init__(self):
        self.last: Optional[float] = None

    def beats(self, rr: list[float], received: float) -> list[float]:
        if not rr:
            return []
        if self.last is not None:
            times, t = [], self.last
            for r in rr:
                t += r
                times.append(t)
            if abs(times[-1] - received) < max(1.5, rr[-1] * 1.5) and \
                    times[-1] <= received + 0.05:
                self.last = times[-1]
                return times
        times, t = [], received
        for r in reversed(rr):
            times.append(t)
            t -= r
        times.reverse()
        self.last = times[-1]
        return times


# -- configuration ------------------------------------------------------------------

@dataclass
class PolarConfig(DeviceConfig):
    #: The Bluetooth link's minimum delay, in ms, taken off the time
    #: stamps when known (0: samples are placed when the fastest reading
    #: arrived).
    latency_ms: float = 0.0
    #: The strap's name (e.g. "Polar H10 A1B2C3D4") or Bluetooth address;
    #: empty connects to the first Polar H10 found.
    device: str = ""
    simulate: bool = False
    accelerometer: bool = False
    acc_rate: int = 25
    acc_range: int = 8
    #: Seconds a search waits for straps to answer.
    scan_seconds: float = 6.0


def find_straps(timeout: float = 6.0) -> list[tuple[str, str]]:
    """(name, address) of the Polar H10 straps in range."""
    from bleak import BleakScanner

    async def scan():
        found = await BleakScanner.discover(timeout=timeout)
        return [(d.name or "", d.address) for d in found
                if (d.name or "").startswith("Polar H10")]
    return asyncio.run(scan())


# -- the link ---------------------------------------------------------------------------

class _Link:
    """The Bluetooth connection, on its own thread with its own event loop.
    Calls ``on_pmd(frame, received)``, ``on_hr(hr, received)``."""

    def __init__(self, config: PolarConfig,
                 on_pmd: Callable[[PmdFrame, float], None],
                 on_hr: Callable[[HeartRate, float], None],
                 on_error: Callable[[str], None]):
        self.config = config
        self.on_pmd, self.on_hr, self.on_error = on_pmd, on_hr, on_error
        self.info: dict[str, str] = {}
        self.battery: Optional[int] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop: Optional[asyncio.Event] = None
        self._ready = threading.Event()
        self._failure: Optional[BaseException] = None
        self._thread: Optional[threading.Thread] = None

    def start(self, timeout: float = 30.0) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="polar-h10")
        self._thread.start()
        if not self._ready.wait(timeout):
            self.stop()
            raise TimeoutError("The Polar H10 did not connect in time.")
        if self._failure is not None:
            raise self._failure

    def stop(self) -> None:
        loop, stop = self._loop, self._stop
        if loop is not None and stop is not None:
            loop.call_soon_threadsafe(stop.set)
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None

    def _run(self):
        loop = asyncio.new_event_loop()
        self._loop = loop
        try:
            loop.run_until_complete(self._main())
        except BaseException as error:  # noqa: BLE001
            if not self._ready.is_set():
                self._failure = error
                self._ready.set()
            else:
                self.on_error(f"The Polar H10 disconnected: {error}")
        finally:
            loop.close()

    async def _find(self):
        from bleak import BleakScanner
        wanted = self.config.device.strip()
        deadline = time.time() + max(10.0, self.config.scan_seconds * 2)
        while time.time() < deadline:
            found = await BleakScanner.discover(
                timeout=self.config.scan_seconds)
            for d in found:
                name = d.name or ""
                if wanted and wanted in (name, d.address):
                    return d
                if not wanted and name.startswith("Polar H10"):
                    return d
        raise ConnectionError(
            f"No Polar H10 {'named ' + wanted if wanted else ''} found. Wear "
            f"the strap (it switches on with skin contact) and keep it close.")

    async def _main(self):
        from bleak import BleakClient
        self._stop = asyncio.Event()
        device = await self._find()
        lost = asyncio.Event()
        async with BleakClient(
                device, disconnected_callback=lambda _: lost.set()) as client:
            for key, uuid in (("firmware", FIRMWARE), ("serial", SERIAL)):
                try:
                    value = await client.read_gatt_char(uuid)
                    self.info[key] = bytes(value).decode(errors="replace")
                except Exception:  # noqa: BLE001
                    pass
            self.info["name"] = device.name or ""
            self.info["address"] = device.address
            await self._read_battery(client)

            def pmd(_, data):
                frame = parse_pmd(bytes(data))
                if frame is not None:
                    self.on_pmd(frame, clock.now())

            def hr(_, data):
                self.on_hr(parse_heart_rate(bytes(data)), clock.now())

            await client.start_notify(PMD_CONTROL, lambda *_: None)
            await client.start_notify(PMD_DATA, pmd)
            await client.write_gatt_char(PMD_CONTROL, ecg_start(),
                                         response=True)
            if self.config.accelerometer:
                await asyncio.sleep(0.5)
                await client.write_gatt_char(
                    PMD_CONTROL, acc_start(self.config.acc_rate,
                                           self.config.acc_range),
                    response=True)
            await client.start_notify(HR_MEASUREMENT, hr)
            self._ready.set()
            while not self._stop.is_set():
                if lost.is_set():
                    raise ConnectionError("Bluetooth link lost")
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=30)
                except asyncio.TimeoutError:
                    await self._read_battery(client)
            for kind in (ECG, ACC):
                try:
                    await client.write_gatt_char(PMD_CONTROL, pmd_stop(kind),
                                                 response=True)
                except Exception:  # noqa: BLE001
                    pass

    async def _read_battery(self, client):
        try:
            self.battery = int((await client.read_gatt_char(BATTERY_LEVEL))[0])
        except Exception:  # noqa: BLE001
            pass


class _Simulated:
    """A simulated strap: an ECG with P, QRS and T waves around 65 beats a
    minute with some variability, its RR intervals once a second, and
    gravity on the accelerometer."""

    def __init__(self, config: PolarConfig, on_pmd, on_hr, on_error):
        self.config = config
        self.on_pmd, self.on_hr = on_pmd, on_hr
        self.info = {"name": "Polar H10 SIMULATED", "serial": "SIMULATED",
                     "firmware": "simulated"}
        self.battery = 87
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self, timeout: float = 0):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        rng = np.random.default_rng()
        t0 = clock.now()
        sensor0 = 7.0e8  # the strap's own clock
        n = 0
        next_beat, beats = t0 + 0.3, []
        pending_rr, last_beat = [], None
        next_hr = t0 + 1.0
        acc_n, acc_period = 0, 1.0 / self.config.acc_rate
        while self._running:
            time.sleep(73 / ECG_RATE)  # the H10 sends 73 samples a frame
            now = clock.now()
            samples = []
            while (n + 1) / ECG_RATE <= now - t0:
                t = t0 + n / ECG_RATE
                while t >= next_beat:
                    beats.append(next_beat)
                    if last_beat is not None:
                        pending_rr.append(next_beat - last_beat)
                    last_beat = next_beat
                    rr = 60 / 65 + 0.05 * math.sin(next_beat / 4) \
                        + rng.normal(0, 0.02)
                    next_beat += rr
                beats = beats[-3:]
                v = rng.normal(0, 8)
                for b in beats:
                    d = t - b
                    v += (120 * math.exp(-((d + 0.16) / 0.025) ** 2)
                          + 1100 * math.exp(-(d / 0.011) ** 2)
                          - 180 * math.exp(-((d - 0.025) / 0.012) ** 2)
                          + 260 * math.exp(-((d - 0.28) / 0.05) ** 2))
                samples.append(v)
                n += 1
            if samples:
                stamp = sensor0 + (n - 1) / ECG_RATE
                self.on_pmd(PmdFrame(ECG, stamp,
                                     np.array(samples)[:, None]),
                            now + 0.01)
            if self.config.accelerometer:
                acc = []
                while (acc_n + 1) * acc_period <= now - t0:
                    acc.append([rng.normal(0, 5), rng.normal(0, 5),
                                1000 + rng.normal(0, 5)])
                    acc_n += 1
                if acc:
                    self.on_pmd(PmdFrame(ACC, sensor0 + (acc_n - 1)
                                         * acc_period, np.array(acc)),
                                now + 0.01)
            if now >= next_hr:
                next_hr += 1.0
                rr, pending_rr = pending_rr, []
                bpm = round(60 / np.mean(rr)) if rr else 65
                self.on_hr(HeartRate(bpm, True, rr), now)


# -- BIDS --------------------------------------------------------------------------------

class BeatWriter(PhysioWriter):
    """The beats as BIDS physioevents: onset (clock time), no duration,
    "beat", the RR interval that ended at it, and the heart rate."""

    def _write(self, block, first_index):
        for row in block:
            t = self.epoch + float(row[0])
            if self.first_sample is None:
                self.first_sample = t
            self.last_sample = t
            self._writer.write([t, None, "beat", round(float(row[1]), 4),
                                int(round(float(row[2])))])


@dataclass
class Graph:
    """The strap's nodes in the session pipeline."""
    ecg: PushSource
    ecg_tap: SweepTap
    ecg_writer: PhysioWriter
    beats: PushSource
    beat_writer: BeatWriter
    acc: Optional[PushSource] = None
    acc_writer: Optional[PhysioWriter] = None


class PolarDevice(Device):
    TYPE = "polar"
    NAME = "Polar H10"
    MANUFACTURER = "Polar Electro Oy"
    MODEL = "H10"
    DESCRIPTION = "Chest strap: ECG at 130 Hz, heart rate and RR " \
                  "intervals, over Bluetooth"
    MODALITY = "physio"
    CONFIG = PolarConfig

    def __init__(self, config: PolarConfig, project_folder: Path,
                 simulate: bool = False):
        super().__init__(config, project_folder, simulate)
        self.graph: Optional[Graph] = None
        self.link = None
        self.heart_rate: Optional[int] = None
        self.contact: Optional[bool] = None
        #: Recent beats: (time, RR interval s, heart rate), for the view.
        self.beats: deque[tuple[float, float, int]] = deque(maxlen=600)
        self._target: Optional[RecordingTarget] = None
        self._clocks = {ECG: ClockMap(), ACC: ClockMap()}
        self._last_stamp: dict[int, float] = {}
        self._beat_clock = BeatClock()
        self._beats_lock = threading.Lock()

    # -- data in -----------------------------------------------------------------

    def _on_pmd(self, frame: PmdFrame, received: float) -> None:
        graph = self.graph
        if graph is None:
            return
        source = graph.ecg if frame.kind == ECG else graph.acc
        if source is None:
            return
        n = len(frame.samples)
        rate = ECG_RATE if frame.kind == ECG else float(self.config.acc_rate)
        previous = self._last_stamp.get(frame.kind)
        self._last_stamp[frame.kind] = frame.sensor_time
        step = 1.0 / rate
        if previous is not None and n:
            measured = (frame.sensor_time - previous) / n
            if 0.5 * step < measured < 1.5 * step:
                step = measured
        last = self._clocks[frame.kind](frame.sensor_time, received)
        for i, values in enumerate(frame.samples):
            source.push([last - (n - 1 - i) * step, *values])

    def _on_hr(self, hr: HeartRate, received: float) -> None:
        self.heart_rate = hr.bpm
        self.contact = hr.contact
        times = self._beat_clock.beats(hr.rr, received)
        graph = self.graph
        with self._beats_lock:
            for t, rr in zip(times, hr.rr):
                beat = (t, rr, round(60 / rr) if rr else hr.bpm)
                self.beats.append(beat)
                if graph is not None:
                    graph.beats.push(list(beat))

    # -- the session pipeline ----------------------------------------------------

    def build(self, pipeline: gp.Pipeline) -> None:
        cfg = self.config
        ecg = PushSource(["ECG"], ["uV"], ECG_RATE,
                         name=self.node_name("ecg"))
        tap = SweepTap(seconds=8, name=self.node_name("ecg_view"))
        ecg_writer = PhysioWriter(self.node_name("ecg_bids"))
        pipeline.connect(ecg, tap)
        pipeline.connect(ecg, ecg_writer)
        # One row per beat, about one a second.
        beats = PushSource(["RR", "HR"], ["s", "bpm"], 1.0,
                           name=self.node_name("beats"))
        beat_writer = BeatWriter(self.node_name("beat_bids"))
        pipeline.connect(beats, beat_writer)
        graph = Graph(ecg, tap, ecg_writer, beats, beat_writer)
        if cfg.accelerometer:
            acc = PushSource(["ACC X", "ACC Y", "ACC Z"], ["mG"] * 3,
                             float(cfg.acc_rate), name=self.node_name("acc"))
            acc_writer = PhysioWriter(self.node_name("acc_bids"),
                                      scale=0.001)
            pipeline.connect(acc, acc_writer)
            graph.acc, graph.acc_writer = acc, acc_writer
        self.graph = graph

    def opened(self) -> None:
        cfg = self.config
        latency = cfg.latency_ms / 1000.0
        self._clocks = {ECG: ClockMap(latency_s=latency),
                        ACC: ClockMap(latency_s=latency)}
        self._last_stamp = {}
        self._beat_clock = BeatClock()
        self.beats.clear()
        self.heart_rate = self.contact = None
        cls = _Simulated if self.simulated else _Link
        link = cls(cfg, self._on_pmd, self._on_hr, self._fail)
        link.start()
        self.link = link

    def closing(self) -> None:
        link, self.link = self.link, None
        if link is not None:
            link.stop()

    def closed(self) -> None:
        self.graph = None
        self._target = None

    # -- recording ---------------------------------------------------------------------

    @property
    def recording(self) -> bool:
        graph = self.graph
        return bool(graph and graph.ecg_writer.recording)

    def _writers(self):
        graph = self.graph
        out = [("ecg", graph.ecg, graph.ecg_writer)]
        if graph.acc is not None:
            out.append(("accel", graph.acc, graph.acc_writer))
        return out

    def begin_recording(self, target: RecordingTarget) -> list[Path]:
        with self._lock:
            graph = self.graph
            if graph is None:
                raise RuntimeError("The Polar H10 is not connected.")
            paths = []
            for recording, source, writer in self._writers():
                path = target.dataset.path(target.entities, target.datatype,
                                           "physio", ".tsv.gz",
                                           recording=recording)
                self.invoke(writer, "begin", {
                    "path": str(path), "meta": {"epoch": source.epoch}})
                paths.append(path)
            events = paths[0].with_name(paths[0].name.replace(
                "_physio.tsv.gz", "_physioevents.tsv.gz"))
            self.invoke(graph.beat_writer, "begin", {
                "path": str(events), "meta": {"epoch": graph.beats.epoch}})
            self._target = target
            return paths + [events]

    def end_recording(self) -> RecordedFiles:
        with self._lock:
            target, graph = self._target, self.graph
            if target is None or graph is None:
                return RecordedFiles()
            # The strap sends a frame about every half second.
            time.sleep(0.6)
            written = {}
            for recording, _, writer in self._writers():
                summary = self.invoke(writer, "end")
                written[recording] = (Path(writer.path), summary.get("first"),
                                      summary.get("last"),
                                      int(summary.get("samples") or 0))
            self.invoke(graph.beat_writer, "end")
            events_path = Path(graph.beat_writer.path)
            self._target = None
            info = dict(self.link.info) if self.link else {}
            acc_rate = self.config.acc_rate

        first = written["ecg"][1]

        def finalize(reference: Optional[Timeline]) -> None:
            self._write_sidecars(target, written, events_path, info,
                                 acc_rate, reference)

        scans, files = [], [events_path, _json(events_path)]
        for path, start, *_ in written.values():
            scans.append((path, datetime.fromtimestamp(start
                                                       or target.started)))
            files += [path, _json(path)]
        return RecordedFiles(scans=scans, files=files, first_sample=first,
                             finalize=finalize,
                             samples=written["ecg"][3])

    def _write_sidecars(self, target, written, events_path, info, acc_rate,
                        reference) -> None:
        software = [f"brAIn dump {__version__}", f"g.Pype {gp.__version__}"]
        if info.get("firmware"):
            software.append(f"H10 firmware {info['firmware']}")
        timestamp = {"Description": "Time of the sample, seconds since "
                                    "1970-01-01 UTC on the recording "
                                    "computer's clock (the strap's own "
                                    "clock spaces the samples)",
                     "Units": "s"}
        for recording, (path, first, last, samples) in written.items():
            start = start_time(reference, first)
            meta = {
                "PhysioType": "generic",
                "SamplingFrequency": ECG_RATE if recording == "ecg"
                else float(acc_rate),
                "StartTime": round(start, 6),
                "Manufacturer": self.MANUFACTURER,
                "ManufacturersModelName": self.MODEL,
                "DeviceSerialNumber": info.get("serial") or NA,
                "SoftwareVersions": ", ".join(software),
                "TaskName": target.entities.task,
                "RecordingDuration": round((last - first), 3)
                if first is not None and last is not None else 0,
                "SimulatedDevice": self.simulated,
                **target.common_metadata,
                "timestamp": timestamp,
            }
            if recording == "ecg":
                meta["Columns"] = ["timestamp", "cardiac"]
                meta["cardiac"] = {"Description": "ECG, single lead, from "
                                                  "the chest strap's two "
                                                  "electrodes",
                                   "Units": "uV"}
            else:
                meta["Columns"] = ["timestamp", "accel_x", "accel_y",
                                   "accel_z"]
                for axis in "xyz":
                    meta[f"accel_{axis}"] = {
                        "Description": f"Acceleration of the strap, {axis} "
                                       f"axis", "Units": "g"}
            write_json(_json(path), meta)
        write_json(_json(events_path), {
            "Description": "Heart beats, from the RR intervals the strap "
                           "reports. Onsets are when each beat ended its RR "
                           "interval, timed by the intervals and anchored "
                           "on arrival (within about 100 ms).",
            "TaskName": target.entities.task,
            "OnsetSource": "timestamp",
            "Columns": ["onset", "duration", "trial_type", "rr_interval",
                        "heart_rate"],
            "onset": {"Description": "Time of the beat, on the timestamp "
                                     "clock", "Units": "s"},
            "trial_type": {"Description": "Event type",
                           "Levels": {"beat": "a heart beat"}},
            "rr_interval": {"Description": "Interval from the previous "
                                           "beat", "Units": "s"},
            "heart_rate": {"Description": "60 / rr_interval",
                           "Units": "beats/min"},
        })

    # -- status -----------------------------------------------------------------------

    @property
    def battery(self) -> Optional[int]:
        return self.link.battery if self.link else None

    def identity(self) -> str:
        if self.simulated:
            return "Polar H10 (simulated)"
        link = self.link
        if link is not None and link.info.get("name"):
            return link.info["name"]
        return self.config.device or ""

    def status(self) -> str:
        name = self.identity() or "Polar H10"
        if self.state == RUNNING:
            text = "recording" if self.recording else "connected"
            hr = f", {self.heart_rate} bpm" if self.heart_rate else ""
            contact = ", no skin contact" if self.contact is False else ""
            return f"{name}: {text}{hr}{contact}"
        return {IDLE: f"{name}: not connected",
                STARTING: f"{name}: connecting…",
                STOPPING: f"{name}: stopping…"}.get(self.state, self.state)


def _json(path: Path) -> Path:
    return path.with_name(path.name.replace(".tsv.gz", ".json"))
