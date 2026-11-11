"""The Shimmer3 GSR+ unit: skin conductance and, from its ear clip, a
photoplethysmogram (PPG).

The unit is paired over Bluetooth Classic and appears as a serial port
(``/dev/cu.Shimmer3-XXXX`` on macOS, ``COMn`` on Windows,
``/dev/rfcomm0`` on Linux). It is driven with pyshimmer, which talks to
Shimmer's LogAndStream firmware. pyshimmer is GPL-3.0, so it is an
optional extra (``pip install braindump[shimmer]``) imported only when a
real unit is opened; the simulated one needs nothing.

The GSR+ ranges automatically; each sample carries the range it was
measured in, and is converted to skin resistance and conductance with
that range's reference resistor, as Shimmer documents. Samples are
spaced by the unit's 32768 Hz clock and moved onto this computer's clock
by the smallest delay seen.

BIDS: ``recording-gsr_physio`` with ``skin_conductance`` (µS),
``skin_resistance`` (kOhm), ``gsr_range`` and, with the ear clip,
``cardiac`` (the PPG, mV).
"""
from __future__ import annotations

import math
import sys
import threading
import time
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

#: Sampling rates the unit can keep exactly (32768 Hz / an integer).
RATES = (32.0, 51.2, 64.0, 128.0, 256.0)
CLOCK_HZ = 32768.0
#: Reference resistor of each GSR range, in kOhm (Shimmer3 GSR+).
RANGE_KOHM = (40.2, 287.0, 1000.0, 3300.0)
#: What each range is meant for, in kOhm.
RANGE_SPAN = ("10-56 kΩ", "56-220 kΩ", "220-680 kΩ", "680 kΩ-4.7 MΩ")
LABELS = ["GSR", "GSR_RES", "GSR_RANGE", "PPG"]
UNITS = ["uS", "kOhm", "", "mV"]


def gsr_from_raw(raw: int) -> tuple[float, float, int]:
    """A GSR_RAW sample (range in the top two bits, 12-bit ADC value
    below) as (conductance µS, resistance kOhm, range)."""
    rng = (raw >> 14) & 0x03
    adc = raw & 0x0FFF
    volts = adc * 3.0 / 4095.0
    ratio = volts / 0.5 - 1.0
    if ratio <= 0:
        return float("nan"), float("nan"), rng
    resistance = RANGE_KOHM[rng] / ratio
    return 1000.0 / resistance, resistance, rng


def ppg_from_raw(raw: int) -> float:
    """Internal ADC A13 (12 bits, 3 V reference) in mV."""
    return raw * 3000.0 / 4095.0


class TickClock:
    """The unit's 24-bit tick counter, unwrapped, in seconds."""

    WRAP = 1 << 24

    def __init__(self):
        self._last: Optional[int] = None
        self._wraps = 0

    def __call__(self, ticks: int) -> float:
        if self._last is not None and ticks < self._last:
            self._wraps += 1
        self._last = ticks
        return (ticks + self._wraps * self.WRAP) / CLOCK_HZ


@dataclass
class ShimmerConfig(DeviceConfig):
    #: The Bluetooth link's minimum delay, in ms, taken off the time
    #: stamps when known (0: samples are placed when the fastest reading
    #: arrived).
    latency_ms: float = 0.0
    #: Serial port of the paired unit; empty uses the first one whose name
    #: says Shimmer.
    port: str = ""
    simulate: bool = False
    sampling_rate: float = 128.0
    #: Record the ear clip's PPG (internal ADC A13).
    ppg: bool = True


def _kind(device: str, text: str) -> int:
    """0 for a port named after a Shimmer, 1 for another Bluetooth serial
    port, 2 for anything else."""
    text = text.lower()
    if "shimmer" in text:
        return 0
    if "bluetooth" in text or "rfcomm" in device.lower() or \
            "bthenum" in text:
        return 1
    return 2


def find_ports() -> list[tuple[str, str]]:
    """(port, description) of the serial ports: Shimmers first, then the
    other Bluetooth ones. macOS names the port after the unit
    (``/dev/cu.Shimmer3-XXXX``); Windows calls every one "Standard Serial
    over Bluetooth link"; Linux has ``/dev/rfcomm*`` once bound."""
    from serial.tools import list_ports
    ports = []
    for p in list_ports.comports():
        if p.device.startswith("/dev/tty.") and sys.platform == "darwin":
            continue  # its call-out twin, /dev/cu.*, is the one to open
        text = f"{p.device} {p.description or ''} {p.name or ''} " \
               f"{p.hwid or ''}"
        ports.append((_kind(p.device, text), p.device,
                      p.description or ""))
    ports.sort()
    return [(device, desc) for _, device, desc in ports]


def default_port() -> str:
    """The port to use when none is set: the one named after a Shimmer,
    else the only Bluetooth serial port there is."""
    from serial.tools import list_ports
    kinds = {}
    for p in list_ports.comports():
        if p.device.startswith("/dev/tty.") and sys.platform == "darwin":
            continue
        text = f"{p.device} {p.description or ''} {p.name or ''} " \
               f"{p.hwid or ''}"
        kinds.setdefault(_kind(p.device, text), []).append(p.device)
    if kinds.get(0):
        return kinds[0][0]
    if len(kinds.get(1, [])) == 1:
        return kinds[1][0]
    return ""


#: How to give the unit a serial port, by system.
PAIRING_HELP = {
    "darwin": "Pair the unit in System Settings → Bluetooth (PIN 1234); "
              "it then appears as /dev/cu.Shimmer3-XXXX.",
    "win32": "Pair the unit in Settings → Bluetooth & devices (PIN 1234); "
             "Windows gives it an outgoing COM port (see 'More Bluetooth "
             "settings' → COM Ports).",
    "linux": "Pair the unit (bluetoothctl: pair, trust; PIN 1234), then "
             "bind a port: sudo rfcomm bind 0 <its address>. It appears as "
             "/dev/rfcomm0 (your user needs the dialout group).",
}


def pairing_help() -> str:
    key = "linux" if sys.platform.startswith("linux") else sys.platform
    return PAIRING_HELP.get(key, PAIRING_HELP["linux"])


# -- the links ----------------------------------------------------------------------

class _Link:
    """The unit, through pyshimmer. ``on_sample(sensor_seconds, raw_gsr,
    raw_ppg_or_None, received)`` is called for every sample, on
    pyshimmer's read thread."""

    def __init__(self, config: ShimmerConfig, on_sample, on_error):
        self.config = config
        self.on_sample, self.on_error = on_sample, on_error
        self.info: dict[str, str] = {}
        self.battery: Optional[float] = None
        self._shimmer = None
        self._ticks = TickClock()

    def start(self):
        try:
            from pyshimmer import DEFAULT_BAUDRATE, EChannelType, \
                ShimmerBluetooth
            from pyshimmer.dev.channels import ESensorGroup
            from serial import Serial
        except ImportError as error:
            raise RuntimeError(
                "The Shimmer needs pyshimmer (GPL-3.0), an optional extra: "
                "pip install 'braindump[shimmer]'") from error
        port = self.config.port or default_port()
        if not port:
            raise ConnectionError(
                "No Shimmer serial port found: choose its port in the "
                "settings. " + pairing_help())
        serial = Serial(port, DEFAULT_BAUDRATE, timeout=None)
        shimmer = ShimmerBluetooth(serial)
        try:
            shimmer.initialize()
            sensors = [ESensorGroup.GSR]
            if self.config.ppg:
                sensors.append(ESensorGroup.CH_A13)
            shimmer.set_sensors(sensors)
            shimmer.set_sampling_rate(self.config.sampling_rate)
            try:
                kind, version = shimmer.get_firmware_version()
                self.info["firmware"] = f"{kind.name} {version}"
            except Exception:  # noqa: BLE001
                pass
            try:
                self.info["name"] = shimmer.get_device_name()
            except Exception:  # noqa: BLE001
                pass
            self.info["port"] = port
            self._read_battery(shimmer)
            channels = (EChannelType.TIMESTAMP, EChannelType.GSR_RAW,
                        EChannelType.INTERNAL_ADC_13)

            def packet(p):
                received = clock.now()
                try:
                    ticks = p[channels[0]]
                    gsr = p[channels[1]]
                    ppg = p[channels[2]] if self.config.ppg else None
                except (KeyError, ValueError):
                    return
                self.on_sample(self._ticks(ticks), gsr, ppg, received)

            shimmer.add_stream_callback(packet)
            shimmer.start_streaming()
        except BaseException:
            try:
                shimmer.shutdown()
            except Exception:  # noqa: BLE001
                pass
            raise
        self._shimmer = shimmer

    def _read_battery(self, shimmer):
        try:
            self.battery = float(shimmer.get_battery_state(in_percent=True))
        except Exception:  # noqa: BLE001
            pass

    def stop(self):
        shimmer, self._shimmer = self._shimmer, None
        if shimmer is None:
            return
        try:
            shimmer.stop_streaming()
        except Exception:  # noqa: BLE001
            pass
        try:
            shimmer.shutdown()
        except Exception:  # noqa: BLE001
            pass


class _Simulated:
    """Skin conductance around 5 µS, drifting slowly, with a skin
    conductance response every 8-20 s, and an ear-clip pulse at about 70
    beats a minute."""

    def __init__(self, config: ShimmerConfig, on_sample, on_error):
        self.config = config
        self.on_sample = on_sample
        self.info = {"name": "Shimmer3 GSR+ (simulated)",
                     "firmware": "simulated"}
        self.battery = 76.0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        rng = np.random.default_rng()
        rate = self.config.sampling_rate
        t0 = clock.now()
        n = 0
        responses = [t0 + 3.0]
        while self._running:
            time.sleep(0.05)
            now = clock.now()
            while (n + 1) / rate <= now - t0:
                t = t0 + n / rate
                if t > responses[-1] + 1:
                    responses.append(responses[-1] + rng.uniform(8, 20))
                us = 5.0 + 0.8 * math.sin(t / 60.0)
                for onset in responses:
                    d = t - onset
                    if d > 0:
                        us += 1.2 * (1 - math.exp(-d / 0.8)) \
                            * math.exp(-d / 6.0)
                us += rng.normal(0, 0.005)
                resistance = 1000.0 / us
                band = next((i for i, top in enumerate((56, 220, 680))
                             if resistance < top), 3)
                ratio = RANGE_KOHM[band] / resistance
                adc = int(round((ratio + 1.0) * 0.5 / 3.0 * 4095))
                raw = (band << 14) | max(0, min(4095, adc))
                phase = (t * 70 / 60) % 1.0
                pulse = math.exp(-((phase - 0.2) / 0.08) ** 2) + \
                    0.35 * math.exp(-((phase - 0.5) / 0.1) ** 2)
                ppg = int(1500 + 400 * pulse + rng.normal(0, 4))
                self.on_sample(n / rate, raw,
                               ppg if self.config.ppg else None, now + 0.02)
                n += 1


# -- the device ----------------------------------------------------------------------

@dataclass
class Graph:
    """The Shimmer's nodes in the session pipeline."""
    source: PushSource
    tap: SweepTap
    writer: PhysioWriter


class ShimmerDevice(Device):
    TYPE = "shimmer"
    NAME = "Shimmer3 GSR+"
    MANUFACTURER = "Shimmer Research"
    MODEL = "Shimmer3 GSR+"
    DESCRIPTION = "Skin conductance (GSR/EDA) and ear-clip PPG, over " \
                  "Bluetooth"
    MODALITY = "physio"
    CONFIG = ShimmerConfig

    def __init__(self, config: ShimmerConfig, project_folder: Path,
                 simulate: bool = False):
        super().__init__(config, project_folder, simulate)
        self.graph: Optional[Graph] = None
        self.link = None
        self._clock = ClockMap()
        self._target: Optional[RecordingTarget] = None
        self.latest: Optional[tuple[float, float, int]] = None

    def _labels(self) -> list[str]:
        return LABELS if self.config.ppg else LABELS[:3]

    def _on_sample(self, sensor: float, raw_gsr: int, raw_ppg,
                   received: float) -> None:
        graph = self.graph
        if graph is None:
            return
        us, kohm, band = gsr_from_raw(int(raw_gsr))
        self.latest = (us, kohm, band)
        row = [self._clock(sensor, received), us, kohm, band]
        if self.config.ppg:
            row.append(ppg_from_raw(int(raw_ppg or 0)))
        graph.source.push(row)

    # -- the session pipeline ---------------------------------------------------------

    def build(self, pipeline: gp.Pipeline) -> None:
        labels = self._labels()
        source = PushSource(labels, UNITS[:len(labels)],
                            float(self.config.sampling_rate),
                            name=self.node_name("source"))
        tap = SweepTap(seconds=60, name=self.node_name("view"))
        writer = PhysioWriter(self.node_name("bids"))
        pipeline.connect(source, tap)
        pipeline.connect(source, writer)
        self.graph = Graph(source, tap, writer)

    def opened(self) -> None:
        self._clock = ClockMap(latency_s=self.config.latency_ms / 1000.0)
        self.latest = None
        cls = _Simulated if self.simulated else _Link
        link = cls(self.config, self._on_sample, self._fail)
        link.start()
        self.link = link

    def closing(self) -> None:
        link, self.link = self.link, None
        if link is not None:
            link.stop()

    def closed(self) -> None:
        self.graph = None
        self._target = None

    # -- recording -------------------------------------------------------------------

    @property
    def recording(self) -> bool:
        graph = self.graph
        return bool(graph and graph.writer.recording)

    def begin_recording(self, target: RecordingTarget) -> list[Path]:
        with self._lock:
            graph = self.graph
            if graph is None:
                raise RuntimeError("The Shimmer is not connected.")
            path = target.dataset.path(target.entities, target.datatype,
                                       "physio", ".tsv.gz", recording="gsr")
            self.invoke(graph.writer, "begin", {
                "path": str(path), "meta": {"epoch": graph.source.epoch}})
            self._target = target
            return [path]

    def end_recording(self) -> RecordedFiles:
        with self._lock:
            graph, target = self.graph, self._target
            if graph is None or target is None:
                return RecordedFiles()
            # The samples still on their way over Bluetooth.
            time.sleep(0.2)
            summary = self.invoke(graph.writer, "end")
            self._target = None
            path = Path(graph.writer.path)
            first, last = summary.get("first"), summary.get("last")
            samples = int(summary.get("samples") or 0)
            info = dict(self.link.info) if self.link else {}

        def finalize(reference: Optional[Timeline]) -> None:
            self._write_sidecar(target, path, first, last, info, reference)

        sidecar = path.with_name(path.name.replace(".tsv.gz", ".json"))
        return RecordedFiles(
            scans=[(path, datetime.fromtimestamp(first or target.started))],
            files=[path, sidecar], first_sample=first, finalize=finalize,
            samples=samples)

    def _write_sidecar(self, target, path, first, last, info, reference):
        cfg = self.config
        software = [f"brAIn dump {__version__}", f"g.Pype {gp.__version__}"]
        if info.get("firmware"):
            software.append(f"Shimmer firmware {info['firmware']}")
        start = start_time(reference, first)
        columns = ["timestamp", "skin_conductance", "skin_resistance",
                   "gsr_range"]
        meta = {
            "PhysioType": "generic",
            "SamplingFrequency": float(cfg.sampling_rate),
            "StartTime": round(start, 6),
            "Manufacturer": self.MANUFACTURER,
            "ManufacturersModelName": self.MODEL,
            "DeviceSerialNumber": info.get("name") or NA,
            "SoftwareVersions": ", ".join(software),
            "TaskName": target.entities.task,
            "RecordingDuration": round(last - first, 3)
            if first is not None and last is not None else 0,
            "SimulatedDevice": self.simulated,
            **target.common_metadata,
            "timestamp": {"Description": "Time of the sample, seconds since "
                                         "1970-01-01 UTC on the recording "
                                         "computer's clock (the unit's own "
                                         "clock spaces the samples)",
                          "Units": "s"},
            "skin_conductance": {"Description": "Skin conductance (GSR/EDA), "
                                                "from two finger electrodes",
                                 "Units": "uS"},
            "skin_resistance": {"Description": "Skin resistance, "
                                               "1000 / skin_conductance",
                                "Units": "kOhm"},
            "gsr_range": {"Description": "The unit's automatic measurement "
                                         "range for this sample",
                          "Levels": {str(i): f"range {i}: {span}, "
                                             f"{RANGE_KOHM[i]:g} kOhm "
                                             f"reference"
                                     for i, span in enumerate(RANGE_SPAN)}},
        }
        if cfg.ppg:
            columns.append("cardiac")
            meta["cardiac"] = {"Description": "Photoplethysmogram (PPG) from "
                                              "the ear clip, internal ADC "
                                              "A13", "Units": "mV"}
        meta["Columns"] = columns
        write_json(path.with_name(path.name.replace(".tsv.gz", ".json")),
                   meta)

    # -- status -------------------------------------------------------------------------

    @property
    def battery(self) -> Optional[float]:
        return self.link.battery if self.link else None

    def identity(self) -> str:
        if self.simulated:
            return "Shimmer3 GSR+ (simulated)"
        link = self.link
        if link is not None and link.info.get("name"):
            return link.info["name"]
        return self.config.port or ""

    def status(self) -> str:
        name = self.identity() or "Shimmer"
        if self.state == RUNNING:
            text = "recording" if self.recording else "connected"
            gsr = f", {self.latest[0]:.2f} µS" if self.latest and \
                self.latest[0] == self.latest[0] else ""
            return f"{name}: {text}{gsr}"
        return {IDLE: f"{name}: not connected",
                STARTING: f"{name}: connecting…",
                STOPPING: f"{name}: stopping…"}.get(self.state, self.state)
