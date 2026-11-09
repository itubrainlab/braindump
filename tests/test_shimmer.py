"""The Shimmer3 GSR+: conversions, its clock, the pyshimmer link (against
a fake unit), and a recording from the simulated one."""
import gzip
import json
import time

import numpy as np
import pytest

from braindump.devices import make_device
from braindump.devices import shimmer as sh
from braindump.project import Project
from braindump.recording import Session, Subject


def _raw(band, kohm):
    ratio = sh.RANGE_KOHM[band] / kohm
    return (band << 14) | int(round((ratio + 1) * 0.5 / 3.0 * 4095))


def test_gsr_conversion():
    for band, kohm in ((0, 30), (1, 150), (2, 400), (3, 2000)):
        us, res, rng = sh.gsr_from_raw(_raw(band, kohm))
        assert rng == band
        assert abs(res - kohm) / kohm < 0.02
        assert abs(us - 1000 / kohm) / (1000 / kohm) < 0.02
    assert np.isnan(sh.gsr_from_raw(0)[0])
    assert abs(sh.ppg_from_raw(4095) - 3000) < 1e-9


def test_tick_clock_unwraps():
    clock = sh.TickClock()
    assert clock(32768) == 1.0
    assert clock((1 << 24) - 1) < clock(10)  # wrapped: later, not earlier
    assert abs(clock(32768 + 10) - (512 + 1 + 10 / 32768)) < 1e-6


def test_port_choice(monkeypatch):
    class Port:
        def __init__(self, device, description, hwid=""):
            self.device, self.description = device, description
            self.name, self.hwid = device, hwid
    ports = [Port("COM3", "Standard Serial over Bluetooth link (COM3)",
                  "BTHENUM\\{00001101}"),
             Port("COM1", "Communications Port (COM1)")]
    import serial.tools.list_ports as lp
    monkeypatch.setattr(lp, "comports", lambda: ports)
    monkeypatch.setattr(sh.sys, "platform", "win32")
    assert sh.default_port() == "COM3"           # the only Bluetooth one
    ports.append(Port("COM4", "Standard Serial over Bluetooth link (COM4)"))
    assert sh.default_port() == ""               # two: the user chooses
    ports.append(Port("/dev/rfcomm0", "n/a"))
    assert sh.find_ports()[0][0] in ("/dev/rfcomm0", "COM3", "COM4")
    monkeypatch.setattr(sh.sys, "platform", "darwin")
    ports[:] = [Port("/dev/tty.Shimmer3-1A2B", "n/a"),
                Port("/dev/cu.Shimmer3-1A2B", "n/a")]
    assert sh.default_port() == "/dev/cu.Shimmer3-1A2B"


def test_link_with_a_fake_unit(monkeypatch):
    """The calls brAIn dump makes on pyshimmer, and what it does with the
    packets."""
    pyshimmer = pytest.importorskip("pyshimmer")
    from pyshimmer import EChannelType
    from pyshimmer.dev.channels import ESensorGroup
    calls = []

    class FakeShimmer:
        def __init__(self, serial):
            calls.append(("open", serial))
            self.callback = None

        def initialize(self):
            calls.append("initialize")

        def set_sensors(self, sensors):
            calls.append(("sensors", tuple(sensors)))

        def set_sampling_rate(self, rate):
            calls.append(("rate", rate))

        def get_firmware_version(self):
            raise RuntimeError("old firmware")

        def get_device_name(self):
            return "Shimmer3-1A2B"

        def get_battery_state(self, in_percent):
            return 81.0

        def add_stream_callback(self, cb):
            self.callback = cb

        def start_streaming(self):
            calls.append("start")
            for k in range(3):
                self.callback({EChannelType.TIMESTAMP: 32768 * (k + 1),
                               EChannelType.GSR_RAW: _raw(1, 150),
                               EChannelType.INTERNAL_ADC_13: 2048})

        def stop_streaming(self):
            calls.append("stop")

        def shutdown(self):
            calls.append("shutdown")

    monkeypatch.setattr(pyshimmer, "ShimmerBluetooth", FakeShimmer)
    import serial
    monkeypatch.setattr(serial, "Serial", lambda port, *a, **k: port)
    samples = []
    link = sh._Link(sh.ShimmerConfig(port="COM7", sampling_rate=64.0),
                    lambda *a: samples.append(a), lambda m: None)
    link.start()
    link.stop()
    assert calls[0] == ("open", "COM7")
    assert ("sensors", (ESensorGroup.GSR, ESensorGroup.CH_A13)) in calls
    assert ("rate", 64.0) in calls
    assert calls[-2:] == ["stop", "shutdown"]
    assert [s[0] for s in samples] == [1.0, 2.0, 3.0]
    assert samples[0][2] == 2048
    assert link.battery == 81.0 and link.info["name"] == "Shimmer3-1A2B"


def test_simulated_recording(tmp_path):
    project = Project.create(tmp_path / "p", "P")
    project.add_device("shimmer", {"simulate": True, "sampling_rate": 64.0})
    unit = make_device("shimmer", project.devices["shimmer"], project.folder)
    unit.start()
    try:
        time.sleep(1.0)
        assert 3 < unit.latest[0] < 9
        session = Session(project, [unit])
        session.start(Subject(sub="01", task="rest"))
        time.sleep(3)
        result = session.stop()
    finally:
        unit.stop()
    assert not result.problems
    beh = project.output_path / "sub-01/beh"
    with gzip.open(beh / "sub-01_task-rest_recording-gsr_physio.tsv.gz",
                   "rt") as f:
        rows = np.array([[float(v) for v in line.split("\t")]
                         for line in f.read().splitlines()])
    assert 150 < len(rows) < 230                     # ~64 Hz
    assert abs(np.median(np.diff(rows[:, 0])) - 1 / 64) < 1e-3
    assert np.allclose(rows[:, 1] * rows[:, 2], 1000, rtol=1e-3)
    meta = json.loads((beh / "sub-01_task-rest_recording-gsr_physio.json")
                      .read_text())
    assert meta["Columns"] == ["timestamp", "skin_conductance",
                               "skin_resistance", "gsr_range", "cardiac"]
    assert meta["SamplingFrequency"] == 64.0
