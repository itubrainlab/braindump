"""The Polar H10: its Bluetooth frames, beat timing, and a recording from
the simulated strap."""
import gzip
import json
import time

import pytest
import numpy as np

from brainstream.devices import make_device
from brainstream.devices.polar import (ACC, ECG, BeatClock, acc_start,
                                       ecg_start, parse_heart_rate, parse_pmd)
from brainstream.project import Project
from brainstream.recording import Session, Subject


def _frame(kind, frame_type, stamp_ns, values, width):
    body = b"".join(int(v).to_bytes(width, "little", signed=True)
                    for v in values)
    return bytes([kind]) + stamp_ns.to_bytes(8, "little") + \
        bytes([frame_type]) + body


def test_parse_ecg_and_acc():
    f = parse_pmd(_frame(ECG, 0, 2_000_000_000, [100, -250, 1200], 3))
    assert f.kind == ECG and f.sensor_time == 2.0
    assert f.samples[:, 0].tolist() == [100, -250, 1200]
    f = parse_pmd(_frame(ACC, 1, 5, [1, -2, 1000, 3, 4, 990], 2))
    assert f.samples.tolist() == [[1, -2, 1000], [3, 4, 990]]
    assert parse_pmd(_frame(ACC, 0x81, 5, [1, 2, 3], 2)) is None


def test_start_commands():
    # The ECG start is PolarBand2lsl's, byte for byte.
    assert ecg_start() == bytes([0x02, 0x00, 0x00, 0x01, 0x82, 0x00, 0x01,
                                 0x01, 0x0E, 0x00])
    assert acc_start(200, 8)[4:6] == bytes([0xC8, 0x00])


def test_parse_heart_rate():
    # 8-bit rate, contact supported and detected, two RR intervals.
    hr = parse_heart_rate(bytes([0x16, 72, 0x00, 0x04, 0x33, 0x03]))
    assert hr.bpm == 72 and hr.contact is True
    assert hr.rr == [1.0, 0x0333 / 1024]
    hr = parse_heart_rate(bytes([0x04, 60]))
    assert hr.contact is False and hr.rr == []
    assert parse_heart_rate(bytes([0x00, 60])).contact is None


def test_beat_clock_continues_and_reanchors():
    clock = BeatClock()
    assert clock.beats([0.8, 0.9], 100.0) == [99.1, 100.0]
    first = clock.beats([0.85], 100.9)
    assert abs(first[0] - 100.85) < 1e-9          # continued
    after_gap = clock.beats([0.8], 130.0)
    assert after_gap == [130.0]                     # anchored again


@pytest.mark.slow
def test_simulated_recording(tmp_path):
    project = Project.create(tmp_path / "p", "P")
    project.add_device("polar", {"simulate": True, "accelerometer": True,
                                 "acc_rate": 50})
    polar = make_device("polar", project.devices["polar"], project.folder)
    polar.start()
    try:
        time.sleep(2.5)
        assert polar.heart_rate and 50 < polar.heart_rate < 90
        session = Session(project, [polar])
        session.start(Subject(sub="01", task="rest"))
        time.sleep(4)
        result = session.stop()
    finally:
        polar.stop()
    assert not result.problems
    beh = project.output_path / "sub-01/beh"
    stem = "sub-01_task-rest_recording-"
    with gzip.open(beh / f"{stem}ecg_physio.tsv.gz", "rt") as f:
        ecg = np.array([[float(v) for v in line.split("\t")]
                        for line in f.read().splitlines()])
    assert 400 < len(ecg) < 650                     # ~130 Hz
    spacing = np.diff(ecg[:, 0])
    assert abs(np.median(spacing) - 1 / 130) < 1e-4
    assert ecg[:, 1].max() > 800                    # R peaks, in µV
    meta = json.loads((beh / f"{stem}ecg_physio.json").read_text())
    assert meta["Columns"] == ["timestamp", "cardiac"]
    assert meta["SamplingFrequency"] == 130
    with gzip.open(beh / f"{stem}accel_physio.tsv.gz", "rt") as f:
        acc = [line.split("\t") for line in f.read().splitlines()]
    assert 150 < len(acc) < 260 and abs(float(acc[0][3]) - 1.0) < 0.1
    with gzip.open(beh / f"{stem}ecg_physioevents.tsv.gz", "rt") as f:
        beats = [line.split("\t") for line in f.read().splitlines()]
    assert 2 <= len(beats) <= 6
    assert beats[0][2] == "beat" and 0.7 < float(beats[0][3]) < 1.2
