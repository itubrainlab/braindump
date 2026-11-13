"""Synchronisation: every device's samples on one clock, and the EEG's
own time measured, so events and recordings line up.

The end-to-end test sends a UDP trigger and an LSL marker at the same
instant: g.Pype places the trigger on the EEG sample acquired then, by
the headset's acquisition stamps; brAIn stream places the marker by its own
clock and the EEG timeline it fitted. They must land on the same sample.
"""
import json
import socket
import time

import numpy as np
import pylsl
import pyedflib

from conftest import connect_all
from brainstream import clock
from brainstream.bids import read_tsv
from brainstream.devices import make_device
from brainstream.devices.base import Timeline, start_time
from brainstream.pipeline import ClockMap
from brainstream.devices.unicorn import fit_timing
from brainstream.project import Project
from brainstream.recording import Session, Subject


def test_clock_is_steady_and_readable():
    a, b = clock.now(), clock.now()
    assert b >= a
    assert abs(clock.now() - time.time()) < 1.0
    assert abs(clock.from_wall(time.time()) - clock.now()) < 0.01
    assert abs(clock.from_perf(time.perf_counter()) - clock.now()) < 0.01


def test_timeline():
    tl = Timeline(t0=100.0, rate=250.025, nominal_rate=250.0)  # 100 ppm
    assert tl.sample(100.0) == 0
    assert tl.sample(100.0 + 3600) == round(3600 * 250.025)
    # One hour in: EEG time runs 100 ppm ahead of the clock.
    assert abs(tl.seconds(100.0 + 3600) - 3600.36) < 1e-9
    assert abs(tl.seconds(tl.clock(9000)) - 9000 / 250) < 1e-9
    assert start_time(tl, 101.0) == round(250.025 / 250, 6)
    assert start_time(None, 101.0) == 0.0


def test_fit_recovers_start_and_rate():
    """Acquisition stamps of a headset whose clock runs 60 ppm fast, with
    a few ms of jitter and the odd late burst."""
    rng = np.random.default_rng(1)
    rate, t0, n = 250.0 * (1 + 60e-6), 1000.0, 250 * 600  # ten minutes
    true = t0 + np.arange(n) / rate
    stamps = true + rng.normal(0, 0.002, n)
    stamps[rng.integers(0, n, 50)] += 0.08
    fit = fit_timing(stamps, 250.0)
    assert fit.method == "fit"
    assert abs(fit.timeline.t0 - t0) < 0.0005
    assert abs(fit.timeline.rate - rate) / rate < 2e-6
    # By the end, the nominal rate would be 36 ms (9 samples) off.
    end = true[-1]
    assert abs(fit.timeline.sample(end) - (n - 1)) <= 1
    assert abs((end - t0) * 250.0 - (n - 1)) > 8


def test_fit_keeps_nominal_rate_when_short():
    stamps = 50.0 + np.arange(100) / 250.0
    fit = fit_timing(stamps, 250.0)
    assert fit.method == "nominal" and fit.timeline.rate == 250.0
    assert abs(fit.timeline.t0 - 50.0) < 1e-9


def test_clock_map_follows_drift_through_jitter():
    """A strap whose crystal runs 40 ppm slow, read over a link with 8 ms
    minimum delay and tens of ms of jitter: the map follows the drift and
    ignores the jitter. The minimum delay itself cannot be seen in the
    data; told it, the map is within 2 ms of the truth throughout."""
    rng = np.random.default_rng(2)
    cmap = ClockMap(window_s=30.0, latency_s=0.008)
    errors = []
    for i in range(20 * 60 * 2):  # 20 minutes, two frames a second
        true = 5000.0 + i * 0.5
        sensor = 7e8 + (true - 5000.0) * (1 - 40e-6)
        received = true + 0.008 + rng.exponential(0.015)
        mapped = cmap(sensor, received)
        if i > 60:
            errors.append(mapped - true)
    errors = np.abs(np.array(errors))
    assert errors.max() < 0.002, errors.max()


def _free_udp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_trigger_and_marker_land_on_the_same_eeg_sample(tmp_path):
    port = _free_udp_port()
    outlet = pylsl.StreamOutlet(pylsl.StreamInfo(
        "BrainStreamSyncTest", "Markers", 1, 0, "string", "bd-sync-test"))
    project = Project.create(tmp_path / "p", "P")
    project.add_device("unicorn", {"simulate": True, "trigger_port": port})
    project.add_device("lsl", {"streams": [{"name": "BrainStreamSyncTest"}]})
    project.add_device("gp3", {"simulate": True})
    devices = [make_device(t, c, project.folder)
               for t, c in project.devices.items()]
    pipeline = connect_all(devices)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        lsl = devices[1]
        deadline = time.time() + 10
        while not lsl.inlets and time.time() < deadline:
            time.sleep(0.1)
        assert lsl.inlets, "the LSL stream was not found"
        time.sleep(1.5)
        session = Session(project, devices)
        session.start(Subject(sub="01", task="sync"))
        time.sleep(1.0)
        for value in range(1, 7):
            # The same instant, through both paths.
            sender.sendto(str(value).encode(), ("127.0.0.1", port))
            outlet.push_sample([f"m{value}"])
            time.sleep(0.45)
        time.sleep(0.5)
        result = session.stop()
    finally:
        sender.close()
        pipeline.stop()
    assert not result.problems
    eeg = project.output_path / "sub-01" / "eeg"
    _, rows = read_tsv(eeg / "sub-01_task-sync_events.tsv")
    triggers = [r for r in rows if r["trial_type"] == "trigger"]
    markers = [r for r in rows if r["source"] == "lsl:BrainStreamSyncTest"]
    assert [r["value"] for r in triggers] == [str(v) for v in range(1, 7)]
    assert [r["value"] for r in markers] == [f"m{v}" for v in range(1, 7)]
    gaps = [int(m["sample"]) - int(t["sample"])
            for t, m in zip(triggers, markers)]
    # Same sample, give or take the jitter of the simulated radio and of
    # the scheduler (a sample is 4 ms).
    assert max(abs(g) for g in gaps) <= 3, gaps
    for t in triggers:
        assert abs(float(t["onset"]) - int(t["sample"]) / 250) < 1e-6

    # The trigger is also in the BDF, on the sample events.tsv names.
    reader = pyedflib.EdfReader(str(eeg / "sub-01_task-sync_eeg.bdf"))
    trig = reader.readSignal(reader.getSignalLabels().index("TRIG"))
    reader.close()
    onsets = np.flatnonzero(trig)
    assert list(onsets) == [int(t["sample"]) for t in triggers]

    meta = json.loads((eeg / "sub-01_task-sync_eeg.json").read_text())
    timing = meta["RecordingTiming"]
    assert timing["Method"] == "fit"
    assert abs(timing["MeasuredSamplingFrequency"] - 250) < 0.5

    # The eye tracker's StartTime agrees with its first time stamp.
    beh = project.output_path / "sub-01" / "beh"
    eye = json.loads((beh / "sub-01_task-sync_recording-eye1_physio.json")
                     .read_text())
    import gzip
    with gzip.open(beh / "sub-01_task-sync_recording-eye1_physio.tsv.gz",
                   "rt") as f:
        first_eye = float(f.readline().split("\t")[0])
    eeg_start = result.files and datetime_from(timing["FirstSampleTime"])
    assert abs(eye["StartTime"] - (first_eye - eeg_start)) < 0.002


def datetime_from(text: str) -> float:
    from datetime import datetime
    return datetime.fromisoformat(text).timestamp()
