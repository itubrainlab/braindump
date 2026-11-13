"""The session pipeline: one g.Pype pipeline carrying every connected
device, recordings begun and ended through g.Pype actions."""
import time

import pytest

from braindump.devices import make_device
from braindump.devices.base import IDLE, RUNNING
from braindump.pipeline import DeviceError, SessionPipeline
from braindump.project import Project
from braindump.recording import Session, Subject


def _devices(tmp_path, kinds):
    project = Project.create(tmp_path / "p", "P")
    for kind in kinds:
        project.add_device(kind, {"simulate": True})
    devices = [make_device(t, c, project.folder)
               for t, c in project.devices.items()]
    pipeline = SessionPipeline()
    for device in devices:
        device.attach(pipeline)
    return project, devices, pipeline


def _wait(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline and not predicate():
        time.sleep(0.05)
    return predicate()


def test_one_gpype_pipeline_carries_every_device(tmp_path):
    project, devices, pipeline = _devices(
        tmp_path, ["unicorn", "gp3", "polar", "shimmer", "lsl", "webcam"])
    pipeline.connect(devices)
    try:
        assert all(d.state == RUNNING for d in devices)
        nodes = {n["name"]: n for n in pipeline.pipeline.get_nodes()}
        # Every device's source and BIDS writer in the one pipeline...
        for d in devices:
            assert any(name.startswith(f"{d.TYPE}_") for name in nodes), \
                d.TYPE
        # ...and the writers record through g.Pype's control API.
        writers = [n for name, n in nodes.items() if name.endswith(
            ("_bids", "_events"))]
        assert writers and all({"begin", "end"} <= set(n["actions"])
                               for n in writers)
        session = Session(project, devices)
        session.start(Subject(sub="01", task="rest"))
        time.sleep(2.0)
        result = session.stop()
    finally:
        pipeline.stop()
    assert not result.problems
    assert all(d.state == IDLE for d in devices)
    assert all(d.graph is None for d in devices)
    eeg = project.output_path / "sub-01" / "eeg"
    beh = project.output_path / "sub-01" / "beh"
    assert (eeg / "sub-01_task-rest_eeg.bdf").exists()
    assert (beh / "sub-01_task-rest_recording-eye1_physio.tsv.gz").exists()
    assert (eeg / "sub-01_task-rest_recording-ecg_physio.tsv.gz").exists()
    assert (beh / "sub-01_task-rest_recording-webcam_video.mp4").exists()


@pytest.mark.slow
def test_connecting_another_device_restarts_the_pipeline(tmp_path):
    project, (gp3, shimmer), pipeline = _devices(tmp_path,
                                                 ["gp3", "shimmer"])
    gp3.start()
    try:
        first = pipeline.pipeline
        assert pipeline.devices == [gp3]
        shimmer.start()
        # A running g.Pype pipeline cannot gain nodes: a new one, with
        # the GP3 still connected.
        assert pipeline.pipeline is not first
        assert pipeline.devices == [gp3, shimmer]
        assert gp3.running and shimmer.running
        assert _wait(lambda: gp3.graph.source.samples > 10)
        # Not while recording: the files would end mid-run.
        session = Session(project, [gp3, shimmer])
        session.start(Subject(sub="01", task="rest"))
        try:
            with pytest.raises(RuntimeError, match="Stop recording"):
                shimmer.stop()
            assert shimmer.running
        finally:
            session.stop()
        shimmer.stop()
        assert pipeline.devices == [gp3] and gp3.running
        assert shimmer.state == IDLE
    finally:
        pipeline.stop()
    assert gp3.state == IDLE and pipeline.pipeline is None


def test_one_device_failing_stops_every_device(tmp_path):
    _, (gp3, shimmer), pipeline = _devices(tmp_path, ["gp3", "shimmer"])
    failures = []
    pipeline.on_failure = lambda device, message: failures.append(
        (device, message))
    errors = []
    shimmer.on_error = errors.append
    gp3_errors = []
    gp3.on_error = gp3_errors.append
    pipeline.connect([gp3, shimmer])
    try:
        shimmer._fail("the strap fell off")
        assert _wait(lambda: not pipeline.running)
        assert _wait(lambda: failures)
    finally:
        pipeline.stop()
    assert gp3.state == shimmer.state == IDLE
    assert failures == [(shimmer, "the strap fell off")]
    assert errors == ["the strap fell off"] and not gp3_errors
    assert shimmer.error == "the strap fell off"


def test_a_device_that_cannot_connect_connects_nothing(tmp_path):
    _, (gp3, hiamp), pipeline = _devices(tmp_path, ["gp3", "hiamp"])
    # A filter the amplifier does not offer at this rate.
    hiamp.config.amp_filters = {"256": {"bandpass": [], "notch": []}}
    hiamp.config.amp_bandpass = 99
    with pytest.raises(DeviceError) as refused:
        pipeline.connect([gp3, hiamp])
    assert refused.value.device is hiamp
    assert str(refused.value).startswith("g.tec g.HIamp:")
    assert not pipeline.running
    assert gp3.state == hiamp.state == IDLE
