"""Both devices, simulated, recorded together through the real
pipelines into BIDS."""
import gzip
import json
import time

import pyedflib

from conftest import connect_all
from braindump.bids import read_tsv
from braindump.devices import make_device
from braindump.project import Project
from braindump.recording import Session, Subject


def test_session_records_both_devices(tmp_path):
    project = Project.create(tmp_path / "study", "Study")
    project.add_device("unicorn", {"simulate": True})
    project.add_device("gp3", {"simulate": True})
    devices = [make_device(t, c, project.folder)
               for t, c in project.devices.items()]
    pipeline = connect_all(devices)
    try:
        time.sleep(1.0)
        session = Session(project, devices)
        subject = Subject(sub="01", ses="1", task="rest", run=1, age="31",
                          notes="pilot")
        files = session.start(subject)
        assert any(f.suffix == ".bdf" for f in files)
        assert all(d.recording for d in devices)
        time.sleep(2.5)
        result = session.stop()
    finally:
        pipeline.stop()
    assert not result.problems
    root = project.output_path
    eeg = root / "sub-01/ses-1/eeg"
    bdf = eeg / "sub-01_ses-1_task-rest_run-1_eeg.bdf"
    reader = pyedflib.EdfReader(str(bdf))
    assert reader.getSignalLabels()[:2] == ["Fz", "C3"]
    reader.close()
    meta = json.loads((eeg / "sub-01_ses-1_task-rest_run-1_eeg.json")
                      .read_text())
    assert meta["SamplingFrequency"] == 250
    assert 2.0 < meta["RecordingDuration"] < 3.5
    beh = root / "sub-01/ses-1/beh"
    eye = json.loads((beh / "sub-01_ses-1_task-rest_run-1_recording-eye1"
                            "_physio.json").read_text())
    assert eye["PhysioType"] == "eyetrack" and eye["RecordedEye"] == "left"
    # Started together: the eye tracker's start within 0.2 s of the EEG's.
    assert abs(eye["StartTime"]) < 0.2
    # The events beside the eye tracking describe the screen.
    events = json.loads((beh / "sub-01_ses-1_task-rest_run-1_events.json")
                        .read_text())
    assert events["StimulusPresentation"]["ScreenOrigin"] == ["top", "left"]
    assert (eeg / "sub-01_ses-1_task-rest_run-1_events.tsv").exists()
    with gzip.open(beh / "sub-01_ses-1_task-rest_run-1_recording-eye2"
                         "_physio.tsv.gz", "rt") as f:
        rows = [line.split("\t") for line in f.read().splitlines()]
    assert 120 < len(rows) < 200 and len(rows[0]) == len(eye["Columns"])
    _, scans = read_tsv(root / "sub-01/ses-1/sub-01_ses-1_scans.tsv")
    names = sorted(r["filename"] for r in scans)
    assert names == sorted([
        "eeg/sub-01_ses-1_task-rest_run-1_eeg.bdf",
        "beh/sub-01_ses-1_task-rest_run-1_recording-eye1_physio.tsv.gz",
        "beh/sub-01_ses-1_task-rest_run-1_recording-eye2_physio.tsv.gz",
        "beh/sub-01_ses-1_task-rest_run-1_recording-gaze_physio.tsv.gz"])
    _, people = read_tsv(root / "participants.tsv")
    assert people[0]["age"] == "31"
    assert (root / "sub-01/sub-01_sessions.tsv").exists()
    assert (root / "sourcedata/notes").is_dir()
    assert session.dataset.next_run("01", "1", "rest") == 2


def test_eye_tracking_alone_goes_to_beh(tmp_path):
    project = Project.create(tmp_path / "s", "S")
    gp3 = make_device("gp3", {"simulate": True}, project.folder)
    gp3.start()
    try:
        session = Session(project, [gp3])
        session.start(Subject(sub="02", task="free"))
        time.sleep(0.5)
        result = session.stop()
    finally:
        gp3.stop()
    assert (project.output_path / "sub-02/beh").is_dir()
    assert result.devices["Gazepoint GP3"] > 10
