"""The g.HIamp through g.Pype's real GHIamp source, on the simulated
amplifier."""
import json
import time

import numpy as np
import pyedflib

from braindump.bids import read_tsv
from braindump.devices import make_device
from braindump.project import Project
from braindump.recording import Session, Subject


def test_records_named_channels_and_trigger_onsets(tmp_path):
    project = Project.create(tmp_path / "p", "P")
    project.add_device("hiamp", {"simulate": True, "channel_count": 16,
                                 "sampling_rate": 512,
                                 "electrodes": ["Fp1", "Fp2", "F3"]})
    amp = make_device("hiamp", project.devices["hiamp"], project.folder)
    amp.start()
    try:
        time.sleep(1.0)
        assert len(amp.signal_columns()) == 16
        assert amp.trigger_column() == 16
        session = Session(project, [amp])
        session.start(Subject(sub="01", task="rest"))
        time.sleep(4.5)
        result = session.stop()
    finally:
        amp.stop()
    assert not result.problems
    eeg = project.output_path / "sub-01" / "eeg"
    reader = pyedflib.EdfReader(str(eeg / "sub-01_task-rest_eeg.bdf"))
    labels = reader.getSignalLabels()
    assert labels[:4] == ["Fp1", "Fp2", "F3", "Ch4"]
    assert labels[-1] == "TRIG" and len(labels) == 17
    assert reader.getSampleFrequency(0) == 512
    trig = reader.readSignal(labels.index("TRIG"))
    reader.close()
    # The pulse holds for 100 ms, but it is one event: its rising edge.
    rising = np.flatnonzero((trig != 0) & (np.r_[0, trig[:-1]] != trig))
    _, events = read_tsv(eeg / "sub-01_task-rest_events.tsv")
    assert [int(e["sample"]) for e in events] == list(rising)
    assert 1 <= len(events) <= 3
    assert all(e["source"] == "digital input" for e in events)
    meta = json.loads((eeg / "sub-01_task-rest_eeg.json").read_text())
    assert meta["SamplingFrequency"] == 512
    assert meta["EEGChannelCount"] == 16 and meta["TriggerChannelCount"] == 1
    assert meta["RecordingTiming"]["Method"] == "fit"
    _, channels = read_tsv(eeg / "sub-01_task-rest_channels.tsv")
    assert channels[0]["name"] == "Fp1" and channels[-1]["type"] == "TRIG"


def test_two_eeg_amplifiers_get_their_own_acq(tmp_path):
    project = Project.create(tmp_path / "p", "P")
    project.add_device("unicorn", {"simulate": True})
    project.add_device("hiamp", {"simulate": True, "channel_count": 8})
    devices = [make_device(t, c, project.folder)
               for t, c in project.devices.items()]
    for d in devices:
        d.start()
    try:
        time.sleep(1.0)
        session = Session(project, devices)
        session.start(Subject(sub="01", task="rest"))
        time.sleep(2.5)
        result = session.stop()
    finally:
        for d in devices:
            d.stop()
    assert not result.problems
    eeg = project.output_path / "sub-01" / "eeg"
    for acq in ("unicorn", "hiamp"):
        assert (eeg / f"sub-01_task-rest_acq-{acq}_eeg.bdf").exists()
        assert (eeg / f"sub-01_task-rest_acq-{acq}_eeg.json").exists()
