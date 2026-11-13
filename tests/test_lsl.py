"""LSL streams recorded as BIDS events, merged with the EEG's."""
import time

import pytest
import pylsl

from conftest import connect_all
from braindump.bids import read_tsv
from braindump.devices import make_device
from braindump.devices.lsl import matches
from braindump.project import Project
from braindump.recording import Session, Subject


def test_selector_matching():
    info = pylsl.StreamInfo("Stim", "Markers", 1, 0, "string", "abc")
    assert matches({"name": "", "type": "Markers"}, info)
    assert matches({"name": "Stim", "source_id": "abc"}, info)
    assert not matches({"name": "Other"}, info)


@pytest.mark.slow
def test_markers_become_events(tmp_path):
    outlet = pylsl.StreamOutlet(pylsl.StreamInfo(
        "BrainDumpTestStim", "Markers", 1, 0, "string", "bd-test-stim"))
    project = Project.create(tmp_path / "p", "P")
    project.add_device("unicorn", {"simulate": True})
    project.add_device("lsl", {"streams": [{"name": "BrainDumpTestStim"}]})
    devices = [make_device(t, c, project.folder)
               for t, c in project.devices.items()]
    pipeline = connect_all(devices)
    lsl = devices[1]
    try:
        deadline = time.time() + 10
        while not lsl.inlets and time.time() < deadline:
            time.sleep(0.1)
        assert lsl.inlets, "the stream was not found"
        time.sleep(1.5)  # the inlet is open before samples flow
        outlet.push_sample(["before"])
        time.sleep(0.3)
        session = Session(project, devices)
        session.start(Subject(sub="01", task="oddball", run=1))
        time.sleep(0.5)
        sent = time.time()
        outlet.push_sample(["stimulus 1"])
        time.sleep(0.5)
        outlet.push_sample(["response"])
        time.sleep(0.5)
        result = session.stop()
    finally:
        pipeline.stop()
    assert not result.problems
    events = project.output_path / "sub-01/eeg/" \
        "sub-01_task-oddball_run-1_events.tsv"
    columns, rows = read_tsv(events)
    assert columns == ["onset", "duration", "trial_type", "value", "source",
                       "sample"]
    assert [r["value"] for r in rows] == ["stimulus 1", "response"]
    assert rows[0]["trial_type"] == "BrainDumpTestStim"
    assert rows[0]["source"] == "lsl:BrainDumpTestStim"
    # Onsets are from the EEG's first sample, which came ~0.1 s after
    # Record; the marker was sent 0.5 s after that.
    onset = float(rows[0]["onset"])
    assert 0.2 < onset < 0.8
    assert int(rows[0]["sample"]) == round(onset * 250)
    assert events.with_suffix(".json").exists()


@pytest.mark.slow
def test_a_procedure_run_again_is_recorded_once(tmp_path):
    """The 2026-11-11 validation recorded every marker twice. Its procedure
    had been started, aborted and started again: the second run's outlet
    had the first's name and source_id, LSL's recovery re-attached the
    first run's inlet to it, and the resolver opened another. Each inlet
    now keeps to its own outlet, so each marker arrives once."""
    def outlet():
        return pylsl.StreamOutlet(pylsl.StreamInfo(
            "BrainDumpTestRerun", "Markers", 1, 0, "string",
            "bd-test-rerun"))

    project = Project.create(tmp_path / "p", "P")
    project.add_device("lsl", {"streams": [{"name": "BrainDumpTestRerun"}]})
    lsl = make_device("lsl", project.devices["lsl"], project.folder)
    pipeline = connect_all([lsl])
    try:
        first = outlet()                    # the procedure's first run
        assert first.wait_for_consumers(10)
        first.push_sample(["first run"])
        time.sleep(0.5)
        del first                           # aborted: its outlet is gone
        time.sleep(1.0)
        second = outlet()                   # started again
        assert second.wait_for_consumers(10)
        # Long enough for the old inlet to have recovered (it did within
        # a second) and for the resolver to look again.
        for i in range(16):
            second.push_sample([f"marker {i}"])
            time.sleep(0.25)
        time.sleep(1.0)
        values = [e.value for e in lsl.log
                  if e.source == "lsl:BrainDumpTestRerun"]
        assert values.count("marker 0") == 1
        assert sorted(set(values)) == sorted(values), values
        assert len([v for v in values if v.startswith("marker")]) == 16
        assert len(lsl.inlets) == 1
    finally:
        pipeline.stop()
