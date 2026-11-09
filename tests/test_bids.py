from datetime import datetime

import pytest

from braindump.bids import Dataset, Entities, clean_label, read_tsv


def test_entities():
    e = Entities("01", "rest", "pre", 2)
    assert e.stem() == "sub-01_ses-pre_task-rest_run-2"
    assert e.stem("eye1") == "sub-01_ses-pre_task-rest_run-2_recording-eye1"
    assert Entities("01", "rest").stem() == "sub-01_task-rest"
    with pytest.raises(ValueError):
        Entities("0-1", "rest")
    assert clean_label("p_01 x") == "p01x"


def test_dataset_files(tmp_path):
    ds = Dataset(tmp_path, "Study", ["A. Author"])
    ds.ensure()
    assert (tmp_path / "dataset_description.json").exists()
    ds.add_participant("01", {"age": 30})
    ds.add_participant("02", {})
    ds.add_participant("01", {"sex": "F"})
    columns, rows = read_tsv(tmp_path / "participants.tsv")
    assert columns[:4] == ["participant_id", "age", "sex", "handedness"]
    assert [r["participant_id"] for r in rows] == ["sub-01", "sub-02"]
    assert rows[0]["age"] == "30" and rows[0]["sex"] == "F"
    assert rows[1]["age"] == "n/a"

    e = Entities("01", "rest", "1", 1)
    f = ds.path(e, "eeg", "eeg", ".bdf")
    f.parent.mkdir(parents=True)
    f.write_bytes(b"")
    moment = datetime(2026, 1, 2, 3, 4, 5, 678901)
    ds.add_scans(e, [(f, moment)])
    _, scans = read_tsv(ds.session_dir(e) / "sub-01_ses-1_scans.tsv")
    assert scans == [{"filename": "eeg/sub-01_ses-1_task-rest_run-1_eeg.bdf",
                      "acq_time": "2026-01-02T03:04:05.678901"}]
    assert ds.next_run("01", "1", "rest") == 2
    assert ds.next_run("01", "1", "other") == 1
    assert ds.next_run("02", "1", "rest") == 1
