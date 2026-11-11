"""The g.HIamp's amplifier filters (issue #5): the band-pass and notch the
amplifier offers, loaded from it, applied to the recorded EEG and described
in the sidecars, while the view's filter only changes what is shown."""
import json
import time

import numpy as np
import pyedflib
import pytest
from scipy import signal

from braindump.bids import read_tsv
from braindump.devices import make_device
from braindump.devices.hiamp import (NO_FILTER, HIampConfig,
                                     default_bandpass)
from braindump.project import Project
from braindump.recording import Session, Subject

RATE = 512


def _device(tmp_path, name, settings=None):
    project = Project.create(tmp_path / name, name)
    project.add_device("hiamp", {"simulate": True, "channel_count": 8,
                                 "sampling_rate": RATE, **(settings or {})})
    return project, make_device("hiamp", project.devices["hiamp"],
                                project.folder)


def test_no_recording_filter_of_its_own():
    cfg = HIampConfig()
    assert not hasattr(cfg, "record_highpass")
    assert cfg.amp_bandpass == cfg.amp_notch == NO_FILTER
    assert cfg.known_filters("bandpass") == []
    assert cfg.chosen_filter("bandpass") is None


def test_default_bandpass():
    def f(i, low, high):
        return {"index": i, "low": low, "high": high, "order": 4, "type": ""}
    # A 0.1 Hz high-pass with no low-pass first...
    assert default_bandpass([f(0, 0.1, 30), f(1, 0.1, 0), f(2, 1, 0)],
                            256)["index"] == 1
    # ...or one whose low-pass is at half the sampling rate...
    assert default_bandpass([f(0, 0.1, 30), f(1, 0.1, 128)],
                            256)["index"] == 1
    # ...else the widest band-pass from 0.1 Hz.
    assert default_bandpass([f(0, 0.1, 30), f(1, 0.1, 100), f(2, 1, 0)],
                            256)["index"] == 1
    assert default_bandpass([f(0, 1, 0)], 256) is None


def test_load_filters_without_connecting(tmp_path):
    _, amp = _device(tmp_path, "load")
    found = amp.load_filters()
    assert not amp.running
    offered = amp.config.known_filters("bandpass")
    assert offered == found["bandpass"] and amp.config.known_filters("notch")
    # Only those for this sampling rate, each with its cut-offs.
    assert all(f["high"] < RATE / 2 for f in offered)
    assert any(f["low"] == 1 and f["high"] == 100 for f in offered)
    # The first load picks the default: 0.1 Hz, no low-pass.
    chosen = amp.config.chosen_filter("bandpass")
    assert (chosen["low"], chosen["high"]) == (0.1, 0)
    # The amplifier was released: it connects, and loads while connected.
    amp.start()
    try:
        assert amp.running
        amp.config.amp_bandpass = NO_FILTER
        assert amp.load_filters()["bandpass"] == offered
        # Not the first load at this rate: "None" stays chosen.
        assert amp.config.amp_bandpass == NO_FILTER
    finally:
        amp.stop()


def test_first_connect_uses_the_default(tmp_path):
    project, amp = _device(tmp_path, "first")
    assert amp.config.known_filters("bandpass") == []
    _, meta, _ = _record(project, amp)
    # Listed as it connected, and recorded through the default at once.
    assert amp.config.known_filters("bandpass")
    used = meta["HardwareFilters"]["Amplifier band-pass"]
    assert (used["LowerCutoffFrequency"], used["UpperCutoffFrequency"]) \
        == (0.1, "n/a")


def _record(project, amp):
    amp.start()
    try:
        time.sleep(1.5)
        session = Session(project, [amp])
        session.start(Subject(sub="01", task="rest", run=1))
        time.sleep(4.0)
        assert not session.stop().problems
    finally:
        amp.stop()
    eeg = project.output_path / "sub-01/eeg"
    reader = pyedflib.EdfReader(str(eeg / "sub-01_task-rest_run-1_eeg.bdf"))
    data = np.array([reader.readSignal(i) for i in range(8)])
    reader.close()
    meta = json.loads((eeg / "sub-01_task-rest_run-1_eeg.json").read_text())
    _, rows = read_tsv(eeg / "sub-01_task-rest_run-1_channels.tsv")
    return data, meta, rows


def _share(data, above):
    f, p = signal.welch(data, fs=RATE, nperseg=RATE, axis=-1)
    p = p.mean(axis=0)
    return p[f > above].sum() / p[f > 2].sum()


def test_recordings_are_filtered_and_described(tmp_path):
    _, probe = _device(tmp_path, "probe")
    probe.load_filters()
    known = probe.config.amp_filters
    band = next(f for f in probe.config.known_filters("bandpass")
                if (f["low"], f["high"]) == (1, 100))
    notch = next(f for f in probe.config.known_filters("notch")
                 if f["low"] == 48)

    # The amplifier's 1-100 Hz band-pass and 50 Hz notch, and a view
    # filter that must not be recorded.
    project, amp = _device(tmp_path, "amplifier", {
        "amp_filters": known, "amp_bandpass": band["index"],
        "amp_notch": notch["index"], "view_filter": "1 – 30 Hz"})
    filtered, meta, rows = _record(project, amp)
    # No filter (chosen as such: the rate's filters are already known).
    project, amp = _device(tmp_path, "none", {
        "amp_filters": known, "view_filter": "1 – 30 Hz"})
    plain, plain_meta, plain_rows = _record(project, amp)

    assert _share(filtered, 150) < _share(plain, 150) / 10

    used = meta["HardwareFilters"]["Amplifier band-pass"]
    assert used["FilterIndex"] == band["index"]
    assert (used["LowerCutoffFrequency"], used["UpperCutoffFrequency"]) \
        == (1, 100)
    assert meta["HardwareFilters"]["Amplifier notch"]["FilterIndex"] == \
        notch["index"]
    assert meta["SoftwareFilters"] == "n/a"
    eeg = [r for r in rows if r["type"] == "EEG"]
    assert all((r["low_cutoff"], r["high_cutoff"], r["notch"])
               == ("1", "100", "50") for r in eeg)
    assert all(r["low_cutoff"] == "n/a" for r in rows if r["type"] != "EEG")

    assert plain_meta["SoftwareFilters"] == plain_meta["HardwareFilters"] \
        == "n/a"
    assert all(r["low_cutoff"] == r["high_cutoff"] == r["notch"] == "n/a"
               for r in plain_rows)


def test_a_filter_not_offered_at_the_rate_stops_the_connection(tmp_path):
    _, probe = _device(tmp_path, "probe")
    probe.load_filters()
    index = probe.config.known_filters("bandpass")[0]["index"]
    _, amp = _device(tmp_path, "rate", {
        "sampling_rate": 1200, "amp_filters": probe.config.amp_filters,
        "amp_bandpass": index})
    with pytest.raises(ValueError, match="not one it offers at 1200 Hz"):
        amp.start()
    assert not amp.running
