"""The Unicorn's recording filter (issue #6): set in the settings, applied
to the recorded EEG, described in the sidecars; the view's filter only
changes what is shown."""
import json
import time

import numpy as np
import pyedflib
from scipy import signal

from braindump import presets
from braindump.bids import read_tsv
from braindump.devices import make_device
from braindump.devices.unicorn import (HIGHPASS, LOWPASS, NYQUIST,
                                       RecordingFilter, UnicornConfig)
from braindump.project import Project
from braindump.recording import Session, Subject


def test_defaults_and_choices():
    cfg = UnicornConfig()
    assert cfg.record_highpass == 0.1
    assert cfg.record_lowpass == NYQUIST == 125
    assert cfg.record_notch == presets.NONE
    # Every cut-off of the Unicorn Recorder's filters can be chosen.
    for spec in presets.FILTERS.values():
        if spec is None:
            continue
        btype, cutoffs = spec
        if btype in ("highpass", "bandpass"):
            assert cutoffs[0] in HIGHPASS
        if btype in ("lowpass", "bandpass"):
            assert cutoffs[-1] in LOWPASS
    assert 0 in HIGHPASS and NYQUIST in LOWPASS
    # A saved project from before keeps working, with the defaults.
    old = UnicornConfig.from_dict({"serial": "UN-1"})
    kept = old.recording_filter()
    assert (kept.low_cutoff, kept.high_cutoff, kept.notches()) == (0.1, None, [])


def test_filter_descriptions():
    default = RecordingFilter()
    assert (default.low_cutoff, default.high_cutoff) == (0.1, None)
    assert default.software_filters()["HighPass"]["CutoffFrequency"] == 0.1
    assert RecordingFilter(0, NYQUIST).sos() is None
    assert RecordingFilter(0, NYQUIST).software_filters() == "n/a"
    band = RecordingFilter(1, 30, "Cascading 50 Hz")
    assert band.software_filters()["BandPass"]["CutoffFrequencies"] == [1, 30]
    assert band.notches() == [50, 100]
    # Matches Unicorn Recorder's own band-pass, then the notch.
    expected = np.vstack([presets.filter_sos("1 – 30 Hz"),
                          presets.notch_sos("Cascading 50 Hz")])
    assert np.allclose(band.sos(), expected)


def _record(tmp_path, name, settings):
    project = Project.create(tmp_path / name, name)
    project.add_device("unicorn", {"simulate": True, **settings})
    device = make_device("unicorn", project.devices["unicorn"],
                         project.folder)
    device.start()
    try:
        time.sleep(1.5)
        session = Session(project, [device])
        session.start(Subject(sub="01", task="rest", run=1))
        time.sleep(4.0)
        assert not session.stop().problems
    finally:
        device.stop()
    eeg = project.output_path / "sub-01/eeg"
    reader = pyedflib.EdfReader(str(eeg / "sub-01_task-rest_run-1_eeg.bdf"))
    data = np.array([reader.readSignal(i) for i in range(8)])
    reader.close()
    meta = json.loads((eeg / "sub-01_task-rest_run-1_eeg.json").read_text())
    _, rows = read_tsv(eeg / "sub-01_task-rest_run-1_channels.tsv")
    return data, meta, rows


def _high_share(data):
    """The share of the EEG's power above 40 Hz."""
    f, p = signal.welch(data, fs=250, nperseg=250, axis=-1)
    p = p.mean(axis=0)
    return p[f > 40].sum() / p[f > 2].sum()


def test_recording_is_filtered_and_described(tmp_path):
    # The view shows everything; the recording is low-passed at 20 Hz.
    filtered, meta, rows = _record(tmp_path, "filtered", {
        "record_highpass": 1.0, "record_lowpass": 20.0,
        "record_notch": "50 Hz", "view_filter": presets.NONE,
        "view_notch": presets.NONE})
    unfiltered, plain, plain_rows = _record(tmp_path, "unfiltered", {
        "record_highpass": 0.0, "record_lowpass": NYQUIST,
        "view_filter": "1 – 30 Hz"})

    assert _high_share(filtered) < _high_share(unfiltered) / 10

    assert meta["SoftwareFilters"]["BandPass"]["CutoffFrequencies"] == [1, 20]
    assert meta["SoftwareFilters"]["Notch"]["Frequencies"] == [50]
    eeg = [r for r in rows if r["type"] == "EEG"]
    other = [r for r in rows if r["type"] != "EEG"]
    assert len(eeg) == 8 and other
    assert all((r["low_cutoff"], r["high_cutoff"], r["notch"])
               == ("1", "20", "50") for r in eeg)
    assert all((r["low_cutoff"], r["high_cutoff"], r["notch"])
               == ("n/a", "n/a", "n/a") for r in other)

    # No recording filter: the view's band-pass is not recorded.
    assert plain["SoftwareFilters"] == "n/a"
    assert all(r["low_cutoff"] == r["high_cutoff"] == "n/a"
               for r in plain_rows)
