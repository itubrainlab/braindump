"""The g.HIamp's recording filters (issue #5): the amplifier's own
band-pass and notch, and a recording filter like the Unicorn's; applied to
the recorded EEG and described in the sidecars, while the view's filter
only changes what is shown."""
import json
import time

import numpy as np
import pyedflib
import pytest
from rhinorecorder import presets
from scipy import signal

from braindump.bids import read_tsv
from braindump.devices import make_device
from braindump.devices.eeg import lowpass_choices
from braindump.devices.hiamp import NO_FILTER, HIampConfig
from braindump.project import Project
from braindump.recording import Session, Subject

RATE = 512


def test_defaults_and_choices():
    cfg = HIampConfig()
    recorded = cfg.recording_filter()
    # 0.1 Hz high-pass, and half the sampling rate (no low-pass).
    assert (recorded.low_cutoff, recorded.high_cutoff) == (0.1, None)
    assert recorded.notches() == []
    assert cfg.amp_bandpass == cfg.amp_notch == NO_FILTER
    assert cfg.chosen_filter("bandpass") is None
    # The low-pass choices stay below half the sampling rate.
    assert 100 in lowpass_choices(256) and 200 not in lowpass_choices(256)
    assert 2000 in lowpass_choices(4800)
    assert all(f in lowpass_choices(256) for f in (0, 20, 30, 50))
    # Projects saved before this keep working, with the defaults.
    old = HIampConfig.from_dict({"serial": "HA-1", "sampling_rate": 1200})
    assert old.recording_filter().low_cutoff == 0.1
    assert old.recording_filter().fs == 1200


def _device(tmp_path, name, settings):
    project = Project.create(tmp_path / name, name)
    project.add_device("hiamp", {"simulate": True, "channel_count": 8,
                                 "sampling_rate": RATE, **settings})
    return project, make_device("hiamp", project.devices["hiamp"],
                                project.folder)


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


def test_connecting_lists_the_amplifiers_filters(tmp_path):
    _, amp = _device(tmp_path, "list", {})
    assert amp.config.known_filters("bandpass") == []
    amp.start()
    amp.stop()
    offered = amp.config.known_filters("bandpass")
    assert offered and amp.config.known_filters("notch")
    # Only those for this sampling rate, each with its cut-offs.
    assert all(f["high"] < RATE / 2 for f in offered)
    assert any(f["low"] == 1 and f["high"] == 100 for f in offered)


def test_recordings_are_filtered_and_described(tmp_path):
    _, probe = _device(tmp_path, "probe", {})
    probe.start()
    probe.stop()
    known = probe.config.amp_filters
    band = next(f for f in probe.config.known_filters("bandpass")
                if (f["low"], f["high"]) == (1, 100))
    notch = next(f for f in probe.config.known_filters("notch")
                 if f["low"] == 48)

    # The amplifier's 1-100 Hz band-pass and 50 Hz notch, no recording
    # filter, and a view filter that must not be recorded.
    project, amp = _device(tmp_path, "amplifier", {
        "amp_filters": known, "amp_bandpass": band["index"],
        "amp_notch": notch["index"], "record_highpass": 0.0,
        "view_filter": "1 – 30 Hz"})
    hardware, meta, rows = _record(project, amp)
    # A recording filter only: 30 Hz low-pass, 50 Hz notch.
    project, amp = _device(tmp_path, "software", {
        "record_highpass": 0.5, "record_lowpass": 30.0,
        "record_notch": "50 Hz", "view_filter": presets.NONE})
    software, soft_meta, soft_rows = _record(project, amp)
    # Nothing at all.
    project, amp = _device(tmp_path, "none", {
        "record_highpass": 0.0, "view_filter": "1 – 30 Hz"})
    plain, plain_meta, plain_rows = _record(project, amp)

    assert _share(hardware, 150) < _share(plain, 150) / 10
    assert _share(software, 60) < _share(plain, 60) / 10

    amp_band = meta["HardwareFilters"]["Amplifier band-pass"]
    assert amp_band["FilterIndex"] == band["index"]
    assert (amp_band["LowerCutoffFrequency"],
            amp_band["UpperCutoffFrequency"]) == (1, 100)
    assert meta["HardwareFilters"]["Amplifier notch"]["FilterIndex"] == \
        notch["index"]
    assert meta["SoftwareFilters"] == "n/a"
    eeg = [r for r in rows if r["type"] == "EEG"]
    assert all((r["low_cutoff"], r["high_cutoff"], r["notch"])
               == ("1", "100", "50") for r in eeg)
    assert all(r["low_cutoff"] == "n/a" for r in rows if r["type"] != "EEG")

    assert soft_meta["HardwareFilters"] == "n/a"
    assert soft_meta["SoftwareFilters"]["BandPass"]["CutoffFrequencies"] \
        == [0.5, 30]
    assert all((r["low_cutoff"], r["high_cutoff"], r["notch"])
               == ("0.5", "30", "50") for r in soft_rows
               if r["type"] == "EEG")

    assert plain_meta["SoftwareFilters"] == plain_meta["HardwareFilters"] \
        == "n/a"
    assert all(r["low_cutoff"] == r["high_cutoff"] == "n/a"
               for r in plain_rows)


def test_settings_that_cannot_apply_stop_the_connection(tmp_path):
    _, amp = _device(tmp_path, "low", {"sampling_rate": 256,
                                       "record_lowpass": 200.0})
    with pytest.raises(ValueError, match="half the sampling rate"):
        amp.start()
    assert not amp.running
    # A filter the amplifier listed at another rate.
    _, probe = _device(tmp_path, "probe", {})
    probe.start()
    probe.stop()
    index = probe.config.known_filters("bandpass")[0]["index"]
    _, amp = _device(tmp_path, "rate", {
        "sampling_rate": 1200, "amp_filters": probe.config.amp_filters,
        "amp_bandpass": index})
    with pytest.raises(ValueError, match="not one it offers at 1200 Hz"):
        amp.start()
    assert not amp.running
