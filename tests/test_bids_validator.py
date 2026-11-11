"""A dataset recorded from every device, simulated, passes the official
BIDS validator with no errors.

Needs the validator: ``pip install bids-validator-deno`` (or set
``BIDS_VALIDATOR`` to its executable); skipped otherwise.
"""
import json
import os
import shutil
import subprocess
import time

import pytest

from braindump.devices import make_device
from braindump.devices.hiamp import _simulated_filters, amplifier_filter
from braindump.project import Project
from braindump.recording import Session, Subject

VALIDATOR = (os.environ.get("BIDS_VALIDATOR")
             or shutil.which("bids-validator-deno")
             or shutil.which("bids-validator"))


@pytest.mark.skipif(not VALIDATOR, reason="no BIDS validator installed")
def test_recorded_dataset_is_valid(tmp_path):
    project = Project.create(tmp_path / "p", "Validation",
                             authors=["A. Author", "B. Author"])
    for kind in ("unicorn", "hiamp", "gp3", "lsl", "polar", "shimmer", "webcam", "obs"):
        project.add_device(kind, {"simulate": True, "accelerometer": True})
    # A notch at several frequencies, written as a list in channels.tsv.
    project.devices["unicorn"]["record_notch"] = "Cascading 50 Hz"
    # The g.HIamp's own band-pass and notch, as if listed when connected.
    known = {kind: [amplifier_filter(e, kind)
                    for e in _simulated_filters(kind, 256)]
             for kind in ("bandpass", "notch")}
    project.devices["hiamp"].update(
        amp_filters={"256": known}, amp_bandpass=known["bandpass"][-1]["index"],
        amp_notch=known["notch"][0]["index"])
    devices = [make_device(t, c, project.folder)
               for t, c in project.devices.items()]
    for d in devices:
        d.start()
    try:
        time.sleep(2.5)  # the LSL test stream is found and sends
        gp3 = next(d for d in devices if d.TYPE == "gp3")
        run = gp3.backend.calibrate(5)
        run.delay = run.timeout = 0.02
        while not run.tick():
            time.sleep(0.01)
        gp3.calibration_done(run.result())
        session = Session(project, devices)
        session.start(Subject(sub="01", ses="1", task="rest", run=1,
                              age="30", sex="F", handedness="R",
                              task_description="Rest", instructions="Relax",
                              head_circumference="56"))
        time.sleep(2.5)
        assert not session.stop().problems
        for d in devices:  # without EEG: beh/
            if d.MODALITY == "eeg":
                d.stop()
        session = Session(project, [d for d in devices if d.running])
        session.start(Subject(sub="02", task="free", run=1))
        time.sleep(1.5)
        assert not session.stop().problems
    finally:
        for d in devices:
            d.stop()
    out = subprocess.run([VALIDATOR, str(project.output_path), "--json"],
                         capture_output=True, text=True, timeout=300)
    report = json.loads(out.stdout)
    errors = [(i["code"], i.get("subCode"), i.get("location"))
              for i in report["issues"]["issues"]
              if i.get("severity") == "error"]
    assert errors == []
