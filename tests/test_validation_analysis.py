"""validation/analyse.py on synthetic recordings, in brAIn dump's layout,
whose properties are known: it passes a good one, and catches each fault.
"""
import gzip
import importlib.util
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pyedflib
import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "analyse", ROOT / "validation" / "analyse.py")
analyse = importlib.util.module_from_spec(spec)
sys.modules["analyse"] = analyse
spec.loader.exec_module(analyse)

FS, GAZE_HZ, FPS = 250.0, 60.0, 30.0
LABELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
T0 = 1.79e9  # clock time of the EEG's first sample
STEM = "sub-01_task-facevalidation"


def procedure(rng, rest=10.0, trials=20):
    """(time, marker) as the PsychoPy procedure sends them, in EEG s."""
    t, out = 3.0, []
    out.append((t, "procedure_start"))
    t += 2
    out.append((t, "rest_open_start"))
    t += rest
    out.append((t, "rest_open_end"))
    t += 2
    out.append((t, "rest_closed_start"))
    t += rest
    out.append((t, "rest_closed_end"))
    t += 3
    out.append((t, "calibration_start"))
    t += 8
    out.append((t, "calibration_end"))
    t += 2
    crosses = []
    for k in range(1, trials + 1):
        x, y = rng.uniform(0.12, 0.88), rng.uniform(0.12, 0.88)
        out.append((t, f"fixation trial={k} x={x:.4f} y={y:.4f}"))
        crosses.append((t, x, y))
        t += 2
        out.append((t, f"face trial={k} image=f{k}.png x={x:.4f} y={y:.4f}"))
        t += 1
        out.append((t, f"blank trial={k}"))
        t += 1
    out.append((t, "debriefing"))
    t += 2
    out.append((t, "procedure_end"))
    return out, crosses, t + 2


def make_dataset(root: Path, *, alpha=True, n170=True, gaze_offset=0.0,
                 marker_shift=0.0, video_delay=0.04, video_scale=1.0,
                 seed=3):
    """Write a recording. ``marker_shift`` moves the markers (only) that
    many seconds late against everything else, as broken timing would."""
    rng = np.random.default_rng(seed)
    markers, crosses, duration = procedure(rng)
    when = {text.split()[0]: t for t, text in markers}
    n = int(duration * FS)
    t = np.arange(n) / FS
    eeg = rng.normal(0, 4, (n, len(LABELS)))
    posterior = [LABELS.index(c) for c in ("PO7", "Oz", "PO8", "Pz")]
    a_open, a_closed = 3.0, (12.0 if alpha else 3.0)
    closed = (t >= when["rest_closed_start"] + 0.5) & \
        (t < when["rest_closed_end"])
    amp = np.where(closed, a_closed, a_open)
    # Alpha as real alpha is: band-limited noise, not locked to anything.
    from scipy import signal
    sos = signal.butter(4, [8, 12], btype="bandpass", fs=FS, output="sos")
    for c in posterior:
        band = signal.sosfiltfilt(sos, rng.normal(0, 1, n))
        eeg[:, c] += amp * band / band.std()
    if n170:
        for time, text in markers:
            if text.startswith("face"):
                d = t - time
                wave = 2.5 * np.exp(-((d - 0.10) / 0.02) ** 2) \
                    - 7.0 * np.exp(-((d - 0.17) / 0.025) ** 2)
                for c in (LABELS.index("PO7"), LABELS.index("PO8")):
                    eeg[:, c] += wave
    eeg_dir = root / "sub-01" / "eeg"
    beh_dir = root / "sub-01" / "beh"
    eeg_dir.mkdir(parents=True)
    beh_dir.mkdir(parents=True)
    w = pyedflib.EdfWriter(str(eeg_dir / f"{STEM}_eeg.bdf"), len(LABELS),
                           file_type=pyedflib.FILETYPE_BDFPLUS)
    w.setSignalHeaders([{"label": lab, "dimension": "uV",
                         "sample_frequency": FS, "physical_min": -750000,
                         "physical_max": 750000, "digital_min": -8388608,
                         "digital_max": 8388607, "transducer": "",
                         "prefilter": ""} for lab in LABELS])
    whole = n - n % int(FS)
    w.writeSamples([np.ascontiguousarray(eeg[:whole, i])
                    for i in range(len(LABELS))])
    w.close()
    (eeg_dir / f"{STEM}_eeg.json").write_text(json.dumps({
        "SamplingFrequency": FS,
        "RecordingTiming": {"MeasuredSamplingFrequency": FS}}))
    (eeg_dir / f"{STEM}_channels.tsv").write_text(
        "name\ttype\tunits\n" + "".join(f"{c}\tEEG\tuV\n" for c in LABELS))
    events = "onset\tduration\ttrial_type\tvalue\tsource\n" + "".join(
        f"{time + marker_shift:.6f}\t0\tbrAIndumpValidation\t{text}\t"
        f"lsl:brAIndumpValidation\n" for time, text in markers)
    (eeg_dir / f"{STEM}_events.tsv").write_text(events)
    (beh_dir / f"{STEM}_events.tsv").write_text(events)

    # Gaze: on the centre, then on each cross 250 ms after it appears;
    # lost while the eyes are closed.
    gt = np.arange(0.5, duration - 0.5, 1 / GAZE_HZ)
    gx = np.full(len(gt), 0.5)
    gy = np.full(len(gt), 0.5)
    for time, x, y in crosses:
        on = gt >= time + 0.25
        gx[on], gy[on] = x, y
    gx += rng.normal(0, 0.004, len(gt)) + gaze_offset
    gy += rng.normal(0, 0.004, len(gt))
    valid = ~((gt >= when["rest_closed_start"] + 0.4) &
              (gt < when["rest_closed_end"] + 0.3))
    rows = np.column_stack([T0 + gt, gx, gy, valid.astype(float)])
    with gzip.open(beh_dir / f"{STEM}_recording-gaze_physio.tsv.gz",
                   "wt") as f:
        for r in rows:
            f.write("\t".join(f"{v:.6f}" for v in r) + "\n")
    (beh_dir / f"{STEM}_recording-gaze_physio.json").write_text(json.dumps({
        "StartTime": gt[0], "Columns": ["timestamp", "BPOGX", "BPOGY",
                                        "BPOGV"]}))
    (beh_dir / f"{STEM}_recording-eye1_physio.json").write_text(json.dumps({
        "ScreenSize": [0.53, 0.30], "ScreenDistance": 0.65}))

    # The screen: each face appears ``video_delay`` after its marker,
    # where its cross was; ``video_scale`` > 1 scales the screen into
    # the video and crops it, as an OBS canvas of another shape does.
    size = (160, 90)
    video = beh_dir / f"{STEM}_recording-screen_video.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, size)
    faces = [(time, x, y) for time, text in markers if text.startswith("face")
             for x, y in [(float(text.split("x=")[1].split()[0]),
                           float(text.split("y=")[1]))]]
    ft = np.arange(0.2, duration - 0.5, 1 / FPS)
    for time in ft:
        frame = np.full((size[1], size[0], 3), 90, np.uint8)
        for on, x, y in faces:
            if on + video_delay <= time < on + 1.0:
                cx = int(x * video_scale * size[0])
                cy = int(y * video_scale * size[1])
                if cx >= size[0] or cy >= size[1]:
                    continue
                frame[max(0, cy - 12):cy + 12, max(0, cx - 9):cx + 9] = 220
        writer.write(frame)
    writer.release()
    with gzip.open(beh_dir / f"{STEM}_recording-screen_physio.tsv.gz",
                   "wt") as f:
        for i, time in enumerate(ft):
            f.write(f"{T0 + time:.6f}\t{i}\n")
    (beh_dir / f"{STEM}_recording-screen_physio.json").write_text(
        json.dumps({"StartTime": ft[0], "Columns": ["timestamp", "frame"]}))
    return root


def run(tmp_path, **kwargs):
    root = make_dataset(tmp_path / "bids", **kwargs)
    checks = analyse.analyse(root, "01", out=tmp_path / "report")
    assert [c.name for c in checks if c.info] == [
        "EEG signal quality", "Screen recording setup"]
    return ({c.name.split()[0]: c for c in checks if not c.info},
            tmp_path / "report")


def test_good_recording_passes(tmp_path):
    checks, report = run(tmp_path)
    for c in checks.values():
        assert c.passed, (c.name, c.lines)
    text = (report / "report.md").read_text()
    assert text.count("**PASS**") == 4
    sync = "\n".join(checks["Synchronisation"].lines)
    assert "250 ms" in sync or "233 ms" in sync or "267 ms" in sync
    assert (report / "n170.png").exists()


def test_catches_no_alpha_effect(tmp_path):
    checks, _ = run(tmp_path, alpha=False)
    assert checks["EEG"].passed is False


def test_catches_inaccurate_gaze(tmp_path):
    checks, _ = run(tmp_path, gaze_offset=0.06)   # about 3.5 degrees
    assert checks["Gaze"].passed is False


def test_catches_missing_n170(tmp_path):
    checks, _ = run(tmp_path, n170=False)
    assert checks["N170"].passed is False


@pytest.mark.parametrize("shift", [0.3, -0.3])
def test_catches_markers_out_of_step(tmp_path, shift):
    """Markers 300 ms late (or early) against the EEG, eyes and screen."""
    checks, _ = run(tmp_path, marker_shift=shift)
    assert checks["Synchronisation"].passed is False
    assert checks["N170"].passed is False


def test_procedure_project_and_analysis_agree():
    """The procedure's marker stream is the one the project listens for
    and the analysis reads; the scripts are valid Python."""
    import ast
    import re
    procedure = (ROOT / "validation" / "face_validation.py").read_text()
    ast.parse(procedure)
    stream = re.search(r'STREAM_NAME = "(\w+)"', procedure).group(1)
    project = (ROOT / "validation" / "make_project.py").read_text()
    ast.parse(project)
    assert re.search(r'MARKERS = "(\w+)"', project).group(1) == stream
    assert analyse.MARKER_SOURCE == f"lsl:{stream}"
    task = re.search(r'TASK = "(\w+)"', project).group(1)
    assert task == "facevalidation"


def test_make_project(tmp_path):
    import subprocess
    out = subprocess.run(
        [sys.executable, str(ROOT / "validation" / "make_project.py"),
         str(tmp_path / "v"), "--unicorn", "UN-1", "--gp3-host", "10.0.0.5"],
        capture_output=True, text=True, check=True)
    project = json.loads(next((tmp_path / "v").glob("*.braindump"))
                         .read_text())
    assert project["task"] == "facevalidation"
    assert list(project["devices"]) == ["unicorn", "gp3", "lsl", "obs"]
    assert project["devices"]["gp3"]["mode"] == "remote"
    assert project["devices"]["gp3"]["host"] == "10.0.0.5"
    assert project["devices"]["lsl"]["streams"][0]["name"] == \
        "brAIndumpValidation"
    assert project["devices"]["obs"]["label"] == "screen"
    assert "Project:" in out.stdout


def test_screen_recording_scaled_and_late(tmp_path):
    """As in the first session with real hardware: OBS's canvas scaled and
    cropped the screen, and its frames came 390 ms after the screen
    showed them. The faces are still found, the steady delay is measured
    and the fix is named."""
    checks, _ = run(tmp_path, video_delay=0.39, video_scale=1.44)
    sync = checks["Synchronisation"]
    assert sync.passed is False
    text = "\n".join(sync.lines)
    found = re.search(r"faces appear in the screen recording \*\*(\d+) ms",
                      text)
    assert found and 380 <= int(found.group(1)) <= 430, text
    assert "Set the OBS input's Delay to" in text


def test_screen_recording_scaled_but_in_time(tmp_path):
    checks, _ = run(tmp_path, video_scale=1.44)
    assert checks["Synchronisation"].passed, checks["Synchronisation"].lines
