"""validation/analyse.py on synthetic recordings, in brAIn stream's layout,
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


PURSUIT = dict(duration=8.0, ax=0.3, ay=0.25, fx=0.2, fy=0.3)


def pursuit_at(t, phase):
    return (0.5 + PURSUIT["ax"] * np.sin(2 * np.pi * PURSUIT["fx"] * t),
            0.5 + PURSUIT["ay"] * np.sin(2 * np.pi * PURSUIT["fy"] * t
                                         + phase))


def procedure(rng, rest=10.0, trials=20, lab=False):
    """(time, marker) as the PsychoPy procedure sends them, in EEG s; with
    ``lab``, the flashes and moving targets too."""
    t, out = 3.0, []
    out.append((t, "procedure_start"))
    t += 2

    def flashes(block):
        nonlocal t
        out.append((t, f"flashes_start block={block}"))
        t += 1
        for n in range(1, 6):
            out.append((t, f"flash n={n} block={block}"))
            t += 1
        out.append((t, f"flashes_end block={block}"))
        t += 1

    def pursuits(first):
        nonlocal t
        for k in (first, first + 1):
            phase = rng.uniform(0, 6.28)
            out.append((t, f"pursuit_ready trial={k}"))
            t += 1
            out.append((t, f"pursuit_start trial={k} duration="
                           f"{PURSUIT['duration']} ax={PURSUIT['ax']} "
                           f"ay={PURSUIT['ay']} fx={PURSUIT['fx']} "
                           f"fy={PURSUIT['fy']} phase={phase:.5f} rate=60"))
            t += PURSUIT["duration"]
            out.append((t, f"pursuit_end trial={k}"))
            t += 1

    if lab:
        flashes("start")
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
    if lab:
        pursuits(1)
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
    if lab:
        pursuits(3)
        flashes("end")
    out.append((t, "debriefing"))
    t += 2
    out.append((t, "procedure_end"))
    return out, crosses, t + 2


def make_dataset(root: Path, *, alpha=True, n170=True, gaze_offset=0.0,
                 marker_shift=0.0, video_delay=0.04, video_scale=1.0,
                 lab=False, gaze_delay=0.0, camera_delay=0.03, noisy=(),
                 seed=3):
    """Write a recording. ``marker_shift`` moves the markers (only) that
    many seconds late against everything else, as broken timing would;
    ``gaze_delay`` the gaze late against everything else. With ``lab``,
    the lab's procedure (flashes, moving targets), a horizontal EOG on a
    last channel and a camera that sees the flashes ``camera_delay``
    late."""
    rng = np.random.default_rng(seed)
    markers, crosses, duration = procedure(rng, lab=lab)
    labels = LABELS + (["HEOG"] if lab else [])
    when = {text.split()[0]: t for t, text in markers}
    n = int(duration * FS)
    t = np.arange(n) / FS
    eeg = rng.normal(0, 4, (n, len(labels)))
    for name in noisy:                 # poor contact: 150 µV of noise
        eeg[:, labels.index(name)] += rng.normal(0, 150, n)
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
    # Gaze: on the centre, then on each cross 250 ms after it appears,
    # following each moving target 30 ms behind it; lost while the eyes
    # are closed.
    gt = np.arange(0.5, duration - 0.5, 1 / GAZE_HZ)
    gx = np.full(len(gt), 0.5)
    gy = np.full(len(gt), 0.5)
    for time, x, y in crosses:
        on = gt >= time + 0.25
        gx[on], gy[on] = x, y
    for time, text in markers:
        if text.startswith("pursuit_start"):
            phase = float(text.split("phase=")[1].split()[0])
            on = (gt >= time) & (gt < time + PURSUIT["duration"])
            gx[on], gy[on] = pursuit_at(gt[on] - time - 0.03, phase)
    gx += rng.normal(0, 0.004, len(gt)) + gaze_offset
    gy += rng.normal(0, 0.004, len(gt))
    valid = ~((gt >= when["rest_closed_start"] + 0.4) &
              (gt < when["rest_closed_end"] + 0.3))
    if lab:
        # A horizontal EOG: 300 µV across the screen, with the eyes.
        eeg[:, labels.index("HEOG")] += 300 * (np.interp(t, gt, gx) - 0.5)
    eeg_dir = root / "sub-01" / "eeg"
    beh_dir = root / "sub-01" / "beh"
    eeg_dir.mkdir(parents=True)
    beh_dir.mkdir(parents=True)
    w = pyedflib.EdfWriter(str(eeg_dir / f"{STEM}_eeg.bdf"), len(labels),
                           file_type=pyedflib.FILETYPE_BDFPLUS)
    w.setSignalHeaders([{"label": lab, "dimension": "uV",
                         "sample_frequency": FS, "physical_min": -750000,
                         "physical_max": 750000, "digital_min": -8388608,
                         "digital_max": 8388607, "transducer": "",
                         "prefilter": ""} for lab in labels])
    whole = n - n % int(FS)
    w.writeSamples([np.ascontiguousarray(eeg[:whole, i])
                    for i in range(len(labels))])
    w.close()
    (eeg_dir / f"{STEM}_eeg.json").write_text(json.dumps({
        "SamplingFrequency": FS,
        "RecordingTiming": {"MeasuredSamplingFrequency": FS}}))
    (eeg_dir / f"{STEM}_channels.tsv").write_text(
        "name\ttype\tunits\n" + "".join(
            f"{c}\t{'EOG' if c == 'HEOG' else 'EEG'}\tuV\n" for c in labels))
    events = "onset\tduration\ttrial_type\tvalue\tsource\n" + "".join(
        f"{time + marker_shift:.6f}\t0\tbrAInstreamValidation\t{text}\t"
        f"lsl:brAInstreamValidation\n" for time, text in markers)
    (eeg_dir / f"{STEM}_events.tsv").write_text(events)
    (beh_dir / f"{STEM}_events.tsv").write_text(events)

    rows = np.column_stack([T0 + gt + gaze_delay, gx, gy,
                            valid.astype(float)])
    with gzip.open(beh_dir / f"{STEM}_recording-gaze_physio.tsv.gz",
                   "wt") as f:
        for r in rows:
            f.write("\t".join(f"{v:.6f}" for v in r) + "\n")
    (beh_dir / f"{STEM}_recording-gaze_physio.json").write_text(json.dumps({
        "StartTime": gt[0] + gaze_delay,
        "Columns": ["timestamp", "BPOGX", "BPOGY", "BPOGV"]}))
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
    flash_on = [time for time, text in markers if text.startswith("flash ")]
    ft = np.arange(0.2, duration - 0.5, 1 / FPS)
    for time in ft:
        frame = np.full((size[1], size[0], 3), 90, np.uint8)
        if any(on + video_delay <= time < on + video_delay + 0.2
               for on in flash_on):
            frame[:] = 250
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
    if lab:
        # The camera: a dim room, lit a little by each flash.
        name = f"{STEM}_recording-camera"
        writer = cv2.VideoWriter(str(beh_dir / f"{name}_video.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
        for time in ft:
            lit = any(on + camera_delay <= time < on + camera_delay + 0.2
                      for on in flash_on)
            level = 70 if lit else 40
            frame = np.clip(rng.normal(level, 3, (size[1], size[0], 3)), 0,
                            255).astype(np.uint8)
            writer.write(frame)
        writer.release()
        with gzip.open(beh_dir / f"{name}_physio.tsv.gz", "wt") as f:
            for i, time in enumerate(ft):
                f.write(f"{T0 + time:.6f}\t{i}\n")
        (beh_dir / f"{name}_physio.json").write_text(
            json.dumps({"StartTime": ft[0], "Columns": ["timestamp",
                                                         "frame"]}))
    return root


def filmed_eyes(delay):
    """Stands in for MediaPipe on the camera's video (a synthetic video has
    no face): the eyes it would find, ``delay`` s after the gaze."""
    def eyes(video, ft, windows):
        rec = filmed_eyes.rec
        rows = []
        for t in ft:
            if any(a <= t <= b for a, b in windows):
                x = np.interp(t - delay, rec.gaze_t, rec.gaze[:, 0])
                y = np.interp(t - delay, rec.gaze_t, rec.gaze[:, 1])
                ok = np.interp(t - delay, rec.gaze_t, rec.gaze[:, 2]) > 0.5
                rows.append((t, 30 * x, 20 * y, 1.0) if ok else
                            (t, np.nan, np.nan, 0.0))
        return np.array(rows)
    return eyes


def run(tmp_path, camera_eyes_delay=None, **kwargs):
    root = make_dataset(tmp_path / "bids", **kwargs)
    if camera_eyes_delay is None:
        checks = analyse.analyse(root, "01", out=tmp_path / "report")
    else:
        real_load, real_eyes = analyse.load, analyse._camera_eyes

        def load(*args, **kw):
            filmed_eyes.rec = real_load(*args, **kw)
            return filmed_eyes.rec
        analyse.load = load
        analyse._camera_eyes = filmed_eyes(camera_eyes_delay)
        try:
            checks = analyse.analyse(root, "01", out=tmp_path / "report")
        finally:
            analyse.load, analyse._camera_eyes = real_load, real_eyes
    assert [c.name for c in checks if c.info] == ["Screen recording setup"]
    key = {"EEG signal quality": "Quality"}
    return ({key.get(c.name, c.name.split()[0]): c for c in checks
             if not c.info}, tmp_path / "report")


def test_good_recording_passes(tmp_path):
    checks, report = run(tmp_path)
    for c in checks.values():
        assert c.passed, (c.name, c.lines)
    text = (report / "report.md").read_text()
    assert text.count("**PASS**") == 5
    sync = "\n".join(checks["Synchronisation"].lines)
    assert "250 ms" in sync or "233 ms" in sync or "267 ms" in sync
    assert (report / "n170.png").exists()


@pytest.mark.slow
def test_catches_no_alpha_effect(tmp_path):
    checks, _ = run(tmp_path, alpha=False)
    assert checks["EEG"].passed is False


@pytest.mark.slow
def test_catches_inaccurate_gaze(tmp_path):
    checks, _ = run(tmp_path, gaze_offset=0.06)   # about 3.5 degrees
    assert checks["Gaze"].passed is False


@pytest.mark.slow
def test_catches_missing_n170(tmp_path):
    checks, _ = run(tmp_path, n170=False)
    assert checks["N170"].passed is False


@pytest.mark.slow
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


@pytest.mark.slow
def test_make_project(tmp_path):
    import subprocess
    out = subprocess.run(
        [sys.executable, str(ROOT / "validation" / "make_project.py"),
         str(tmp_path / "v"), "--unicorn", "UN-1", "--gp3-host", "10.0.0.5"],
        capture_output=True, text=True, check=True)
    project = json.loads(next((tmp_path / "v").glob("*.brainstream"))
                         .read_text())
    assert project["task"] == "facevalidation"
    assert list(project["devices"]) == ["unicorn", "gp3", "lsl", "obs"]
    assert project["devices"]["gp3"]["mode"] == "remote"
    assert project["devices"]["gp3"]["host"] == "10.0.0.5"
    assert project["devices"]["lsl"]["streams"][0]["name"] == \
        "brAInstreamValidation"
    assert project["devices"]["obs"]["label"] == "screen"
    assert "Project:" in out.stdout


@pytest.mark.slow
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
    assert "Set the Screen Capture's Delay to" in text


@pytest.mark.slow
def test_screen_recording_scaled_but_in_time(tmp_path):
    checks, _ = run(tmp_path, video_scale=1.44)
    assert checks["Synchronisation"].passed, checks["Synchronisation"].lines


def test_lab_recording_passes(tmp_path):
    """The lab's procedure: flashes in the screen recording and the camera,
    the eyes on the moving targets, the EOG with the gaze."""
    checks, _ = run(tmp_path, lab=True, camera_eyes_delay=0.02)
    for c in checks.values():
        assert c.passed, (c.name, c.lines)
    sync = "\n".join(checks["Synchronisation"].lines)
    assert re.search(r"moving targets \*\*(2\d|3\d|4\d) ms", sync), sync
    assert re.search(r"EOG moves with the horizontal gaze, the gaze "
                     r"\*\*[+-]\d+ ms", sync), sync
    found = re.search(r"the camera films the eyes' jumps \*\*\+(\d+) ms",
                      sync)
    assert found and 0 <= int(found.group(1)) <= 50, sync
    found = re.search(r"flashes' light shows in the camera \*\*(\d+) ms",
                      sync)
    assert found and 20 <= int(found.group(1)) <= 70, sync
    assert "flashes appear in the screen recording" in sync
    assert "HEOG (EOG)" in "\n".join(checks["Quality"].lines)


@pytest.mark.slow
def test_catches_gaze_late_against_the_eeg(tmp_path):
    """The eye tracker's samples 150 ms late against the EEG and the
    markers: the moving targets and the EOG both show it."""
    checks, _ = run(tmp_path, lab=True, gaze_delay=0.15)
    sync = checks["Synchronisation"]
    assert sync.passed is False
    text = "\n".join(sync.lines)
    assert "✗ the EOG moves" in text, text


@pytest.mark.slow
def test_catches_poor_eeg(tmp_path):
    checks, _ = run(tmp_path, noisy=["PO7", "Fz"])
    assert checks["Quality"].passed is False
    assert "PO7" in "\n".join(checks["Quality"].lines)


@pytest.mark.slow
def test_make_lab_project(tmp_path):
    """The lab's setup: the g.HIamp's filters chosen by their cut-offs
    (here the simulated amplifier's), channel 64 an EOG, the GP3 and OBS
    on the stimulus computer, a camera here."""
    import subprocess
    names = tmp_path / "cap.txt"
    names.write_text("\n".join([f"E{i}" for i in range(1, 64)] + ["HEOG"]))
    out = subprocess.run(
        [sys.executable, str(ROOT / "validation" / "make_project.py"),
         str(tmp_path / "v"), "--lab", "--stimulus-host", "10.0.0.7",
         "--electrodes", str(names), "--simulate"],
        capture_output=True, text=True, check=True,
        env={**__import__("os").environ,
             "BRAINSTREAM_HOME": str(tmp_path / "home")})
    project = json.loads(next((tmp_path / "v").glob("*.brainstream"))
                         .read_text())
    devices = project["devices"]
    assert list(devices) == ["hiamp", "gp3", "lsl", "obs", "webcam"]
    hiamp = devices["hiamp"]
    assert (hiamp["channel_count"], hiamp["sampling_rate"]) == (64, 512)
    assert hiamp["channel_types"] == ["64=EOG"]
    assert hiamp["electrodes"][-1] == "HEOG"
    assert "0.01 – 100 Hz" in out.stdout and "50 Hz" in out.stdout
    assert devices["gp3"]["host"] == "10.0.0.7"
    assert devices["webcam"]["label"] == "camera"


@pytest.mark.slow
def test_catches_camera_late(tmp_path):
    """The camera's frames 200 ms late: its eyes move after the gaze, and
    the Delay to set is named."""
    checks, _ = run(tmp_path, lab=True, camera_eyes_delay=0.2,
                    camera_delay=0.2)
    sync = checks["Synchronisation"]
    assert sync.passed is False
    text = "\n".join(sync.lines)
    assert "✗ the camera films" in text, text
    assert "to the Video Stream's Delay" in text


@pytest.mark.slow
def test_camera_without_a_face_is_not_measured(tmp_path):
    """A camera that shows no face: the comparison is reported as not
    measured, and the camera's flashes still count."""
    checks, _ = run(tmp_path, lab=True, camera_eyes_delay=None)
    sync = checks["Synchronisation"]
    text = "\n".join(sync.lines)
    assert "camera's eyes: not measured" in text or \
        "do not follow the gaze" in text, text
    assert sync.passed, text


@pytest.mark.slow
def test_camera_seeing_nothing_fails(tmp_path):
    """No face, and no flashes' light: the camera's timing is unknown."""
    checks, _ = run(tmp_path, lab=True, camera_eyes_delay=None,
                    camera_delay=1e6)
    sync = checks["Synchronisation"]
    assert sync.passed is False
    assert "timing could not be measured" in "\n".join(sync.lines)


def test_recordings_from_before_the_rename_are_read(tmp_path):
    """Validation recordings made as brAIn dump: the procedure's markers
    came from the stream brAIndumpValidation."""
    root = make_dataset(tmp_path / "bids")
    for path in root.rglob("*_events.tsv"):
        path.write_text(path.read_text().replace(
            "brAInstreamValidation", "brAIndumpValidation"))
    rec = analyse.load(root, "01")
    assert rec.markers and rec.first("rest_open_start") is not None
