"""Check a validation recording (validation/face_validation.py) made with
brAIn dump: are the signals right, and are they in step?

    python validation/analyse.py <BIDS folder> --sub 01 [--ses 1]
        [--task facevalidation] [--run 1] [--out report_folder]

Four checks, each passed or failed against the criteria below, written to
``report.md`` (with figures, when matplotlib is installed) and printed:

1. **Alpha**: posterior alpha (8-13 Hz) is stronger with the eyes closed
   than open (the Berger effect): ratio >= 1.5.
2. **Gaze accuracy**: while the participant looks at each fixation cross
   (0.6-2.0 s after it appears), the median gaze is within 1.5 degrees
   of it, for at least 80 % of the crosses.
3. **N170**: the average EEG after the faces, at the occipito-temporal
   electrodes (PO7/PO8), has a negative peak between 130 and 200 ms that
   is reliably below zero across trials (one-sample t-test, p < 0.05).
4. **Synchronisation**: every device's data are on one timeline.
   - the markers in eeg/ and beh/ events.tsv are the same, to 1 ms;
   - the eyes reach each new cross 80-500 ms after its marker (a saccade
     cannot precede what it looks at, nor take half a second);
   - gaze is lost within 3 s of "eyes closed" and back within 3 s of the
     tone, and the EEG's alpha rises within 3 s of eyes closed;
   - in the screen recording, each face appears 0-200 ms after its marker
     (display, capture and NDI), with a spread (IQR) under 50 ms;
   - every recording covers the whole procedure.

Times are compared on the EEG's timeline: brAIn dump gives event onsets
and every recording's StartTime in it (eeg.json, RecordingTiming).
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import signal, stats

MARKER_SOURCE = "lsl:brAIndumpValidation"
ALPHA = (8.0, 13.0)
#: Electrodes where each effect is looked for, in order of preference.
POSTERIOR = ("Oz", "O1", "O2", "PO7", "PO8", "POz", "Pz", "P7", "P8")
N170_SITES = ("PO7", "PO8", "P7", "P8", "PO9", "PO10", "O1", "O2")
N170_WINDOW = (0.130, 0.200)
EPOCH = (-0.2, 0.6)


# -- loading -------------------------------------------------------------------------

def _tsv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    head = lines[0].split("\t")
    return [dict(zip(head, line.split("\t"))) for line in lines[1:] if line]


def _gz_rows(path: Path) -> np.ndarray:
    with gzip.open(path, "rt") as f:
        rows = [[np.nan if v == "n/a" else float(v)
                 for v in line.rstrip("\n").split("\t")] for line in f]
    return np.array(rows, dtype=float)


@dataclass
class Marker:
    onset: float
    text: str

    def field(self, name: str) -> Optional[str]:
        m = re.search(rf"\b{name}=(\S+)", self.text)
        return m.group(1) if m else None

    @property
    def kind(self) -> str:
        return self.text.split()[0]


@dataclass
class Recording:
    stem: str
    eeg: Optional[np.ndarray] = None          # (samples, channels), µV
    labels: list[str] = field(default_factory=list)
    eeg_types: list[str] = field(default_factory=list)
    fs: float = 250.0
    #: Measured rate / nominal rate: clock seconds to EEG seconds.
    k: float = 1.0
    markers: list[Marker] = field(default_factory=list)
    beh_markers: list[Marker] = field(default_factory=list)
    gaze_t: Optional[np.ndarray] = None       # EEG seconds
    gaze: Optional[np.ndarray] = None         # (n, 3): x, y, valid
    screen_m: Optional[tuple[float, float]] = None
    distance_m: Optional[float] = None
    video: Optional[Path] = None
    frame_t: Optional[np.ndarray] = None      # EEG seconds
    notes: list[str] = field(default_factory=list)

    def first(self, kind: str) -> Optional[Marker]:
        return next((m for m in self.markers if m.kind == kind), None)

    def all(self, kind: str) -> list[Marker]:
        return [m for m in self.markers if m.kind == kind]


def load(root: Path, sub: str, ses: str = "", task: str = "facevalidation",
         run: str = "") -> Recording:
    base = root / f"sub-{sub}" / (f"ses-{ses}" if ses else "")
    stem = f"sub-{sub}" + (f"_ses-{ses}" if ses else "") + f"_task-{task}"
    pattern = stem + (f"*_run-{run}" if run else "")
    rec = Recording(stem)
    eeg_dir, beh_dir = base / "eeg", base / "beh"

    bdfs = sorted(eeg_dir.glob(f"{pattern}*_eeg.bdf"))
    bdfs = [b for b in bdfs if "acq-unicorn" in b.name] or bdfs
    if bdfs:
        import pyedflib
        bdf = bdfs[0]
        reader = pyedflib.EdfReader(str(bdf))
        try:
            rec.labels = reader.getSignalLabels()
            rec.fs = float(reader.getSampleFrequency(0))
            rec.eeg = np.column_stack([reader.readSignal(i)
                                       for i in range(len(rec.labels))])
        finally:
            reader.close()
        side = bdf.with_name(bdf.name.replace("_eeg.bdf", "_eeg.json"))
        meta = json.loads(side.read_text()) if side.exists() else {}
        timing = meta.get("RecordingTiming", {})
        if timing.get("MeasuredSamplingFrequency"):
            rec.k = timing["MeasuredSamplingFrequency"] / rec.fs
        channels = _tsv(bdf.with_name(bdf.name.replace("_eeg.bdf",
                                                       "_channels.tsv")))
        types = {c["name"]: c["type"] for c in channels}
        rec.eeg_types = [types.get(label, "EEG") for label in rec.labels]
    else:
        rec.notes.append("No EEG recording found.")

    def markers(folder: Path) -> list[Marker]:
        for path in sorted(folder.glob(f"{pattern}*_events.tsv")):
            rows = _tsv(path)
            return [Marker(float(r["onset"]), r.get("value", ""))
                    for r in rows if r.get("source") == MARKER_SOURCE]
        return []
    rec.markers = markers(eeg_dir) or markers(beh_dir)
    rec.beh_markers = markers(beh_dir)
    if not rec.markers:
        rec.notes.append(f"No markers from {MARKER_SOURCE} found.")

    def physio(path: Path):
        meta = json.loads(path.with_name(path.name.replace(
            ".tsv.gz", ".json")).read_text())
        rows = _gz_rows(path)
        t = rows[:, 0]
        return meta, rows, meta.get("StartTime", 0.0) + (t - t[0]) * rec.k

    gaze = sorted(beh_dir.glob(f"{pattern}*_recording-gaze_physio.tsv.gz"))
    if gaze:
        meta, rows, t = physio(gaze[0])
        cols = meta["Columns"]
        rec.gaze_t = t
        rec.gaze = rows[:, [cols.index("BPOGX"), cols.index("BPOGY"),
                            cols.index("BPOGV")]]
        eye = gaze[0].with_name(gaze[0].name.replace("recording-gaze",
                                                     "recording-eye1")
                                .replace(".tsv.gz", ".json"))
        if eye.exists():
            e = json.loads(eye.read_text())
            if isinstance(e.get("ScreenSize"), list):
                rec.screen_m = tuple(e["ScreenSize"])
            if isinstance(e.get("ScreenDistance"), (int, float)):
                rec.distance_m = float(e["ScreenDistance"])
    else:
        rec.notes.append("No gaze recording found.")

    videos = sorted(beh_dir.glob(f"{pattern}*_recording-screen_video.mp4"))
    if videos:
        rec.video = videos[0]
        frames = videos[0].with_name(videos[0].name.replace(
            "_video.mp4", "_physio.tsv.gz"))
        if frames.exists():
            _, _, rec.frame_t = physio(frames)
    else:
        rec.notes.append("No screen recording (recording-screen) found.")
    return rec


# -- checks ----------------------------------------------------------------------------------

@dataclass
class Check:
    name: str
    passed: Optional[bool]               # None: could not be checked
    lines: list[str] = field(default_factory=list)
    figure: Optional[str] = None


def _channels(rec: Recording, preferred) -> list[int]:
    eeg = [i for i, t in enumerate(rec.eeg_types) if t == "EEG"]
    chosen = [i for name in preferred for i in eeg
              if rec.labels[i].upper() == name.upper()]
    return chosen or eeg


def _segment(rec: Recording, start: float, end: float) -> np.ndarray:
    a, b = int(round(start * rec.fs)), int(round(end * rec.fs))
    return rec.eeg[max(0, a):max(0, b)]


def _alpha_power(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    f, p = signal.welch(x, fs, nperseg=int(2 * fs), axis=0,
                        detrend="linear")
    band = (f >= ALPHA[0]) & (f <= ALPHA[1])
    return f, p, p[band].mean(axis=0)


def check_alpha(rec: Recording, plt=None, out: Path = None) -> Check:
    c = Check("EEG alpha, eyes closed vs open", None)
    names = ("rest_open_start", "rest_open_end", "rest_closed_start",
             "rest_closed_end")
    found = {n: rec.first(n) for n in names}
    if rec.eeg is None or not all(found.values()):
        c.lines.append("Needs the EEG and the four resting markers.")
        return c
    # Leave out the first 2 s of each period (settling, closing the eyes).
    open_ = _segment(rec, found["rest_open_start"].onset + 2,
                     found["rest_open_end"].onset)
    closed = _segment(rec, found["rest_closed_start"].onset + 2,
                      found["rest_closed_end"].onset)
    chans = _channels(rec, POSTERIOR)
    f, p_open, a_open = _alpha_power(open_[:, chans], rec.fs)
    _, p_closed, a_closed = _alpha_power(closed[:, chans], rec.fs)
    ratios = a_closed / a_open
    ratio = float(np.exp(np.mean(np.log(ratios))))
    c.passed = ratio >= 1.5
    c.lines.append(f"Alpha power (8-13 Hz), closed / open, geometric mean "
                   f"over {', '.join(rec.labels[i] for i in chans)}: "
                   f"**{ratio:.2f}** (pass: >= 1.5)")
    for i, r in zip(chans, ratios):
        c.lines.append(f"- {rec.labels[i]}: {r:.2f}")
    if plt is not None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        keep = (f >= 1) & (f <= 30)
        ax.semilogy(f[keep], p_open[keep].mean(axis=1), label="eyes open")
        ax.semilogy(f[keep], p_closed[keep].mean(axis=1),
                    label="eyes closed")
        ax.axvspan(*ALPHA, color="0.9")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (µV²/Hz)")
        ax.legend()
        c.figure = _save(fig, out, "alpha.png", plt)
    return c


def _degrees(rec: Recording, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Screen-fraction offsets as visual angle (degrees)."""
    w, h = rec.screen_m or (0.53, 0.30)
    d = rec.distance_m or 0.65
    return np.degrees(np.arctan2(np.hypot(dx * w, dy * h), d))


def _fixations(rec: Recording):
    out = []
    for m in rec.all("fixation"):
        x, y = m.field("x"), m.field("y")
        if x is not None and y is not None:
            out.append((m.onset, float(x), float(y)))
    return out


def check_gaze(rec: Recording, plt=None, out: Path = None) -> Check:
    c = Check("Gaze accuracy at the fixation crosses", None)
    fixations = _fixations(rec)
    if rec.gaze is None or not fixations:
        c.lines.append("Needs the gaze recording and the fixation markers.")
        return c
    errors, offsets, valid_share = [], [], []
    for onset, x, y in fixations:
        sel = (rec.gaze_t >= onset + 0.6) & (rec.gaze_t <= onset + 2.0)
        g = rec.gaze[sel]
        ok = g[:, 2] > 0
        valid_share.append(ok.mean() if len(g) else 0.0)
        if ok.sum() < 10:
            errors.append(np.nan)
            continue
        gx, gy = np.median(g[ok, 0]), np.median(g[ok, 1])
        offsets.append((gx - x, gy - y))
        errors.append(float(_degrees(rec, np.array(gx - x),
                                     np.array(gy - y))))
    errors = np.array(errors)
    measured = errors[~np.isnan(errors)]
    good = np.mean(measured <= 1.5) if len(measured) else 0.0
    median = float(np.median(measured)) if len(measured) else float("nan")
    c.passed = bool(len(measured) >= 0.8 * len(fixations)
                    and median <= 1.5 and good >= 0.8)
    c.lines.append(f"Crosses with gaze data: {len(measured)}/{len(fixations)}"
                   f"; median error **{median:.2f}°** (pass: <= 1.5°), "
                   f"{good * 100:.0f} % of crosses within 1.5° (pass: >= 80 %)")
    if offsets:
        o = np.array(offsets)
        c.lines.append(f"Mean offset (screen fraction): x {o[:, 0].mean():+.3f},"
                       f" y {o[:, 1].mean():+.3f}; valid gaze "
                       f"{np.mean(valid_share) * 100:.0f} % of samples")
    if rec.screen_m is None:
        c.lines.append("The screen's size was not recorded: degrees assume a "
                       "53 cm wide screen at 65 cm.")
    if plt is not None and offsets:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        o = np.array(offsets)
        ax.scatter(o[:, 0], o[:, 1], s=14)
        ax.axhline(0, color="0.7")
        ax.axvline(0, color="0.7")
        lim = max(0.05, np.abs(o).max() * 1.2)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(lim, -lim)
        ax.set_xlabel("gaze - cross, x (screen fraction)")
        ax.set_ylabel("gaze - cross, y")
        c.figure = _save(fig, out, "gaze.png", plt)
    return c


def check_n170(rec: Recording, plt=None, out: Path = None) -> Check:
    c = Check("N170 after the faces", None)
    faces = rec.all("face")
    if rec.eeg is None or len(faces) < 5:
        c.lines.append("Needs the EEG and the face markers.")
        return c
    chans = _channels(rec, N170_SITES)
    sos = signal.butter(4, [0.5, 30], btype="bandpass", fs=rec.fs,
                        output="sos")
    eeg = signal.sosfiltfilt(sos, rec.eeg[:, chans], axis=0)
    a, b = int(EPOCH[0] * rec.fs), int(EPOCH[1] * rec.fs)
    epochs = []
    for m in faces:
        i = int(round(m.onset * rec.fs))
        if i + a >= 0 and i + b <= len(eeg):
            e = eeg[i + a:i + b]
            e = e - e[:-a].mean(axis=0)          # baseline: -200..0 ms
            epochs.append(e.mean(axis=1))        # over the sites
    epochs = np.array(epochs)
    # Leave out epochs with artefacts (beyond ±100 µV).
    clean = epochs[np.abs(epochs).max(axis=1) < 100]
    t = np.arange(a, b) / rec.fs
    if len(clean) < 5:
        c.lines.append(f"Only {len(clean)} of {len(epochs)} epochs free of "
                       f"artefacts (±100 µV).")
        c.passed = False
        return c
    erp = clean.mean(axis=0)
    window = (t >= N170_WINDOW[0] - 0.03) & (t <= N170_WINDOW[1] + 0.03)
    peak = np.flatnonzero(window)[np.argmin(erp[window])]
    latency = float(t[peak])
    around = (t >= latency - 0.02) & (t <= latency + 0.02)
    amplitudes = clean[:, around].mean(axis=1)
    tstat, p = stats.ttest_1samp(amplitudes, 0.0)
    in_window = N170_WINDOW[0] <= latency <= N170_WINDOW[1]
    c.passed = bool(in_window and amplitudes.mean() < 0 and p / 2 < 0.05)
    c.lines.append(f"Sites: {', '.join(rec.labels[i] for i in chans)}; "
                   f"{len(clean)}/{len(epochs)} epochs")
    c.lines.append(f"Negative peak at **{latency * 1000:.0f} ms** (pass: "
                   f"{N170_WINDOW[0] * 1000:.0f}-{N170_WINDOW[1] * 1000:.0f} ms),"
                   f" {amplitudes.mean():+.1f} µV, t({len(clean) - 1}) = "
                   f"{tstat:.2f}, one-sided p = {p / 2:.3g} (pass: < 0.05)")
    if plt is not None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sem = clean.std(axis=0) / np.sqrt(len(clean))
        ax.fill_between(t * 1000, erp - sem, erp + sem, alpha=0.3)
        ax.plot(t * 1000, erp)
        ax.axvspan(N170_WINDOW[0] * 1000, N170_WINDOW[1] * 1000,
                   color="0.9", zorder=0)
        ax.axvline(0, color="0.5")
        ax.axhline(0, color="0.7")
        ax.invert_yaxis()
        ax.set_xlabel("Time from face onset (ms)")
        ax.set_ylabel("µV (negative up)")
        c.figure = _save(fig, out, "n170.png", plt)
    return c


def _saccade_latencies(rec: Recording) -> list[float]:
    """For each new cross: when the gaze first comes within 3 degrees of
    it, from its marker."""
    out = []
    for onset, x, y in _fixations(rec):
        sel = (rec.gaze_t >= onset - 0.2) & (rec.gaze_t <= onset + 1.5)
        t, g = rec.gaze_t[sel], rec.gaze[sel]
        ok = g[:, 2] > 0
        near = ok & (_degrees(rec, g[:, 0] - x, g[:, 1] - y) <= 3.0)
        # Only crosses the gaze was not already on.
        before = near & (t < onset)
        if before.any() or not near.any():
            continue
        out.append(float(t[np.argmax(near)] - onset))
    return out


def _video_latencies(rec: Recording) -> list[float]:
    """For each face: when it appears in the screen recording, from its
    marker (the first frame where the face's place changes)."""
    import cv2
    faces = [(m.onset, float(m.field("x")), float(m.field("y")))
             for m in rec.all("face") if m.field("x") is not None]
    if not faces or rec.frame_t is None:
        return []
    windows = [(onset - 0.5, onset + 0.6) for onset, _, _ in faces]
    series: list[list[tuple[float, float]]] = [[] for _ in faces]
    capture = cv2.VideoCapture(str(rec.video))
    i = 0
    while True:
        ok, frame = capture.read()
        if not ok or i >= len(rec.frame_t):
            break
        t = rec.frame_t[i]
        i += 1
        for k, (lo, hi) in enumerate(windows):
            if lo <= t <= hi:
                h, w = frame.shape[:2]
                _, x, y = faces[k]
                r = int(0.06 * h)
                cx, cy = int(x * w), int(y * h)
                roi = frame[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
                series[k].append((t, float(roi.mean())))
    capture.release()
    out = []
    for (onset, _, _), s in zip(faces, series):
        if len(s) < 5:
            continue
        t = np.array([v[0] for v in s])
        v = np.array([v[1] for v in s])
        base = v[t < onset - 0.1]
        if len(base) < 2:
            continue
        threshold = max(8.0, 6 * np.median(np.abs(base - np.median(base))))
        changed = (np.abs(v - np.median(base)) > threshold) & \
            (t >= onset - 0.3)
        if changed.any():
            out.append(float(t[np.argmax(changed)] - onset))
    return out


def check_sync(rec: Recording, plt=None, out: Path = None) -> Check:
    c = Check("Synchronisation of EEG, markers, eyes and screen", None)
    verdicts: list[bool] = []

    # Markers: the same in eeg/ and beh/.
    if rec.markers and rec.beh_markers:
        pairs = list(zip(rec.markers, rec.beh_markers))
        same = len(rec.markers) == len(rec.beh_markers) and all(
            a.text == b.text for a, b in pairs)
        diff = max(abs(a.onset - b.onset) for a, b in pairs) * 1000
        ok = same and diff <= 1.0
        verdicts.append(ok)
        c.lines.append(f"- {'✓' if ok else '✗'} markers in eeg/ and beh/ "
                       f"events.tsv: {len(rec.markers)} and "
                       f"{len(rec.beh_markers)}, onsets within "
                       f"{diff:.3f} ms (pass: same, <= 1 ms)")

    # Coverage: every recording spans the procedure.
    if rec.markers:
        start, end = rec.markers[0].onset, rec.markers[-1].onset
        spans = {}
        if rec.eeg is not None:
            spans["EEG"] = (0.0, len(rec.eeg) / rec.fs)
        if rec.gaze_t is not None:
            spans["gaze"] = (rec.gaze_t[0], rec.gaze_t[-1])
        if rec.frame_t is not None:
            spans["screen"] = (rec.frame_t[0], rec.frame_t[-1])
        for name, (a, b) in spans.items():
            ok = a <= start + 0.5 and b >= end - 0.5
            verdicts.append(ok)
            c.lines.append(f"- {'✓' if ok else '✗'} {name} covers the "
                           f"procedure ({a:.1f}-{b:.1f} s; markers "
                           f"{start:.1f}-{end:.1f} s)")

    # Eyes vs markers: saccades to each new cross.
    latencies = _saccade_latencies(rec) if rec.gaze is not None else []
    if latencies:
        med = float(np.median(latencies)) * 1000
        ok = 80 <= med <= 500
        verdicts.append(ok)
        c.lines.append(f"- {'✓' if ok else '✗'} the eyes reach a new cross "
                       f"**{med:.0f} ms** after its marker (median of "
                       f"{len(latencies)}; pass: 80-500 ms)")

    # Eyes and EEG vs markers: closing and opening the eyes.
    closed, opened = rec.first("rest_closed_start"), rec.first(
        "rest_closed_end")
    if rec.gaze is not None and closed and opened:
        lost = _first_run(rec.gaze_t, rec.gaze[:, 2] <= 0, closed.onset, 0.5)
        back = _first_run(rec.gaze_t, rec.gaze[:, 2] > 0, opened.onset, 0.5)
        ok = lost is not None and back is not None and 0 <= lost <= 3 and \
            0 <= back <= 3
        verdicts.append(ok)
        c.lines.append(f"- {'✓' if ok else '✗'} gaze lost "
                       f"{_ms(lost)} after 'eyes closed', back {_ms(back)} "
                       f"after the tone (pass: 0-3 s)")
    if rec.eeg is not None and closed and rec.first("rest_open_start"):
        rise = _alpha_rise(rec, rec.first("rest_open_start").onset,
                           closed.onset, opened.onset if opened else None)
        if rise is not None:
            ok = 0 <= rise <= 3
            verdicts.append(ok)
            c.lines.append(f"- {'✓' if ok else '✗'} EEG alpha rises "
                           f"{_ms(rise)} after 'eyes closed' (pass: 0-3 s)")

    # Screen vs markers: each face in the screen recording.
    video = _video_latencies(rec) if rec.video is not None else []
    n_faces = len(rec.all("face"))
    if rec.video is not None and n_faces and len(video) < n_faces / 2:
        verdicts.append(False)
        c.lines.append(f"- ✗ faces found in the screen recording for only "
                       f"{len(video)} of {n_faces} markers (pass: at least "
                       f"half): is OBS capturing the stimulus screen?")
        video = []
    if video:
        v = np.array(video) * 1000
        med = float(np.median(v))
        iqr = float(np.subtract(*np.percentile(v, [75, 25])))
        ok = 0 <= med <= 200 and iqr < 50
        verdicts.append(ok)
        c.lines.append(f"- {'✓' if ok else '✗'} faces appear in the screen "
                       f"recording **{med:.0f} ms** after their markers "
                       f"(IQR {iqr:.0f} ms, {len(v)} faces; pass: 0-200 ms, "
                       f"IQR < 50 ms)")
        if plt is not None:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(v, bins=20)
            if latencies:
                ax.hist(np.array(latencies) * 1000, bins=20, alpha=0.6)
            ax.set_xlabel("ms after the marker")
            ax.legend(["face on screen (video)"] +
                      (["gaze on the new cross"] if latencies else []))
            c.figure = _save(fig, out, "sync.png", plt)
    c.passed = all(verdicts) if verdicts else None
    if not verdicts:
        c.lines.append("Nothing to compare: needs markers and at least one "
                       "other recording.")
    return c


def _first_run(t: np.ndarray, condition: np.ndarray, start: float,
               length: float) -> Optional[float]:
    """When, after ``start``, ``condition`` first holds for ``length`` s."""
    after = t >= start
    idx = np.flatnonzero(after)
    run_start = None
    for i in idx:
        if condition[i]:
            if run_start is None:
                run_start = t[i]
            if t[i] - run_start >= length:
                return float(run_start - start)
        else:
            run_start = None
    return None


def _alpha_rise(rec: Recording, open_start: float, closed: float,
                opened: Optional[float]) -> Optional[float]:
    """When the posterior alpha envelope crosses halfway between its
    eyes-open and eyes-closed levels, after the eyes close."""
    chans = _channels(rec, POSTERIOR)
    sos = signal.butter(4, ALPHA, btype="bandpass", fs=rec.fs, output="sos")
    x = signal.sosfiltfilt(sos, rec.eeg[:, chans], axis=0)
    env = np.abs(signal.hilbert(x, axis=0)).mean(axis=1)
    smooth = int(0.25 * rec.fs)
    env = np.convolve(env, np.ones(smooth) / smooth, mode="same")
    t = np.arange(len(env)) / rec.fs
    end = opened if opened is not None else closed + 20
    open_part = env[(t >= open_start + 2) & (t < closed)]
    closed_part = env[(t >= closed + 5) & (t < end)]
    if not len(open_part) or not len(closed_part):
        return None
    level_open, level_closed = np.median(open_part), np.median(closed_part)
    if not level_closed > level_open:
        return None
    half = (level_open + level_closed) / 2
    return _first_run(t, env >= half, closed, 0.5)


def _ms(seconds: Optional[float]) -> str:
    return "never" if seconds is None else f"{seconds * 1000:.0f} ms"


def _save(fig, out: Optional[Path], name: str, plt) -> Optional[str]:
    if out is None:
        plt.close(fig)
        return None
    fig.tight_layout()
    fig.savefig(out / name, dpi=110)
    plt.close(fig)
    return name


# -- the report ---------------------------------------------------------------------------------

def analyse(root: Path, sub: str, ses: str = "", task: str = "facevalidation",
            run: str = "", out: Optional[Path] = None) -> list[Check]:
    rec = load(root, sub, ses, task, run)
    plt = None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: F811
        except ImportError:
            plt = None
    checks = [check_alpha(rec, plt, out), check_gaze(rec, plt, out),
              check_n170(rec, plt, out), check_sync(rec, plt, out)]
    if out is not None:
        _write_report(out / "report.md", rec, checks)
    return checks


def _verdict(passed: Optional[bool]) -> str:
    return {True: "PASS", False: "FAIL", None: "NOT CHECKED"}[passed]


def _write_report(path: Path, rec: Recording, checks: list[Check]) -> None:
    lines = [f"# brAIn dump validation: {rec.stem}", ""]
    lines += [f"> {note}" for note in rec.notes]
    lines += ["", "| Check | Result |", "|---|---|"]
    lines += [f"| {c.name} | **{_verdict(c.passed)}** |" for c in checks]
    for c in checks:
        lines += ["", f"## {c.name}: {_verdict(c.passed)}", ""] + c.lines
        if c.figure:
            lines += ["", f"![{c.name}]({c.figure})"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("bids", type=Path, help="the BIDS dataset folder")
    p.add_argument("--sub", required=True)
    p.add_argument("--ses", default="")
    p.add_argument("--task", default="facevalidation")
    p.add_argument("--run", default="")
    p.add_argument("--out", type=Path, default=None,
                   help="report folder (default: <bids>/derivatives/"
                        "validation/sub-<sub>)")
    args = p.parse_args(argv)
    out = args.out or (args.bids / "derivatives" / "validation"
                       / f"sub-{args.sub}")
    checks = analyse(args.bids, args.sub, args.ses, args.task, args.run, out)
    for c in checks:
        print(f"\n[{_verdict(c.passed)}] {c.name}")
        for line in c.lines:
            print("   " + line.replace("**", ""))
    print(f"\nReport: {out / 'report.md'}")
    return 0 if all(c.passed for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
