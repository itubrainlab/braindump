"""Check a validation recording (validation/face_validation.py) made with
brAIn dump: are the signals right, and are they in step?

    python validation/analyse.py <BIDS folder> --sub 01 [--ses 1]
        [--task facevalidation] [--run 1] [--out report_folder]

Five checks, each passed or failed against the criteria below, written to
``report.md`` (with figures, when matplotlib is installed) and printed:

1. **Alpha**: posterior alpha (8-13 Hz) is stronger with the eyes closed
   than open (the Berger effect): ratio >= 1.5.
2. **EEG signal quality**: during the faces, at least 90 % of the EEG
   channels, and every N170 and posterior site used, have 1-30 Hz noise
   under 50 µV (SD).
3. **Gaze accuracy**: while the participant looks at each fixation cross
   (0.6-2.0 s after it appears), the median gaze is within 1.5 degrees
   of it, for at least 80 % of the crosses.
4. **N170**: the average EEG after the faces, at the occipito-temporal
   electrodes (PO7/PO8, average reference), has a negative peak between
   130 and 200 ms that is reliably below zero across trials (one-sample
   t-test, p < 0.05).
5. **Synchronisation**: every device's data are on one timeline.
   - the markers in eeg/ and beh/ events.tsv are the same, to 1 ms;
   - the eyes reach each new cross 80-500 ms after its marker (a saccade
     cannot precede what it looks at, nor take half a second);
   - the eyes follow the moving targets -50 to 150 ms behind them (smooth
     pursuit of a predictable path lags little);
   - the EOG (a channel typed EOG) moves with the gaze within 40 ms:
     EEG against eye tracker, with no marker in between;
   - gaze is lost within 3 s of "eyes closed"; the EEG's alpha rise is
     reported;
   - in the screen recording, each face and each flash appears -20 to
     200 ms after its marker (what the Screen Capture's Delay leaves: 0
     when it is set right, give or take a frame), with an IQR under 50 ms;
   - the camera, filming the participant: the eyes' jumps to each new
     cross, timed in its video (MediaPipe's face landmarks, as OpenGP3's
     webcam tracking finds the eyes) and in the gaze, within 60 ms of
     each other (IQR < 100 ms); and, when their light on the participant
     shows, the flashes within 60 ms of their markers;
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
    #: Markers recorded more than once (same text, same onset), left out
    #: of ``markers``.
    duplicates: int = 0
    gaze_t: Optional[np.ndarray] = None       # EEG seconds
    gaze: Optional[np.ndarray] = None         # (n, 3): x, y, valid
    eyes_valid: Optional[np.ndarray] = None   # (n, 2): left, right valid
    screen_m: Optional[tuple[float, float]] = None
    screen_px: Optional[tuple[int, int]] = None
    distance_m: Optional[float] = None
    video: Optional[Path] = None
    frame_t: Optional[np.ndarray] = None      # EEG seconds
    camera: Optional[Path] = None
    camera_t: Optional[np.ndarray] = None     # EEG seconds
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
    bdfs = [b for b in bdfs if "acq-unicorn" in b.name] or \
        [b for b in bdfs if "acq-hiamp" in b.name] or bdfs
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
    # A marker recorded twice would count its trial twice: keep each once.
    seen, unique = set(), []
    for m in rec.markers:
        key = (m.text, round(m.onset, 4))
        if key not in seen:
            seen.add(key)
            unique.append(m)
    rec.duplicates = len(rec.markers) - len(unique)
    rec.markers = unique
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
        if "LPOGV" in cols and "RPOGV" in cols:
            rec.eyes_valid = rows[:, [cols.index("LPOGV"),
                                      cols.index("RPOGV")]]
        eye = gaze[0].with_name(gaze[0].name.replace("recording-gaze",
                                                     "recording-eye1")
                                .replace(".tsv.gz", ".json"))
        if eye.exists():
            e = json.loads(eye.read_text())
            if isinstance(e.get("ScreenSize"), list):
                rec.screen_m = tuple(e["ScreenSize"])
            if isinstance(e.get("ScreenDistance"), (int, float)):
                rec.distance_m = float(e["ScreenDistance"])
            if isinstance(e.get("ScreenResolution"), list):
                rec.screen_px = tuple(int(v) for v in e["ScreenResolution"])
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
    cameras = sorted(p for p in beh_dir.glob(f"{pattern}*_video.mp4")
                     if "recording-screen" not in p.name)
    if cameras:
        rec.camera = cameras[0]
        frames = cameras[0].with_name(cameras[0].name.replace(
            "_video.mp4", "_physio.tsv.gz"))
        if frames.exists():
            _, _, rec.camera_t = physio(frames)
    return rec


# -- checks ----------------------------------------------------------------------------------

@dataclass
class Check:
    name: str
    passed: Optional[bool]               # None: could not be checked
    lines: list[str] = field(default_factory=list)
    figure: Optional[str] = None
    #: Information beside the checks, not passed or failed.
    info: bool = False


def _channels(rec: Recording, preferred) -> list[int]:
    eeg = [i for i, t in enumerate(rec.eeg_types) if t == "EEG"]
    chosen = [i for name in preferred for i in eeg
              if rec.labels[i].upper() == name.upper()]
    return chosen or eeg


def _segment(rec: Recording, start: float, end: float) -> np.ndarray:
    a, b = int(round(start * rec.fs)), int(round(end * rec.fs))
    return rec.eeg[max(0, a):max(0, b)]


def _alpha_power(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    # The median over the 2 s windows: a movement or a blink in one window
    # does not count as alpha.
    f, p = signal.welch(x, fs, nperseg=int(2 * fs), axis=0,
                        detrend="linear", average="median")
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
    if rec.eyes_valid is not None:
        one_eye, big = 0, 0
        for (onset, x, y), err in zip(fixations, errors):
            sel = (rec.gaze_t >= onset + 0.6) & (rec.gaze_t <= onset + 2.0)
            share = rec.eyes_valid[sel].mean(axis=0) if sel.any() else \
                np.zeros(2)
            if share.min() < 0.5 <= share.max():
                one_eye += 1
                big += int(err == err and err > 1.5)
        if one_eye:
            c.lines.append(f"Crosses seen by one eye only: {one_eye} ({big} of "
                           f"them over 1.5°). The gaze is the mean of the eyes "
                           f"seen, so when one is lost it moves to the other "
                           f"eye's calibration: at the edges of the GP3's "
                           f"range, keep stimuli further in.")
    regions = {"top": [], "middle": [], "bottom": []}
    for (onset, x, y), share in zip(fixations, valid_share):
        key = "top" if y < 1 / 3 else "middle" if y < 2 / 3 else "bottom"
        regions[key].append(share)
    c.lines.append("Valid gaze at the crosses, by height on the screen: " +
                   ", ".join(f"{k} {np.mean(v) * 100:.0f} % ({len(v)})"
                             for k, v in regions.items() if v))
    weak = [k for k, v in regions.items() if v and np.mean(v) < 0.6]
    if weak:
        c.lines.append(f"The eye tracker loses the eyes at the {' and '.join(weak)}"
                       f" of the screen: check the GP3's angle and the "
                       f"participant's height and distance (the eye view's "
                       f"hints), or keep the stimuli within its range.")
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
    every = [i for i, t in enumerate(rec.eeg_types) if t == "EEG"]
    filtered = signal.sosfiltfilt(sos, rec.eeg[:, every], axis=0)
    # The average reference, as N170 studies use: a mastoid reference
    # (the Unicorn's) lies next to PO7/PO8 and takes most of it away.
    reference = "average"
    if len(every) >= 4:
        filtered = filtered - filtered.mean(axis=1, keepdims=True)
    else:
        reference = "as recorded"
    eeg = filtered[:, [every.index(i) for i in chans]]
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
    c.lines.append(f"Sites: {', '.join(rec.labels[i] for i in chans)} "
                   f"({reference} reference, 0.5-30 Hz); "
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


def check_eeg_quality(rec: Recording) -> Check:
    """Each EEG channel's noise and mains hum during the faces: at least
    90 % of the channels, and every site the alpha and N170 checks use,
    must have 1-30 Hz noise under 50 µV (SD)."""
    c = Check("EEG signal quality", None)
    if rec.eeg is None:
        c.lines.append("No EEG.")
        return c
    faces, blanks = rec.all("face"), rec.all("blank")
    lo = faces[0].onset if faces else 0.0
    hi = blanks[-1].onset + 1 if blanks else len(rec.eeg) / rec.fs
    x = _segment(rec, lo, hi)
    eeg = [i for i, t in enumerate(rec.eeg_types) if t == "EEG"]
    f, p = signal.welch(x[:, eeg], rec.fs, nperseg=int(2 * rec.fs), axis=0,
                        detrend="linear")
    df = f[1] - f[0]
    mains = np.sqrt(p[(f >= 48) & (f <= 62)].sum(axis=0) * df)
    sos = signal.butter(4, [1, 30], btype="bandpass", fs=rec.fs,
                        output="sos")
    sd = signal.sosfiltfilt(sos, x[:, eeg], axis=0).std(axis=0)
    poor = [rec.labels[i] for i, s_ in zip(eeg, sd) if s_ > 50]
    key = {rec.labels[i] for i in _channels(rec, N170_SITES)
           + _channels(rec, POSTERIOR)}
    key_poor = sorted(key & set(poor))
    share = 1 - len(poor) / len(eeg)
    c.passed = bool(share >= 0.9 and not key_poor)
    c.lines.append(f"During the faces: **{share * 100:.0f} %** of {len(eeg)} "
                   f"EEG channels with noise under 50 µV (pass: >= 90 %, and "
                   f"every site the alpha and N170 checks use)")
    others = [f"{rec.labels[i]} ({t})" for i, t in enumerate(rec.eeg_types)
              if t not in ("EEG", "TRIG", "MISC")]
    if others:
        c.lines.append(f"Not EEG, left out: {', '.join(others)}")
    if poor:
        c.lines.append(f"Poor contact: {', '.join(poor)}"
                       + (f"; among the sites the checks use: "
                          f"{', '.join(key_poor)}" if key_poor else "")
                       + ". Add gel or reseat these electrodes, and check the "
                         "reference and ground; the data viewer should show "
                         "a clean signal on each before recording.")
    c.lines.append("")
    c.lines.append("Poor contact shows as noise over 50 µV (1-30 Hz SD); "
                   "strong mains hum (over 50 µV RMS) hints at high "
                   "impedance, but the amplifier's notch and the checks' "
                   "filters take it out.")
    c.lines.append("")
    c.lines.append("| Channel | Mains (µV RMS) | Noise 1-30 Hz (µV SD) | |")
    c.lines.append("|---|---|---|---|")
    for i, m, s_ in zip(eeg, mains, sd):
        verdict = "poor" if s_ > 50 else "good, mains high" if m > 50 \
            else "good"
        c.lines.append(f"| {rec.labels[i]} | {m:.1f} | {s_:.1f} | "
                       f"{verdict} |")
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
    """For each face: when the screen recording shows it, from its marker.

    The face replaces the cross, so the picture changes: the first frame
    after the marker that differs from the one before it (more than the
    frames before had changed) is when the face arrived. Measured on the
    whole picture, so it does not matter how OBS scales or crops the
    screen into its canvas."""
    import cv2
    faces = [m.onset for m in rec.all("face")]
    if not faces or rec.frame_t is None:
        return []
    ft = rec.frame_t
    wanted = np.zeros(len(ft), bool)
    for onset in faces:
        wanted |= (ft >= onset - 1.0) & (ft <= onset + 0.9)
    diffs = np.full(len(ft), np.nan)
    capture = cv2.VideoCapture(str(rec.video))
    previous, i = None, 0
    while i < len(ft):
        if not wanted[i] and not (i + 1 < len(ft) and wanted[i + 1]):
            if not capture.grab():
                break
            previous, i = None, i + 1
            continue
        ok, frame = capture.read()
        if not ok:
            break
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                           (240, 135)).astype(np.float32)
        if previous is not None:
            diffs[i] = float(np.abs(small - previous).mean())
        previous, i = small, i + 1
    capture.release()
    out = []
    for onset in faces:
        base = diffs[(ft >= onset - 0.9) & (ft < onset - 0.25)]
        base = base[~np.isnan(base)]
        window = np.flatnonzero((ft >= onset - 0.2) & (ft <= onset + 0.8))
        if len(base) < 3 or not len(window):
            continue
        threshold = max(0.5, 5 * float(np.median(base)))
        changed = window[np.nan_to_num(diffs[window]) > threshold]
        if len(changed):
            out.append(float(ft[changed[0]] - onset))
    return out


def _brightness(video: Path, ft: np.ndarray,
                windows: list[tuple[float, float]]) -> np.ndarray:
    """Each frame's mean brightness (0-255), for the frames within
    ``windows`` (EEG seconds); NaN elsewhere."""
    import cv2
    wanted = np.zeros(len(ft), bool)
    for a, b in windows:
        wanted |= (ft >= a) & (ft <= b)
    out = np.full(len(ft), np.nan)
    capture = cv2.VideoCapture(str(video))
    for i in range(len(ft)):
        if not wanted[i]:
            if not capture.grab():
                break
            continue
        ok, frame = capture.read()
        if not ok:
            break
        out[i] = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
    capture.release()
    return out


def _flash_latencies(rec: Recording, video: Optional[Path],
                     ft: Optional[np.ndarray]) -> tuple[list[float], int]:
    """For each flash: when the video first shows the light, from its
    marker; and how many flashes there were. The light is found as the
    frame passing halfway from the dark before the flash to the brightest
    frame after it, so a camera that sees only the light the screen casts
    on the room finds it too."""
    flashes = [m.onset for m in rec.all("flash")]
    if video is None or ft is None or not flashes:
        return [], len(flashes)
    light = _brightness(video, ft, [(t - 0.6, t + 0.7) for t in flashes])
    out = []
    for onset in flashes:
        base = light[(ft >= onset - 0.5) & (ft < onset - 0.05)]
        after = np.flatnonzero((ft >= onset - 0.1) & (ft <= onset + 0.6))
        base = base[~np.isnan(base)]
        if len(base) < 2 or not len(after):
            continue
        peak = np.nanmax(light[after])
        dark = float(np.median(base))
        if not peak - dark > max(2.0, 4 * float(np.std(base))):
            continue
        lit = after[light[after] >= dark + (peak - dark) / 2]
        if len(lit):
            out.append(float(ft[lit[0]] - onset))
    return out, len(flashes)


def _pursuit_lags(rec: Recording) -> list[tuple[float, float]]:
    """For each moving target: how far behind it the gaze is (s), and the
    gaze's error from the target at that lag (degrees). The lag is the
    shift of the target's path that brings it closest to the gaze, over
    the movement after its first second (once the eyes have caught up)."""
    out = []
    for m in rec.all("pursuit_start"):
        try:
            duration, ax, ay, fx, fy, phase = (
                float(m.field(k)) for k in ("duration", "ax", "ay", "fx",
                                             "fy", "phase"))
        except (TypeError, ValueError):
            continue
        sel = (rec.gaze_t >= m.onset + 1.0) & (rec.gaze_t <= m.onset + duration)
        sel &= rec.gaze[:, 2] > 0
        if sel.sum() < 30:
            continue
        t = rec.gaze_t[sel] - m.onset
        gx, gy = rec.gaze[sel, 0], rec.gaze[sel, 1]
        lags = np.arange(-0.3, 0.5001, 0.005)
        errors = []
        for lag in lags:
            tt = t - lag
            px = 0.5 + ax * np.sin(2 * np.pi * fx * tt)
            py = 0.5 + ay * np.sin(2 * np.pi * fy * tt + phase)
            errors.append(np.median(_degrees(rec, gx - px, gy - py)))
        best = int(np.argmin(errors))
        out.append((float(lags[best]), float(errors[best])))
    return out


def _xcorr(a: np.ndarray, b: np.ndarray, steps: int = 30
           ) -> tuple[int, float]:
    """The shift (in samples) of ``b`` against ``a`` that correlates them
    best, and that correlation; positive: ``b`` comes later."""
    a = (a - a.mean()) / (a.std() or 1)
    b = (b - b.mean()) / (b.std() or 1)
    lags = np.arange(-steps, steps + 1)
    r = np.array([np.mean(a[max(0, -k):len(a) - max(0, k)]
                          * b[max(0, k):len(b) - max(0, -k)]) for k in lags])
    k = int(np.argmax(np.abs(r)))
    return int(lags[k]), float(r[k])


def _gaze_velocity(rec: Recording, grid: np.ndarray, axis: int
                   ) -> np.ndarray:
    """The gaze's velocity along ``axis`` on ``grid`` (s); 0 where it is
    not valid."""
    ok = rec.gaze[:, 2] > 0
    valid = np.interp(grid, rec.gaze_t, ok.astype(float)) > 0.99
    g = np.interp(grid, rec.gaze_t[ok], rec.gaze[ok, axis])
    v = np.diff(g, prepend=g[0])
    v[~valid] = 0
    return v


def _eog_lag(rec: Recording) -> Optional[tuple[float, float, str]]:
    """How far the gaze lags the EOG (s), their correlation, and the
    direction (horizontal or vertical) the EOG follows: the eye movements
    measured by the EEG amplifier and by the eye tracker, compared
    directly, with no marker in between. From the calibration to the end,
    on their velocities (the saccades), 100 Hz."""
    eog = [i for i, t in enumerate(rec.eeg_types) if t == "EOG"]
    start = rec.first("calibration_end")
    if not eog or rec.gaze is None or rec.eeg is None or start is None:
        return None
    grid = np.arange(start.onset, rec.markers[-1].onset, 0.01)
    if len(grid) < 1000:
        return None
    sos = signal.butter(2, [0.2, 15], btype="bandpass", fs=rec.fs,
                        output="sos")
    e = signal.sosfiltfilt(sos, rec.eeg[:, eog[0]])
    e = np.interp(grid, np.arange(len(e)) / rec.fs, e)
    ev = np.diff(e, prepend=e[0])
    best = None
    for axis, name in ((0, "horizontal"), (1, "vertical")):
        k, r = _xcorr(ev, _gaze_velocity(rec, grid, axis))
        if best is None or abs(r) > abs(best[1]):
            best = (k * 0.01, r, name)
    return best


def _camera_eyes(video: Path, ft: np.ndarray,
                 windows: list[tuple[float, float]]):
    """The eyes in the camera's video, by MediaPipe's face landmarks (as
    OpenGP3's webcam eye tracking finds them): for each frame within
    ``windows``, its time, the iris's horizontal and vertical position
    within the eye (hundredths of its width), and whether the eyes are
    open. None when MediaPipe is not available."""
    import cv2
    try:
        from opengp3.webcam import WebcamEyeDetector
        detector = WebcamEyeDetector()
    except Exception:  # noqa: BLE001
        return None
    wanted = np.zeros(len(ft), bool)
    for a, b in windows:
        wanted |= (ft >= a) & (ft <= b)
    rows = []
    capture = cv2.VideoCapture(str(video))
    for i in range(len(ft)):
        if not wanted[i]:
            if not capture.grab():
                break
            continue
        ok, frame = capture.read()
        if not ok:
            break
        eyes = detector.update(frame)
        if eyes:
            rows.append((ft[i], np.mean([e.feature_x for e in eyes]),
                         np.mean([e.feature_y for e in eyes]), 1.0))
        else:
            rows.append((ft[i], np.nan, np.nan, 0.0))
    capture.release()
    return np.array(rows) if rows else np.empty((0, 4))


def _jump_time(t: np.ndarray, v: np.ndarray, onset: float
               ) -> Optional[float]:
    """When ``v`` jumps from its level before ``onset`` to its level after:
    the first of two samples past halfway, from ``onset``. None when there
    is no clear jump (smaller than 5 times the noise before it)."""
    before = (t >= onset - 1.0) & (t <= onset - 0.1)
    after = (t >= onset + 0.9) & (t <= onset + 1.9)
    if before.sum() < 4 or after.sum() < 4:
        return None
    lo, hi = float(np.median(v[before])), float(np.median(v[after]))
    noise = float(np.median(np.abs(v[before] - lo))) * 1.48 + 1e-9
    if abs(hi - lo) < 5 * noise:
        return None
    s = (t >= onset - 0.2) & (t <= onset + 0.9)
    past = (v[s] - lo) / (hi - lo) > 0.5
    idx = np.flatnonzero(past[:-1] & past[1:])
    return float(t[s][idx[0]] - onset) if len(idx) else None


def _camera_lag(rec: Recording):
    """How far the camera's video lags the eye tracker: the eyes' jumps to
    each new cross, timed in the camera's video (the iris within the eye,
    by MediaPipe's face landmarks, as OpenGP3's webcam tracking finds it)
    and in the gaze, by the same rule. No marker is in between, and the
    gaze is on the EEG's timeline (checked against the EOG), so this
    places the camera on it too. Returns (median lag s, IQR s, jumps,
    share of frames with the eyes found), or None."""
    if rec.camera is None or rec.camera_t is None or rec.gaze is None:
        return None
    fixations = _fixations(rec)
    if len(fixations) < 5:
        return None
    windows = [(o - 1.2, o + 2.0) for o, _, _ in fixations]
    eyes = _camera_eyes(rec.camera, rec.camera_t, windows)
    if eyes is None or len(eyes) < 50:
        return None
    from scipy.signal import medfilt
    seen = eyes[:, 3] > 0
    found = float(np.mean(seen))
    if seen.sum() < 50:
        return None
    et = eyes[seen, 0]
    # The iris jitters by a frame or two: a 3-frame median steadies it.
    cam = [medfilt(eyes[seen, 1], 3), medfilt(eyes[seen, 2], 3)]
    ok = rec.gaze[:, 2] > 0
    gt, gaze = rec.gaze_t[ok], [rec.gaze[ok, 0], rec.gaze[ok, 1]]
    lags, previous = [], None
    for onset, x, y in fixations:
        if previous is not None:
            dx, dy = abs(x - previous[0]), abs(y - previous[1])
            # The direction the eyes moved most in (at least 15 % of the
            # screen): the iris moves a few pixels in the camera.
            axis = 0 if dx >= dy else 1
            if max(dx, dy) >= 0.15:
                c_ = _jump_time(et, cam[axis], onset)
                g_ = _jump_time(gt, gaze[axis], onset)
                if c_ is not None and g_ is not None:
                    lags.append(c_ - g_)
        previous = (x, y)
    if len(lags) < 5:
        return (None, None, len(lags), found)
    lags = np.array(lags)
    return (float(np.median(lags)),
            float(np.subtract(*np.percentile(lags, [75, 25]))),
            len(lags), found)


def video_setup(rec: Recording) -> Check:
    """Whether OBS's picture has the screen's shape: if not, OBS scales or
    crops the screen into its canvas."""
    c = Check("Screen recording setup", None, info=True)
    if rec.video is None:
        c.lines.append("No screen recording.")
        return c
    import cv2
    capture = cv2.VideoCapture(str(rec.video))
    w, h = capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(
        cv2.CAP_PROP_FRAME_HEIGHT)
    capture.release()
    c.lines.append(f"Video {w:.0f}×{h:.0f}" + (
        f", {len(rec.frame_t)} frames at "
        f"{1 / np.median(np.diff(rec.frame_t)):.1f} fps"
        if rec.frame_t is not None and len(rec.frame_t) > 1 else ""))
    screen = rec.screen_px
    if screen and h:
        ratio = (w / h) / (screen[0] / screen[1])
        if abs(ratio - 1) > 0.02:
            c.lines.append(f"The screen is {screen[0]}×{screen[1]}: OBS's "
                           f"canvas has another shape, so the screen is "
                           f"scaled or cropped into it. In OBS: Settings → "
                           f"Video → Base (Canvas) Resolution to the screen's "
                           f"shape, and the capture fitted to it (right-click "
                           f"→ Transform → Fit to screen).")
        else:
            c.lines.append(f"Same shape as the screen ({screen[0]}×"
                           f"{screen[1]}).")
    return c


def check_sync(rec: Recording, plt=None, out: Path = None) -> Check:
    c = Check("Synchronisation of EEG, markers, eyes, screen and camera", None)
    verdicts: list[bool] = []

    # Markers: each recorded once.
    if rec.markers:
        ok = rec.duplicates == 0
        verdicts.append(ok)
        c.lines.append(f"- {'✓' if ok else '✗'} each marker recorded once "
                       f"({rec.duplicates} recorded twice; pass: none). "
                       + ("" if ok else "The checks count each once; the "
                          "recording's events.tsv has them twice: a brAIn "
                          "dump problem to report."))

    # Markers: the same in eeg/ and beh/.
    if rec.markers and rec.beh_markers:
        beh = []
        for m in rec.beh_markers:
            if not beh or (m.text, round(m.onset, 4)) != \
                    (beh[-1].text, round(beh[-1].onset, 4)):
                beh.append(m)
        pairs = list(zip(rec.markers, beh))
        same = len(rec.markers) == len(beh) and all(
            a.text == b.text for a, b in pairs)
        diff = max(abs(a.onset - b.onset) for a, b in pairs) * 1000
        ok = same and diff <= 1.0
        verdicts.append(ok)
        c.lines.append(f"- {'✓' if ok else '✗'} markers in eeg/ and beh/ "
                       f"events.tsv: {len(rec.markers)} and "
                       f"{len(beh)}, onsets within "
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

    # Eyes vs markers: following the moving targets.
    pursuit = _pursuit_lags(rec) if rec.gaze is not None else []
    if pursuit:
        lag = float(np.median([p[0] for p in pursuit])) * 1000
        err = float(np.median([p[1] for p in pursuit]))
        ok = -50 <= lag <= 150
        verdicts.append(ok)
        c.lines.append(f"- {'✓' if ok else '✗'} the eyes follow the moving "
                       f"targets **{lag:.0f} ms** behind them (median of "
                       f"{len(pursuit)}; pass: -50 to 150 ms), "
                       f"{err:.1f}° from the target at that lag")
    elif rec.all("pursuit_start"):
        c.lines.append("- the moving targets: too little valid gaze to "
                       "measure")

    # EEG vs eyes: the EOG against the gaze, no markers in between.
    eog = _eog_lag(rec)
    if eog is not None:
        lag, r, axis = eog
        if abs(r) >= 0.3:
            ok = abs(lag) <= 0.04
            verdicts.append(ok)
            c.lines.append(f"- {'✓' if ok else '✗'} the EOG moves with the "
                           f"{axis} gaze, the gaze **{lag * 1000:+.0f} ms** "
                           f"after it (r = {r:+.2f}; pass: within 40 ms)")
        else:
            c.lines.append(f"- the EOG does not follow the gaze (best r = "
                           f"{r:+.2f}, {axis}): not measured. Check the EOG "
                           f"electrodes (beside the eyes) and their channel.")

    # Eyes and EEG vs markers: closing and opening the eyes.
    closed, opened = rec.first("rest_closed_start"), rec.first(
        "rest_closed_end")
    if rec.gaze is not None and closed and opened:
        lost = _first_run(rec.gaze_t, rec.gaze[:, 2] <= 0, closed.onset, 0.5)
        back = _first_run(rec.gaze_t, rec.gaze[:, 2] > 0, opened.onset, 0.5)
        ok = lost is not None and 0 <= lost <= 3
        verdicts.append(ok)
        c.lines.append(f"- {'✓' if ok else '✗'} gaze lost {_ms(lost)} after "
                       f"'eyes closed' (pass: 0-3 s); back {_ms(back)} "
                       f"after the tone (information: where the eyes go "
                       f"after opening is up to the participant)")
    if rec.eeg is not None and closed and rec.first("rest_open_start"):
        rise = _alpha_rise(rec, rec.first("rest_open_start").onset,
                           closed.onset, opened.onset if opened else None)
        c.lines.append(f"- EEG alpha rises {_ms(rise)} after 'eyes closed' "
                       f"(information: usually within 1-3 s, but how fast "
                       f"varies between people)")

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
        # Corrected by the Screen Capture's Delay, a frame comes when it was on
        # screen: 0, give or take a frame (17 ms at 60 fps).
        ok = -20 <= med <= 200 and iqr < 50
        verdicts.append(ok)
        c.lines.append(f"- {'✓' if ok else '✗'} faces appear in the screen "
                       f"recording **{med:.0f} ms** after their markers "
                       f"(IQR {iqr:.0f} ms, {len(v)} faces; pass: -20 to 200 "
                       f"ms, IQR < 50 ms)")
        if med > 50 and iqr < 50:
            c.lines.append(f"  The delay is steady: it is OBS's capture and "
                           f"output. Set the Screen Capture's Delay to "
                           f"{med:.0f} ms in brAIn dump, and its frames are "
                           f"placed when they were on screen.")
        if plt is not None:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(v, bins=20)
            if latencies:
                ax.hist(np.array(latencies) * 1000, bins=20, alpha=0.6)
            ax.set_xlabel("ms after the marker")
            ax.legend(["face on screen (video)"] +
                      (["gaze on the new cross"] if latencies else []))
            c.figure = _save(fig, out, "sync.png", plt)
    # Screen vs markers: the flashes.
    flashes, n_flashes = _flash_latencies(rec, rec.video, rec.frame_t) \
        if rec.video is not None else ([], 0)
    if n_flashes and len(flashes) < n_flashes / 2:
        verdicts.append(False)
        c.lines.append(f"- ✗ flashes found in the screen recording for only "
                       f"{len(flashes)} of {n_flashes} (pass: at least half): "
                       f"is OBS capturing the stimulus screen?")
    elif flashes:
        v = np.array(flashes) * 1000
        med = float(np.median(v))
        iqr = float(np.subtract(*np.percentile(v, [75, 25])))
        ok = -20 <= med <= 200 and iqr < 50
        verdicts.append(ok)
        c.lines.append(f"- {'✓' if ok else '✗'} flashes appear in the screen "
                       f"recording **{med:.0f} ms** after their markers (IQR "
                       f"{iqr:.0f} ms, {len(v)}/{n_flashes}; pass: -20 to 200 "
                       f"ms, IQR < 50 ms)")

    # Camera vs eye tracker: the eyes it films against the gaze.
    if rec.camera is not None:
        measured = len(verdicts)
        camera = _camera_lag(rec)
        if camera is None:
            c.lines.append("- the camera's eyes: not measured (MediaPipe "
                           "missing, or too few frames with a face)")
        elif camera[0] is None:
            c.lines.append(f"- the eyes' jumps found in the camera's video: "
                           f"{camera[2]} (eyes found in {camera[3] * 100:.0f}"
                           f" % of frames): too few to measure. The camera "
                           f"should see the participant's face, close enough "
                           f"for the eyes, and lit.")
        else:
            lag, iqr, n, found = camera
            ok = abs(lag) <= 0.06 and iqr < 0.1
            verdicts.append(ok)
            c.lines.append(f"- {'✓' if ok else '✗'} the camera films the eyes' "
                           f"jumps **{lag * 1000:+.0f} ms** after the eye "
                           f"tracker records them (IQR {iqr * 1000:.0f} ms, "
                           f"{n} jumps, eyes found in {found * 100:.0f} % of "
                           f"frames; pass: within 60 ms, IQR < 100 ms)")
            if abs(lag) > 0.02 and iqr < 0.1:
                c.lines.append(f"  Steady: add {lag * 1000:.0f} ms to the "
                               f"Video Stream's Delay in brAIn dump, and its "
                               f"frames are placed when they were taken.")
            elif iqr >= 0.1:
                c.lines.append("  Uneven: the camera's frames arrive with a "
                               "varying delay (a camera that buffers frames, "
                               "or a busy computer). Try a lower resolution "
                               "or frame rate, or another camera.")
        # The flashes, as their light on the participant: only when seen.
        cam_flashes, n_cam = _flash_latencies(rec, rec.camera, rec.camera_t)
        if n_cam and len(cam_flashes) >= n_cam / 2:
            v = np.array(cam_flashes) * 1000
            med = float(np.median(v))
            iqr = float(np.subtract(*np.percentile(v, [75, 25])))
            ok = -60 <= med <= 60 and iqr < 70
            verdicts.append(ok)
            c.lines.append(f"- {'✓' if ok else '✗'} the flashes' light shows in "
                           f"the camera **{med:.0f} ms** after their markers "
                           f"(IQR {iqr:.0f} ms, {len(v)}/{n_cam}; pass: within "
                           f"60 ms, IQR < 70 ms)")
        elif n_cam:
            c.lines.append(f"- the flashes' light shows in the camera for "
                           f"{len(cam_flashes)} of {n_cam} (information: a "
                           f"camera on the participant sees them only in a "
                           f"dim room)")
        if len(verdicts) == measured:
            verdicts.append(False)
            c.lines.append("- ✗ the camera's timing could not be measured: "
                           "neither the participant's eyes nor the flashes' "
                           "light were found in its video")

    # The camera covers the procedure too.
    if rec.markers and rec.camera_t is not None:
        start, end = rec.markers[0].onset, rec.markers[-1].onset
        a_, b_ = rec.camera_t[0], rec.camera_t[-1]
        ok = a_ <= start + 0.5 and b_ >= end - 0.5
        verdicts.append(ok)
        c.lines.append(f"- {'✓' if ok else '✗'} camera covers the procedure "
                       f"({a_:.1f}-{b_:.1f} s)")
        gaps = np.diff(rec.camera_t)
        if len(gaps):
            nominal = float(np.median(gaps))
            dropped = int(np.sum(np.round(gaps / nominal) - 1))
            c.lines.append(f"  {len(rec.camera_t)} frames at "
                           f"{1 / nominal:.1f} fps, {max(0, dropped)} "
                           f"missing (information)")

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
    checks = [check_alpha(rec, plt, out), check_eeg_quality(rec),
              check_gaze(rec, plt, out), check_n170(rec, plt, out),
              check_sync(rec, plt, out), video_setup(rec)]
    if out is not None:
        _write_report(out / "report.md", rec, checks)
    return checks


def _verdict(check) -> str:
    if check.info:
        return "INFO"
    return {True: "PASS", False: "FAIL", None: "NOT CHECKED"}[check.passed]


def _write_report(path: Path, rec: Recording, checks: list[Check]) -> None:
    lines = [f"# brAIn dump validation: {rec.stem}", ""]
    lines += [f"> {note}" for note in rec.notes]
    lines += ["", "| Check | Result |", "|---|---|"]
    lines += [f"| {c.name} | **{_verdict(c)}** |" for c in checks
              if not c.info]
    for c in checks:
        lines += ["", f"## {c.name}: {_verdict(c)}", ""] + c.lines
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
        print(f"\n[{_verdict(c)}] {c.name}")
        for line in c.lines:
            print("   " + line.replace("**", ""))
    print(f"\nReport: {out / 'report.md'}")
    return 0 if all(c.passed for c in checks if not c.info) else 1


if __name__ == "__main__":
    sys.exit(main())
