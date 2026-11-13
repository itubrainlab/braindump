"""brAIn dump validation procedure (PsychoPy).

Checks that brAIn dump records the lab's signals correctly and in step:
EEG (g.HIamp or Unicorn), gaze (Gazepoint GP3 or GP3 HD), the screen (OBS
over NDI), a camera, and this procedure's own events, sent as Lab
Streaming Layer markers.

    1. Instructions
    2. 10 flashes of the whole screen (for the screen recording and the
       camera)
    3. 30 s resting, eyes open
    4. 30 s resting, eyes closed (a tone says when to open them)
    5. Eye-tracking calibration (drawn here, run by the GP3's server)
    6. 2 moving targets to follow with the eyes (10 s each)
    7. 30 faces in random order: a fixation cross at a random place (2 s),
       the face where the cross was (1 s), a blank screen (1 s)
    8. 2 more moving targets, then 10 more flashes
    9. Debriefing

Every step is an LSL marker (stream "brAIndumpValidation", type
"Markers"), stamped at the screen flip that shows it. Run it with
PsychoPy (Coder: open and Run; or from a terminal with PsychoPy's Python):

    python face_validation.py --faces /path/to/faces --participant 01

``--dry-run`` runs a short version in a window, for trying it out.
After the session, ``validation/analyse.py`` checks the recording.
See validation/README.md for the whole procedure.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import re
import socket
import sys
from datetime import datetime
from pathlib import Path

STREAM_NAME = "brAIndumpValidation"
STREAM_TYPE = "Markers"
SOURCE_ID = "braindump-validation-markers"

#: Cross, face and blank durations (s), and the area the cross may be in
#: (fraction of the screen from each edge kept free).
FIXATION_S, FACE_S, BLANK_S = 2.0, 1.0, 1.0
MARGIN = 0.12
#: Face height, as a fraction of the screen's height.
FACE_HEIGHT = 0.30
#: The flashes: white for FLASH_ON_S, then black for FLASH_OFF_S.
FLASH_ON_S, FLASH_OFF_S = 0.2, 0.8
#: The moving targets: a Lissajous path around the centre, in screen
#: fractions; slow enough to follow smoothly (under 20 degrees/s).
PURSUIT_S, PURSUIT_AX, PURSUIT_AY = 10.0, 0.30, 0.25
PURSUIT_FX, PURSUIT_FY = 0.20, 0.30
IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--faces", default=os.environ.get("BRAINDUMP_FACES", ""),
                   help="folder of face images (at least --trials of them)")
    p.add_argument("--participant", default="",
                   help="participant label, for the log file")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--flashes", type=int, default=10,
                   help="flashes at the start and at the end (0: none)")
    p.add_argument("--pursuits", type=int, default=2,
                   help="moving targets before and after the faces (0: none)")
    p.add_argument("--rest", type=float, default=30.0,
                   help="seconds of each resting period")
    p.add_argument("--gp3-host", default="127.0.0.1",
                   help="the GP3's Open Gaze API server (opengp3 server)")
    p.add_argument("--gp3-port", type=int, default=4242)
    p.add_argument("--points", type=int, default=9, choices=(5, 9),
                   help="calibration points")
    p.add_argument("--screen", type=int, default=0,
                   help="the display the GP3 is under")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="short timings, in a window: for trying it out")
    p.add_argument("--no-wait", action="store_true",
                   help="do not wait for brAIn dump to receive the markers")
    p.add_argument("--auto", action="store_true",
                   help="go on by itself after each screen, without keys: "
                        "for testing the recording without a participant")
    return p.parse_args(argv)


# -- markers ----------------------------------------------------------------------------

class Markers:
    """The LSL marker stream, and a log of every marker sent."""

    def __init__(self):
        from pylsl import StreamInfo, StreamOutlet, local_clock
        info = StreamInfo(STREAM_NAME, STREAM_TYPE, 1, 0, "string",
                          SOURCE_ID)
        info.desc().append_child_value("manufacturer", "brAIn dump")
        info.desc().append_child_value("procedure", "face validation")
        self.outlet = StreamOutlet(info)
        self.clock = local_clock
        self.sent: list[tuple[float, str]] = []

    def send(self, text: str, stamp: float | None = None) -> None:
        stamp = self.clock() if stamp is None else stamp
        self.outlet.push_sample([text], stamp)
        self.sent.append((stamp, text))

    def on_flip(self, win, text: str) -> None:
        """Send ``text`` stamped at the next screen flip."""
        win.callOnFlip(lambda: self.send(text, self.clock()))

    def wait_for_brain_dump(self, timeout: float) -> bool:
        return self.outlet.wait_for_consumers(timeout)


# -- the GP3's calibration, over the Open Gaze API ------------------------------------------

_TAG = re.compile(r"<\s*(\w+)(.*?)/?\s*>")
_ATTR = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"([^"]*)"')


class OpenGaze:
    """Just enough of the Open Gaze API (OpenGP3's server, or Gazepoint
    Control) to run a calibration whose targets this window draws."""

    def __init__(self, host: str, port: int):
        self.sock = socket.create_connection((host, port), timeout=3.0)
        self.sock.setblocking(False)
        self._buf = ""

    def send(self, verb: str, ident: str, **attrs) -> None:
        text = "".join(f' {k}="{v}"' for k, v in attrs.items())
        self.sock.sendall(f'<{verb} ID="{ident}"{text} />\r\n'.encode())

    def messages(self):
        """Every complete message received so far: (tag, attributes)."""
        try:
            while True:
                data = self.sock.recv(65536)
                if not data:
                    break
                self._buf += data.decode("utf-8", "replace")
        except (BlockingIOError, InterruptedError):
            pass
        out = []
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            m = _TAG.match(line.strip())
            if m:
                out.append((m.group(1), dict(_ATTR.findall(m.group(2)))))
        return out

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


NINE = [(0.5, 0.5), (0.15, 0.15), (0.5, 0.15), (0.85, 0.15), (0.15, 0.5),
        (0.85, 0.5), (0.15, 0.85), (0.5, 0.85), (0.85, 0.85)]


def calibrate(win, markers: Markers, args, visual, event, core) -> str:
    """Run the GP3's calibration, drawing its targets here. Returns a
    summary, or raises OSError when the server cannot be reached."""
    gaze = OpenGaze(args.gp3_host, args.gp3_port)
    try:
        if args.points == 5:
            gaze.send("SET", "CALIBRATE_RESET")
        else:
            gaze.send("SET", "CALIBRATE_CLEAR")
            for x, y in NINE:
                gaze.send("SET", "CALIBRATE_ADDPOINT", X=f"{x:.3f}",
                          Y=f"{y:.3f}")
        gaze.send("SET", "CALIBRATE_DELAY", VALUE="0.6")
        gaze.send("SET", "CALIBRATE_TIMEOUT", VALUE="1.2")
        gaze.send("SET", "CALIBRATE_SHOW", STATE="0")
        gaze.send("SET", "CALIBRATE_START", STATE="1")
        markers.send("calibration_start")
        outer = visual.Circle(win, radius=0.04, fillColor="white",
                              lineColor=None, units="height")
        inner = visual.Circle(win, radius=0.006, fillColor="red",
                              lineColor=None, units="height")
        target, started, result = None, 0.0, None
        deadline = core.getTime() + 60
        while result is None and core.getTime() < deadline:
            for tag, a in gaze.messages():
                if tag != "CAL":
                    continue
                if a.get("ID") == "CALIB_START_PT":
                    fx, fy = float(a.get("CALX", 0.5)), float(a.get("CALY",
                                                                    0.5))
                    target, started = _to_height(win, fx, fy), core.getTime()
                    markers.send(f"calibration_point n={a.get('PT')} "
                                 f"x={fx:.4f} y={fy:.4f}")
                elif a.get("ID") == "CALIB_RESULT":
                    result = a
            if target is not None:
                shrink = min(1.0, (core.getTime() - started) / 1.8)
                outer.radius = 0.04 - 0.025 * shrink
                outer.pos = inner.pos = target
                outer.draw()
                inner.draw()
            win.flip()
            if event.getKeys(["escape"]):
                gaze.send("SET", "CALIBRATE_START", STATE="0")
                raise Escaped
        if result is None:
            raise OSError("the server sent no calibration result")
        valid = sum(1 for k, v in result.items()
                    if re.fullmatch(r"[LR]V\d+", k) and v == "1")
        points = sum(1 for k in result if re.fullmatch(r"CALX\d+", k))
        summary = f"calibration_result points={points} valid_eye_points={valid}"
        markers.send(summary)
        markers.send("calibration_end")
        return summary
    finally:
        gaze.close()


def _to_height(win, fx: float, fy: float):
    """Screen fractions (from the top left) in "height" units."""
    w, h = win.size
    aspect = w / h
    return ((fx - 0.5) * aspect, 0.5 - fy)


# -- the procedure ---------------------------------------------------------------------------

def faces_from(folder: str, n: int, rng: random.Random) -> list[Path]:
    if not folder:
        sys.exit("Give the face images with --faces (see validation/README.md).")
    images = sorted(p for p in Path(folder).expanduser().iterdir()
                    if p.suffix.lower() in IMAGE_TYPES)
    if not images:
        sys.exit(f"No images in {folder}.")
    if len(images) >= n:
        return rng.sample(images, n)
    print(f"Only {len(images)} images: some are shown more than once.")
    return [rng.choice(images) for _ in range(n)]


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        args.rest, args.trials = min(args.rest, 4.0), min(args.trials, 6)
        args.flashes, args.pursuits = min(args.flashes, 3), \
            min(args.pursuits, 1)
    rng = random.Random(args.seed)
    faces = faces_from(args.faces, args.trials, rng)

    from psychopy import core, event, logging, monitors, visual
    from psychopy import sound
    logging.console.setLevel(logging.WARNING)

    markers = Markers()
    win = visual.Window(fullscr=not args.dry_run, screen=args.screen,
                        size=(1280, 720), units="height", color=(0, 0, 0),
                        colorSpace="rgb", allowGUI=args.dry_run,
                        waitBlanking=True, monitor=monitors.Monitor("default"))
    win.mouseVisible = False
    rate = win.getActualFrameRate(nIdentical=20, nMaxFrames=120) or 60.0
    frames = lambda seconds: max(1, int(round(seconds * rate)))
    text = visual.TextStim(win, height=0.035, wrapWidth=1.2, color="white")
    cross = visual.ShapeStim(win, vertices="cross", size=0.05,
                             fillColor="white", lineColor=None)
    dot = visual.Circle(win, radius=0.006, fillColor="white", lineColor=None)
    aspect = win.size[0] / win.size[1]
    white = visual.Rect(win, width=aspect * 1.1, height=1.1,
                        fillColor="white", lineColor=None)
    black = visual.Rect(win, width=aspect * 1.1, height=1.1,
                        fillColor="black", lineColor=None)
    target = visual.Circle(win, radius=0.012, fillColor="white",
                           lineColor=None)
    centre = visual.Circle(win, radius=0.003, fillColor="red",
                           lineColor=None)
    tone = sound.Sound("A", secs=0.4)

    def show(message: str, keys=("space",)) -> None:
        text.text = message
        text.draw()
        win.flip()
        event.clearEvents()
        if args.auto:
            core.wait(1.0)
            return
        pressed = event.waitKeys(keyList=list(keys) + ["escape"])
        if "escape" in pressed:
            raise Escaped

    def hold(seconds: float, stim=None, marker: str | None = None) -> None:
        """Show ``stim`` for ``seconds``, counted in frames, with
        ``marker`` sent at its first flip."""
        for i in range(frames(seconds)):
            if stim is not None:
                stim.draw()
            if i == 0 and marker:
                markers.on_flip(win, marker)
            win.flip()
            if event.getKeys(["escape"]):
                raise Escaped

    def flashes(block: str) -> None:
        """The whole screen white for a moment, ``args.flashes`` times: a
        sharp change for the screen recording and the camera to show."""
        if args.flashes <= 0:
            return
        hold(1.0, black, f"flashes_start block={block}")
        for n in range(1, args.flashes + 1):
            hold(FLASH_ON_S, white, f"flash n={n} block={block}")
            hold(FLASH_OFF_S, black)
        markers.on_flip(win, f"flashes_end block={block}")
        win.flip()

    pursuit_count = [0]

    def pursuits(how_many: int) -> None:
        """Targets moving smoothly along a known path, for the eyes to
        follow: the gaze then traces the path, and its delay against the
        markers is measured over the whole movement, not only at a jump."""
        import math
        for _ in range(how_many):
            pursuit_count[0] += 1
            k = pursuit_count[0]
            phase = rng.uniform(0, 2 * math.pi)
            n = frames(PURSUIT_S)

            def at(i):
                t = i / rate
                return (0.5 + PURSUIT_AX * math.sin(2 * math.pi * PURSUIT_FX * t),
                        0.5 + PURSUIT_AY * math.sin(2 * math.pi * PURSUIT_FY * t
                                                    + phase))
            x0, y0 = at(0)
            target.pos = centre.pos = ((x0 - 0.5) * aspect, 0.5 - y0)
            for i in range(frames(1.0)):
                target.draw()
                centre.draw()
                if i == 0:
                    markers.on_flip(win, f"pursuit_ready trial={k}")
                win.flip()
            for i in range(n):
                x, y = at(i)
                target.pos = centre.pos = ((x - 0.5) * aspect, 0.5 - y)
                target.draw()
                centre.draw()
                if i == 0:
                    markers.on_flip(
                        win, f"pursuit_start trial={k} duration={n / rate:.4f} "
                        f"ax={PURSUIT_AX} ay={PURSUIT_AY} fx={PURSUIT_FX} "
                        f"fy={PURSUIT_FY} phase={phase:.5f} rate={rate:.4f}")
                win.flip()
                if event.getKeys(["escape"]):
                    raise Escaped
            markers.on_flip(win, f"pursuit_end trial={k}")
            win.flip()
            core.wait(0.5)

    log_rows = []
    try:
        # 0. brAIn dump must be receiving the markers before anything.
        if not args.no_wait:
            text.text = ("Waiting for brAIn dump to receive the marker stream "
                         f"'{STREAM_NAME}'...\n\nIn brAIn dump: connect the "
                         "devices (the LSL streams device listens for it), "
                         "then press Record.\n\n(Space to go on anyway.)")
            while not markers.wait_for_brain_dump(0.2):
                text.draw()
                win.flip()
                if event.getKeys(["space"]):
                    break
                if event.getKeys(["escape"]):
                    raise Escaped
        markers.send(f"procedure_start trials={args.trials} "
                     f"refresh={rate:.2f} participant={args.participant}")

        # 1. Instructions
        markers.on_flip(win, "instructions")
        show("Welcome, and thank you for taking part.\n\n"
             "This short session checks the lab's recording equipment.\n\n"
             "The screen will flash a few times. Then you will rest for "
             f"{args.rest:.0f} seconds with your eyes open, looking at a dot, "
             f"and {args.rest:.0f} seconds with your eyes closed. Then the "
             "eye tracker is calibrated: follow the dot with your eyes. "
             "Follow the moving dots with your eyes, too. Then you will see "
             "faces: look at each cross, and keep looking there when the "
             "face appears.\n\n"
             "Please sit still and relax your face and jaw.\n\n"
             "Experimenter: make sure brAIn dump is recording.\n\n"
             "Press space to begin.")

        # 2. Flashes
        if args.flashes > 0:
            show("First, the screen flashes a few times. Just look at the "
                 "screen.\n\nPress space to start.")
            flashes("start")

        # 3. Resting, eyes open
        show(f"Rest with your eyes OPEN for {args.rest:.0f} seconds, looking "
             "at the dot in the centre.\n\nPress space to start.")
        hold(args.rest, dot, "rest_open_start")
        markers.on_flip(win, "rest_open_end")
        win.flip()

        # 4. Resting, eyes closed
        show(f"Now rest with your eyes CLOSED for {args.rest:.0f} seconds.\n\n"
             "Close your eyes when you press space, and keep them closed "
             "until you hear a tone.\n\nPress space to start.")
        hold(args.rest, None, "rest_closed_start")
        win.callOnFlip(tone.play)
        markers.on_flip(win, "rest_closed_end")
        win.flip()
        core.wait(0.5)

        # 5. Eye-tracking calibration
        show("Open your eyes.\n\nNext, the eye tracker is calibrated: follow "
             "the dot with your eyes until it stops.\n\nPress space to start.")
        try:
            summary = calibrate(win, markers, args, visual, event, core)
            show("Calibration done.\n\nPress space to continue.")
        except OSError as error:
            markers.send(f"calibration_manual reason={type(error).__name__}")
            show("The eye tracker's server could not be reached "
                 f"({args.gp3_host}:{args.gp3_port}).\n\nExperimenter: "
                 "calibrate the GP3 in brAIn dump now (GP3 page, Calibrate), "
                 "then press space here.")
            markers.send("calibration_end")

        # 6. Moving targets
        if args.pursuits > 0:
            show("Now follow the moving dot with your eyes, as closely as "
                 "you can, until it stops.\n\nPress space to start.")
            pursuits(args.pursuits)

        # 7. Faces
        show(f"Now {args.trials} faces.\n\nEach time, look at the cross where "
             "it appears, and keep looking there when the face comes.\n\n"
             "Press space to start.")
        # Every face is loaded before the first trial: loading one between
        # screens would hold the screen before it for an extra frame.
        images = []
        for face in faces:
            image = visual.ImageStim(win, image=str(face))
            w, h = image.size
            image.size = (FACE_HEIGHT * w / h, FACE_HEIGHT)
            images.append(image)
        for trial, (face, image) in enumerate(zip(faces, images), start=1):
            fx = rng.uniform(MARGIN, 1 - MARGIN)
            fy = rng.uniform(MARGIN, 1 - MARGIN)
            pos = ((fx - 0.5) * aspect, 0.5 - fy)
            cross.pos = image.pos = pos
            hold(FIXATION_S, cross,
                 f"fixation trial={trial} x={fx:.4f} y={fy:.4f}")
            hold(FACE_S, image, f"face trial={trial} image={face.name} "
                                f"x={fx:.4f} y={fy:.4f}")
            hold(BLANK_S, None, f"blank trial={trial}")
            log_rows.append({"trial": trial, "image": face.name,
                             "x": f"{fx:.4f}", "y": f"{fy:.4f}"})

        # 8. Moving targets and flashes again: the timing at the end
        if args.pursuits > 0:
            show("Follow the moving dot with your eyes again.\n\n"
                 "Press space to start.")
            pursuits(args.pursuits)
        if args.flashes > 0:
            show("The screen flashes once more. Just look at the screen."
                 "\n\nPress space to start.")
            flashes("end")

        # 9. Debriefing
        markers.on_flip(win, "debriefing")
        show("That is all, thank you!\n\nThis session checked that the EEG, "
             "the eye tracker, the screen recording, the camera and the "
             "experiment's events are recorded correctly and in time with "
             "each other.\n\n"
             "Experimenter: stop recording in brAIn dump, then press space.")
        markers.send("procedure_end")
        status = 0
    except Escaped:
        markers.send("procedure_aborted reason=escape")
        status = 1
    except KeyboardInterrupt:
        markers.send("procedure_aborted reason=interrupt")
        status = 1
    finally:
        _save_log(args, markers, log_rows)
        win.close()
    return status


class Escaped(KeyboardInterrupt):
    """The participant or experimenter pressed Escape."""


def _save_log(args, markers: Markers, rows) -> None:
    """The markers as sent, with their LSL times, and the trials: a check
    on what brAIn dump received."""
    folder = Path(__file__).resolve().parent / "data"
    folder.mkdir(exist_ok=True)
    stem = f"{args.participant or 'anonymous'}_{datetime.now():%Y%m%d_%H%M%S}"
    with open(folder / f"{stem}_markers.csv", "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["lsl_time", "marker"])
        out.writerows((f"{t:.6f}", m) for t, m in markers.sent)
    if rows:
        with open(folder / f"{stem}_trials.csv", "w", newline="") as f:
            out = csv.DictWriter(f, fieldnames=list(rows[0]))
            out.writeheader()
            out.writerows(rows)
    print(f"Log: {folder / stem}_markers.csv")


if __name__ == "__main__":
    sys.exit(main())
