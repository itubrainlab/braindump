"""Measure how late OBS's NDI frames are: the value for the OBS input's
Delay setting in brAIn dump.

    python validation/screen_delay.py [--psychopy PATH] [--flashes 20]

PsychoPy shows the screen dark and light in turn, once a second, sending
an LSL marker stamped at each flip. Meanwhile this receives OBS's NDI
stream as brAIn dump does (frames placed by their sender time stamps,
with no delay taken off) and finds the first frame showing each change.
The median time from flip to frame is OBS's delay: capture, rendering and
NDI.

Run it with brAIn dump's Python; PsychoPy (the standalone app's Python by
default) does the flashing. Keep OBS capturing the screen the flashes are
on, and don't touch anything while it runs (about half a minute).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

STREAM = "brAIndumpScreenDelay"
PSYCHOPY_APP = "/Applications/PsychoPy.app/Contents"

#: The flasher, run by PsychoPy's Python.
FLASHER = r'''
import sys
from psychopy import core, visual, event
from pylsl import StreamInfo, StreamOutlet, local_clock
n, screen = int(sys.argv[1]), int(sys.argv[2])
outlet = StreamOutlet(StreamInfo("%s", "Markers", 1, 0, "string",
                                 "braindump-screen-delay"))
outlet.wait_for_consumers(20)
win = visual.Window(fullscr=True, screen=screen, color=(-0.6,) * 3,
                    units="norm", allowGUI=False, waitBlanking=True)
win.mouseVisible = False
rate = win.getActualFrameRate(nIdentical=20, nMaxFrames=120) or 60.0
for _ in range(int(rate * 2)):
    win.flip()
for k in range(n):
    level = 0.6 if k %% 2 == 0 else -0.6
    win.color = (level,) * 3
    win.callOnFlip(lambda k=k, level=level: outlet.push_sample(
        ["%%s %%d" %% ("light" if level > 0 else "dark", k)], local_clock()))
    for _ in range(int(round(rate))):
        win.flip()
        if event.getKeys(["escape"]):
            break
outlet.push_sample(["end"], local_clock())
core.wait(0.5)
win.close()
''' % STREAM


def psychopy_python(path: str) -> tuple[list[str], dict]:
    env = dict(os.environ)
    if path:
        return [path], env
    app = Path(PSYCHOPY_APP)
    if (app / "MacOS" / "python").exists():
        env["PYTHONHOME"] = str(app / "Resources")
        return [str(app / "MacOS" / "python")], env
    sys.exit("Give PsychoPy's Python with --psychopy.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--psychopy", default="",
                   help="PsychoPy's python (default: PsychoPy.app's)")
    p.add_argument("--flashes", type=int, default=20)
    p.add_argument("--screen", type=int, default=0)
    p.add_argument("--source", default="",
                   help="the NDI source (default: the first OBS one)")
    args = p.parse_args(argv)

    import pylsl
    from braindump import clock
    from braindump.devices.video import NdiSource

    source = NdiSource(args.source, 0.0)
    source.open()
    print(f"NDI: {source.info.get('name')}")
    frames: list[tuple[float, float]] = []
    running = True

    def receive():
        while running:
            frame = source.read()
            if frame is not None:
                small = frame.image[::8, ::8]
                frames.append((frame.time, float(small.mean())))
    reader = threading.Thread(target=receive, daemon=True)
    reader.start()

    script = Path(tempfile.mkdtemp()) / "flasher.py"
    script.write_text(FLASHER)
    command, env = psychopy_python(args.psychopy)
    flasher = subprocess.Popen(command + [str(script), str(args.flashes),
                                          str(args.screen)], env=env,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    found = pylsl.resolve_byprop("name", STREAM, 1, 30)
    if not found:
        sys.exit("The flasher did not start (is PsychoPy's Python right?).")
    inlet = pylsl.StreamInlet(found[0])
    markers: list[tuple[float, str]] = []
    deadline = time.monotonic() + args.flashes + 40
    while time.monotonic() < deadline:
        sample, stamp = inlet.pull_sample(timeout=1.0)
        if sample is None:
            continue
        # LSL's clock onto brAIn dump's: both steady, so an offset.
        offset = clock.now() - pylsl.local_clock()
        stamp += inlet.time_correction()
        markers.append((stamp + offset, sample[0]))
        if sample[0] == "end":
            break
    flasher.wait(timeout=30)
    time.sleep(1.0)
    running = False
    reader.join(timeout=2)
    source.close()

    times = np.array([t for t, _ in frames])
    levels = np.array([v for _, v in frames])
    delays = []
    for onset, text in markers:
        if text == "end":
            continue
        light = text.startswith("light")
        before = levels[(times > onset - 0.6) & (times < onset - 0.1)]
        after = np.flatnonzero((times >= onset - 0.2) & (times <= onset + 0.9))
        if len(before) < 3 or not len(after):
            continue
        base = float(np.median(before))
        other = levels[after].max() if light else levels[after].min()
        half = (base + other) / 2
        changed = after[(levels[after] > half) if light
                        else (levels[after] < half)]
        if len(changed) and abs(other - base) > 20:
            delays.append(times[changed[0]] - onset)
    if len(delays) < args.flashes // 2:
        print(f"Only {len(delays)} of {args.flashes} changes were seen in "
              f"the NDI stream: is OBS capturing this screen?")
        return 1
    d = np.array(delays) * 1000
    q1, q3 = np.percentile(d, [25, 75])
    print(f"{len(d)} changes: median {np.median(d):.0f} ms "
          f"(IQR {q1:.0f}-{q3:.0f} ms, range {d.min():.0f}-{d.max():.0f})")
    print(f"Fps of the stream: {1 / np.median(np.diff(times)):.1f}")
    print(f"\nSet the OBS input's Delay in brAIn dump to "
          f"{np.median(d):.0f} ms.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
