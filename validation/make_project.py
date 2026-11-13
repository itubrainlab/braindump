"""Make the brAIn stream project the validation procedure records with.

The lab (two computers, see validation/LAB.md), on the recording computer:

    python validation/make_project.py "~/brAIn stream/lab-validation" --lab \\
        --stimulus-host 192.168.1.20 --electrodes validation/cap64.txt

The same as::

    --eeg hiamp --channels 64 --rate 512 --channel-type 64=EOG \\
    --bandpass 0.01-100 --notch 50 --gp3-host <stimulus> --webcam

A single computer with a Unicorn (the first sessions)::

    python validation/make_project.py "~/brAIn stream/validation" \\
        --unicorn UN-2024.10.18 --gp3-host 127.0.0.1

Devices: the EEG (g.HIamp or Unicorn), the GP3 (or GP3 HD) read over the
Open Gaze API (it runs under ``opengp3 server`` on the stimulus computer,
so that the procedure can draw its calibration), the procedure's LSL
markers, OBS's screen over NDI, and, with ``--webcam``, a camera on this
computer. The g.HIamp's amplifier filters are chosen by their cut-offs
from those it offers, so it must be connected (or ``--simulate``).
``--simulate`` makes every device simulated, to try the whole chain
without hardware (the analysis's physiological checks then fail, as a
simulator has no eyes or faces).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from brainstream.devices.gp3 import GP3Config
from brainstream.devices.lsl import LslConfig
from brainstream.devices.unicorn import UnicornConfig
from brainstream.devices.video import ObsConfig, WebcamConfig
from brainstream.project import Project, remember

TASK = "facevalidation"
MARKERS = "brAInstreamValidation"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("folder", help="where the project goes")
    p.add_argument("--name", default="brAIn stream validation")
    p.add_argument("--lab", action="store_true",
                   help="the lab's two-computer setup: g.HIamp (64 channels, "
                        "64 bipolar EOG, 0.01-100 Hz, 50 Hz notch), the GP3 "
                        "HD and OBS on the stimulus computer, a camera here")
    p.add_argument("--stimulus-host", default="",
                   help="the stimulus computer (runs opengp3 server, OBS "
                        "and PsychoPy); sets --gp3-host")
    eeg = p.add_argument_group("EEG")
    eeg.add_argument("--eeg", choices=("unicorn", "hiamp"), default=None)
    eeg.add_argument("--unicorn", default="", help="the Unicorn's serial")
    eeg.add_argument("--hiamp", default="", help="the g.HIamp's serial "
                     "(empty: the one connected)")
    eeg.add_argument("--channels", type=int, default=64)
    eeg.add_argument("--rate", type=int, default=512,
                     help="the g.HIamp's sampling rate")
    eeg.add_argument("--channel-type", action="append", default=None,
                     metavar="N=TYPE",
                     help="a channel that is not EEG, e.g. 64=EOG (repeat)")
    eeg.add_argument("--bandpass", default="",
                     help="the amplifier's band-pass, low-high in Hz, e.g. "
                          "0.01-100")
    eeg.add_argument("--notch", type=float, default=0.0,
                     help="the amplifier's notch, in Hz (50 in Europe)")
    eeg.add_argument("--electrodes", type=Path, default=None,
                     help="a text file of the channel names in the "
                          "amplifier's order (one per line, or commas)")
    eeg.add_argument("--reference", default="")
    eeg.add_argument("--ground", default="")
    gaze = p.add_argument_group("eye tracker")
    gaze.add_argument("--gp3-host", default="127.0.0.1",
                      help="the computer running opengp3 server")
    gaze.add_argument("--gp3-port", type=int, default=4242)
    gaze.add_argument("--screen-width-cm", type=float, default=0.0,
                      help="the stimulus screen's width (for accuracy in "
                           "degrees); 0 asks the GP3's server")
    gaze.add_argument("--distance-cm", type=float, default=65.0)
    video = p.add_argument_group("video")
    video.add_argument("--ndi-source", default="",
                       help="OBS's NDI source name (empty: the first OBS "
                            "one); with OBS on the stimulus computer, its "
                            "name starts with that computer's")
    video.add_argument("--webcam", nargs="?", const="first", default="",
                       help="also record a camera on this computer: its "
                            "name as the system lists it, or nothing for "
                            "the first")
    p.add_argument("--simulate", action="store_true")
    args = p.parse_args(argv)
    if args.lab:
        args.eeg = args.eeg or "hiamp"
        args.channel_type = args.channel_type or ["64=EOG"]
        args.bandpass = args.bandpass or "0.01-100"
        args.notch = args.notch or 50.0
        args.webcam = args.webcam or "first"
        args.name = args.name if args.name != p.get_default("name") \
            else "brAIn stream lab validation"
    if args.stimulus_host:
        args.gp3_host = args.stimulus_host
    args.eeg = args.eeg or "unicorn"
    args.channel_type = args.channel_type or []
    return args


def electrode_names(path: Path | None) -> list[str]:
    if path is None:
        return []
    text = path.expanduser().read_text(encoding="utf-8")
    return [n.strip() for line in text.splitlines()
            for n in line.split(",") if n.strip() and
            not n.strip().startswith("#")]


def hiamp_config(args) -> dict:
    """The g.HIamp's configuration, with its amplifier filters chosen by
    their cut-offs among those it offers (it is opened briefly to list
    them)."""
    from brainstream.devices import make_device
    from brainstream.devices.hiamp import (HIampConfig,
                                           describe_amplifier_filter)
    names = electrode_names(args.electrodes)
    typed = {int(e.split("=")[0]) for e in args.channel_type}
    eeg_names = [n for i, n in enumerate(names, 1) if i not in typed]
    if args.electrodes and len(names) != args.channels:
        sys.exit(f"{args.electrodes} names {len(names)} channels, not "
                 f"{args.channels}.")
    cfg = HIampConfig(serial=args.hiamp, simulate=args.simulate,
                      sampling_rate=args.rate, channel_count=args.channels,
                      electrodes=names, channel_types=list(args.channel_type),
                      reference=args.reference, ground=args.ground,
                      placement="10-10" if len(eeg_names) > 32 else "10-20")
    if not (args.bandpass or args.notch):
        return cfg.to_dict()
    device = make_device("hiamp", cfg.to_dict(), Path(args.folder))
    try:
        found = device.load_filters()
    except Exception as error:  # noqa: BLE001
        sys.exit(f"Could not list the g.HIamp's filters ({error}). Connect "
                 f"the amplifier and run this again, or leave out "
                 f"--bandpass/--notch and choose them in brAIn stream "
                 f"(g.HIamp, Settings, Load from amplifier).")
    cfg = device.config
    if args.bandpass:
        low, high = (float(v) for v in args.bandpass.split("-"))
        match = [f for f in found.get("bandpass", [])
                 if abs(f["low"] - low) < 1e-6 and abs(f["high"] - high) < 1e-6]
        if not match:
            offered = "; ".join(describe_amplifier_filter(f)
                                for f in found.get("bandpass", []))
            sys.exit(f"The g.HIamp offers no {args.bandpass} Hz band-pass at "
                     f"{args.rate} Hz. It offers: {offered}")
        cfg.amp_bandpass = match[0]["index"]
    if args.notch:
        match = [f for f in found.get("notch", [])
                 if f["low"] <= args.notch <= f["high"]]
        if not match:
            sys.exit(f"The g.HIamp offers no {args.notch:g} Hz notch at "
                     f"{args.rate} Hz.")
        cfg.amp_notch = match[0]["index"]
    for kind in ("bandpass", "notch"):
        chosen = cfg.chosen_filter(kind)
        if chosen:
            print(f"g.HIamp {kind}: "
                  f"{describe_amplifier_filter(chosen, kind == 'notch')}")
    return cfg.to_dict()


def webcam_config(args) -> dict:
    cfg = WebcamConfig(label="camera", simulate=args.simulate)
    if args.webcam not in ("", "first") and not args.simulate:
        from brainstream.devices.video import find_cameras
        cameras = find_cameras()
        match = [c for c in cameras if args.webcam.lower() in c.name.lower()]
        if not match:
            sys.exit(f"No camera called {args.webcam!r}: "
                     f"{', '.join(c.name for c in cameras) or 'none found'}.")
        cfg.camera, cfg.camera_id = match[0].index, match[0].id
        cfg.camera_name = match[0].name
    return cfg.to_dict()


def main(argv=None) -> int:
    args = parse_args(argv)
    project = Project.create(Path(args.folder), args.name, task=TASK,
                             description="Validation of brAIn stream's "
                                         "recordings: EEG alpha and signal "
                                         "quality, gaze accuracy, N170 to "
                                         "faces and the synchronisation of "
                                         "the devices.")
    sim = args.simulate
    if args.eeg == "hiamp":
        project.add_device("hiamp", hiamp_config(args))
    else:
        project.add_device("unicorn", UnicornConfig(
            serial=args.unicorn, simulate=sim).to_dict())
    project.add_device("gp3", GP3Config(
        mode="remote", host=args.gp3_host, port=args.gp3_port,
        screen_width_cm=args.screen_width_cm, distance_cm=args.distance_cm,
        simulate=sim).to_dict())
    project.add_device("lsl", LslConfig(
        streams=[{"name": MARKERS, "type": "Markers", "source_id": ""}],
        simulate=False).to_dict())
    project.add_device("obs", ObsConfig(source=args.ndi_source,
                                        label="screen",
                                        simulate=sim).to_dict())
    if args.webcam:
        project.add_device("webcam", webcam_config(args))
    remember(project.path)
    print(f"Project: {project.path}\nRecordings: {project.output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
