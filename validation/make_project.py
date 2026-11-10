"""Make the brAIn dump project the validation procedure records with.

    python validation/make_project.py ~/brAIn\\ dump/validation \\
        [--unicorn UN-2024.10.18] [--gp3-host 127.0.0.1] [--simulate]

Devices: the Unicorn (EEG), the GP3 read over the Open Gaze API (the GP3
runs under ``opengp3 server``, so that the procedure can draw its
calibration), the procedure's LSL markers, and OBS's screen over NDI.
``--simulate`` makes every device simulated, to try the whole chain
without hardware (the analysis's physiological checks then fail, as a
simulator has no eyes or faces).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from braindump.devices.gp3 import GP3Config
from braindump.devices.lsl import LslConfig
from braindump.devices.unicorn import UnicornConfig
from braindump.devices.video import ObsConfig
from braindump.project import Project, remember

TASK = "facevalidation"
MARKERS = "brAIndumpValidation"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("folder", help="where the project goes")
    p.add_argument("--name", default="brAIn dump validation")
    p.add_argument("--unicorn", default="", help="the headset's serial")
    p.add_argument("--gp3-host", default="127.0.0.1",
                   help="the computer running opengp3 server")
    p.add_argument("--gp3-port", type=int, default=4242)
    p.add_argument("--screen-width-cm", type=float, default=0.0,
                   help="the stimulus screen's width (for accuracy in "
                        "degrees); 0 asks the GP3's server")
    p.add_argument("--distance-cm", type=float, default=65.0)
    p.add_argument("--ndi-source", default="",
                   help="OBS's NDI source name (empty: the first OBS one)")
    p.add_argument("--simulate", action="store_true")
    args = p.parse_args(argv)

    project = Project.create(Path(args.folder), args.name, task=TASK,
                             description="Validation of brAIn dump's "
                                         "recordings: EEG alpha, gaze "
                                         "accuracy, N170 to faces and the "
                                         "synchronisation of the devices.")
    sim = args.simulate
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
    remember(project.path)
    print(f"Project: {project.path}\nRecordings: {project.output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
