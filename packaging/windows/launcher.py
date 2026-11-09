"""Start brAIn dump from its unpacked (or installed) folder, Windows.

    pythonw.exe -I launcher.py [braindump options]

The folder carries its own Python and the wheels of brAIn dump, Rhino
Recorder and OpenGP3, but not g.Pype or g.tec's gtec-unicorn driver: their
licences do not allow them in a bundle. On first launch, and after an
update, bootstrap.py installs them and the other libraries from PyPI into
%LOCALAPPDATA%\\brAIn dump\\runtime.
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = (HERE / "build-id").read_text().strip()
LIB = (Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
       / "brAIn dump" / "runtime" / BUILD)

if not (LIB / ".complete").exists():
    setup = subprocess.run([sys.executable, "-I", str(HERE / "bootstrap.py"),
                            str(HERE), str(LIB)])
    if setup.returncode != 0:
        sys.exit(1)

sys.path.insert(0, str(LIB))
sys.argv[0] = "braindump"
from braindump.app import main  # noqa: E402

sys.exit(main())
