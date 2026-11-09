"""Check a built package with every device simulated, without a display.

    <package python> -I smoke_test.py <library folder> <expected version>

Run with the package's own Python against the library folder its first
launch set up, so it checks what users get: the bundled Python, the
libraries installed from PyPI, and the bundled wheels. It loads every
device's driver, opens the main window, then records from every device
(simulated) into a BIDS dataset and checks the files.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, sys.argv[1])
EXPECTED = sys.argv[2]
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["BRAINDUMP_HOME"] = tempfile.mkdtemp()

import braindump  # noqa: E402
import gpype  # noqa: E402

assert braindump.__version__ == EXPECTED, braindump.__version__

# Every driver loads on this system.
drivers = {}
for name, check in (
        ("gtec_unicorn", lambda m: m.__version__),
        ("opengp3", lambda m: m.__version__),
        ("libusb_package", lambda m: m.get_library_path()),
        ("pylsl", lambda m: m.library_version()),
        ("bleak", lambda m: "ok"),
        ("serial", lambda m: m.__version__),
        ("pyshimmer", lambda m: "ok"),
        ("cv2", lambda m: m.__version__)):
    try:
        module = __import__(name)
        drivers[name] = check(module)
    except Exception as error:  # noqa: BLE001
        sys.exit(f"{name} does not load: {type(error).__name__}: {error}")
print(f"brAIn dump {braindump.__version__}, g.Pype {gpype.__version__}, "
      f"Python {sys.version.split()[0]} on {sys.platform}", flush=True)
for name, value in drivers.items():
    print(f"  {name}: {value}", flush=True)

from PySide6.QtWidgets import QApplication  # noqa: E402

from braindump.devices import DEVICE_TYPES, make_device  # noqa: E402
from braindump.project import Project  # noqa: E402
from braindump.recording import Session, Subject  # noqa: E402
from braindump.ui import theme  # noqa: E402
from braindump.ui.main_window import MainWindow  # noqa: E402

folder = Path(tempfile.mkdtemp())
project = Project.create(folder / "smoke", "Smoke test")
for kind in DEVICE_TYPES:
    project.add_device(kind, {"simulate": True})

app = QApplication([])
theme.apply(app)
window = MainWindow(project, simulate=True)
window.show()
app.processEvents()
window.close()
print("main window: ok", flush=True)

devices = [make_device(k, c, project.folder) for k, c in
           project.devices.items()]
for d in devices:
    d.start()
try:
    time.sleep(3)
    session = Session(project, devices)
    session.start(Subject(sub="01", task="smoke"))
    time.sleep(3)
    result = session.stop()
finally:
    for d in devices:
        d.stop()
assert not result.problems, result.problems
eeg = project.output_path / "sub-01" / "eeg"
expected = ["eeg.bdf", "recording-eye1_physio.tsv.gz", "events.tsv",
            "recording-ecg_physio.tsv.gz", "recording-gsr_physio.tsv.gz"]
for suffix in expected:
    path = eeg / f"sub-01_task-smoke_{suffix}"
    assert path.exists() and path.stat().st_size > 0, path
for name, samples in result.devices.items():
    print(f"  {name}: {samples} samples", flush=True)
    assert samples > 0 or name == "LSL streams", name
print("recording: ok")
print("PASS")
