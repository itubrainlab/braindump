"""Start brAIn dump.

    braindump                         choose a project, then record
    braindump --project study.braindump
    braindump --simulate              every device simulated
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import APP_NAME, __version__


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="braindump",
        description="Multi-device recording (g.tec Unicorn EEG, Gazepoint "
                    "GP3 eye tracking) into BIDS, for the ITU brAIn lab.")
    parser.add_argument("--project", metavar="FILE",
                        help="open this project (a .braindump file) without "
                             "the start dialog")
    parser.add_argument("--simulate", action="store_true",
                        help="simulate every device (no hardware)")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    from . import macos
    if not args.simulate and macos.needs_relaunch():
        # macOS ends a process using Bluetooth for an app that does not
        # declare it, which a terminal does not; see macos.py.
        return macos.relaunch_with_bluetooth(
            sys.argv[1:] if argv is None else list(argv))

    from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

    from . import project as proj
    from .ui import theme
    from .ui.main_window import MainWindow
    from .ui.project_dialog import StartDialog

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setDesktopFileName("braindump")
    theme.apply(app)

    project = None
    if args.project:
        try:
            project = proj.Project.load(Path(args.project))
            proj.remember(project.path)
        except Exception as error:  # noqa: BLE001
            QMessageBox.critical(None, APP_NAME,
                                 f"Could not open {args.project}:\n{error}")
    # The start dialog comes back when the user switches project.
    while True:
        if project is None:
            dialog = StartDialog()
            if dialog.exec() != QDialog.DialogCode.Accepted:
                break
            project = dialog.project
        window = MainWindow(project, simulate=args.simulate)
        switch = []
        window.switch_project.connect(lambda: switch.append(True))
        window.show()
        app.exec()
        if not switch:
            break
        project = None
    return 0


if __name__ == "__main__":
    sys.exit(main())
