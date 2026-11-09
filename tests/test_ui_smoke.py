"""The main window opens on a project, connects simulated devices and
draws their views (offscreen)."""
import time

from PySide6.QtWidgets import QApplication

from braindump.project import Project
from braindump.ui import theme
from braindump.ui.main_window import MainWindow


def test_main_window_simulated(tmp_path):
    app = QApplication.instance() or QApplication([])
    theme.apply(app)
    project = Project.create(tmp_path / "p", "P")
    project.add_device("unicorn", {})
    project.add_device("gp3", {})
    project.add_device("lsl", {})
    window = MainWindow(project, simulate=True)
    window.show()
    for d in window.devices.values():
        d.start()
    deadline = time.time() + 2
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
    for key, page in window.pages.items():
        window.column.select(key)
        page.refresh()
    window._refresh_status()
    assert all(d.running for d in window.devices.values())
    window.close()
    assert not any(d.running for d in window.devices.values())
