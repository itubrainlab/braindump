"""brAIn stream's main window: a toolbar over a viewer, with a column of the
project's devices on the left."""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                               QStackedWidget, QToolButton, QVBoxLayout,
                               QWidget)

from .. import APP_NAME, __version__
from ..devices import DEVICE_TYPES, make_device
from ..pipeline import SessionPipeline
from ..devices.base import IDLE, RUNNING, STARTING, Device
from ..project import Project
from ..recording import RecordingResult, Session
from . import icons, theme
from .device_column import PROJECT, DeviceColumn
from .device_page import DevicePage
from .forms import Relay, divider, tool
from .gp3_page import GP3Page
from .hiamp_page import HIampPage
from .lsl_page import LslPage
from .polar_page import PolarPage
from .shimmer_page import ShimmerPage
from .project_page import ProjectPage
from .record_dialog import RecordDialog
from .unicorn_page import UnicornPage
from .video_page import VideoPage

VIEW_INTERVAL_MS = 40
STATUS_INTERVAL_MS = 500

HELP = f"""<h3>{APP_NAME} {__version__}</h3>
<p>Records the brAIn lab's devices together into a BIDS dataset.</p>
<ol>
<li><b>Devices</b> (left column): pick one to see its data and settings.
Add a supported device with <b>+</b>, remove one with <b>×</b>.
Settings are saved in the project as you change them.</li>
<li><b>Connect</b> (▶) connects every device; each page also has its own
Connect button.</li>
<li><b>Gazepoint GP3</b>: check the eyes in the eye view, then
<b>Calibrate</b> (and <b>Validate</b>, on this computer). The quality
shows beside it, and goes into the BIDS sidecars.</li>
<li><b>LSL streams</b>: list the streams to record (by name, type or
source id); each sample they send while recording becomes an event in the
run's <code>_events.tsv</code>, with the EEG's triggers.</li>
<li><b>Polar H10</b>: wear the strap (it switches on with skin contact);
its ECG, heart rate and RR intervals are recorded as BIDS physio, each
beat as a physioevent.</li>
<li><b>Shimmer3 GSR+</b>: pair the unit over Bluetooth first; its skin
conductance (and ear-clip PPG) is recorded as BIDS physio.</li>
<li><b>Record</b> (●) asks for the participant, session, task and run,
and records every connected device until you press it again (or
■).</li>
</ol>
<p>The dataset: EEG as BDF with <code>_eeg.json</code>,
<code>_channels.tsv</code> and <code>_events.tsv</code>; eye tracking as
behavioural data, under <code>beh/</code>, in BIDS physio
(<code>recording-eye1</code> left, <code>recording-eye2</code>
right, <code>recording-gaze</code> every GP3 channel), with fixations and
blinks in <code>_physioevents</code>. <code>scans.tsv</code> holds each
file's start time, to the microsecond, on this computer's clock.</p>
<p>Built on g.Pype and OpenGP3. g.Pype and the Unicorn
driver are under g.tec's non-commercial licence.</p>"""


class _Toolbar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Toolbar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)


class MainWindow(QMainWindow):
    #: The user asked for another project.
    switch_project = Signal()

    def __init__(self, project: Project, simulate: bool = False):
        super().__init__()
        self.project = project
        self.simulate = simulate
        self.devices: dict[str, Device] = {}
        self.pages: dict[str, DevicePage] = {}
        #: The one g.Pype pipeline every connected device is part of.
        self.pipeline = SessionPipeline()
        self.session: Optional[Session] = None
        self.record_started: Optional[float] = None
        self._pending_record = False
        self._closing = False
        self.relay = Relay(self)
        self.relay.state.connect(self._on_state)
        self.relay.error.connect(self._on_error)
        self.relay.done.connect(lambda callback, result: callback(result))

        self.setWindowTitle(f"{project.name} — {APP_NAME}")
        self.setWindowIcon(icons.app())
        self.resize(1440, 860)
        central = QWidget()
        column = QVBoxLayout(central)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(self._toolbar())
        body = QHBoxLayout()
        body.setSpacing(0)
        self.column = DeviceColumn()
        self.column.selected.connect(self._select)
        self.column.add_requested.connect(self._add_device)
        self.column.remove_requested.connect(self._remove_device)
        body.addWidget(self.column)
        self.stack = QStackedWidget()
        body.addWidget(self.stack, 1)
        column.addLayout(body, 1)
        self.setCentralWidget(central)

        self.project_page = ProjectPage(project)
        self.project_page.changed.connect(self._project_changed)
        self.project_page.add_requested.connect(self._add_device)
        self.project_page.remove_requested.connect(self._remove_device)
        self.stack.addWidget(self.project_page)

        self.status_left = QLabel()
        self.status_right = QLabel()
        self.status_right.setObjectName("Hint")
        self.statusBar().addWidget(self.status_left, 1)
        self.statusBar().addPermanentWidget(self.status_right)

        for key in project.devices:
            if key in DEVICE_TYPES:
                self._create_device(key)
        self._rebuild_column()
        first = next(iter(self.devices), PROJECT)
        self.column.select(first)

        self.view_timer = QTimer(self)
        self.view_timer.setInterval(VIEW_INTERVAL_MS)
        self.view_timer.timeout.connect(self._refresh_view)
        self.view_timer.start()
        self.status_timer = QTimer(self)
        self.status_timer.setInterval(STATUS_INTERVAL_MS)
        self.status_timer.timeout.connect(self._refresh_status)
        self.status_timer.start()
        self._refresh_status()

    # -- toolbar -------------------------------------------------------------

    def _toolbar(self) -> QWidget:
        bar = _Toolbar()
        row = QHBoxLayout(bar)
        row.setContentsMargins(10, 4, 10, 4)
        row.setSpacing(6)
        logo = QLabel()
        logo.setPixmap(icons.app().pixmap(QSize(26, 26)))
        row.addWidget(logo)
        for text, name in (("br", "Brand"), ("AI", "BrandAI"), ("n", "Brand"),
                           (" dump", "BrandSub")):
            label = QLabel(text)
            label.setObjectName(name)
            row.addWidget(label)
        row.setSpacing(0)
        row.addSpacing(12)
        row.addWidget(divider())
        row.addSpacing(12)
        self.project_label = QLabel(self.project.name)
        self.project_label.setObjectName("ProjectName")
        self.project_label.setToolTip(str(self.project.path))
        row.addWidget(self.project_label)
        row.addSpacing(12)
        row.addWidget(divider())
        row.addSpacing(8)
        self.connect_button = tool(icons.connect(), "Connect every device")
        self.connect_button.clicked.connect(self.connect_all)
        self.stop_button = tool(icons.disconnect(), "Stop recording and "
                                "disconnect every device")
        self.stop_button.clicked.connect(self._stop_clicked)
        self.record_button = tool(icons.record(theme.RECORDING),
                                  "Record from every connected device")
        self.record_button.clicked.connect(self._record_clicked)
        for b in (self.connect_button, self.stop_button, self.record_button):
            row.addWidget(b)
            row.addSpacing(4)
        row.addSpacing(8)
        self.record_time = QLabel("")
        self.record_time.setObjectName("RecordTime")
        row.addWidget(self.record_time)
        row.addStretch(1)
        self.icon_row = QHBoxLayout()
        self.icon_row.setSpacing(6)
        row.addLayout(self.icon_row)
        row.addSpacing(10)
        row.addWidget(divider())
        row.addSpacing(10)
        settings = tool(icons.gear(), "Project settings")
        settings.clicked.connect(lambda: self.column.select(PROJECT))
        switch = tool(icons.folder(), "Open another project")
        switch.clicked.connect(self._switch)
        help_ = tool(icons.question(), "Help")
        help_.clicked.connect(
            lambda: QMessageBox.about(self, f"About {APP_NAME}", HELP))
        for b in (settings, switch, help_):
            row.addWidget(b)
            row.addSpacing(4)
        return bar

    # -- devices ---------------------------------------------------------------

    def _create_device(self, key: str) -> None:
        device = make_device(key, self.project.devices[key],
                             self.project.folder, self.simulate)
        device.attach(self.pipeline)
        device.on_state = lambda state, k=key: self.relay.state.emit(k, state)
        device.on_error = lambda message, k=key: self.relay.error.emit(
            k, message)
        page_cls = {"unicorn": UnicornPage, "hiamp": HIampPage,
                    "gp3": GP3Page,
                    "lsl": LslPage, "polar": PolarPage,
                    "shimmer": ShimmerPage, "webcam": VideoPage,
                    "obs": VideoPage}[key]
        page = page_cls(device)
        page.config_changed.connect(lambda k=key: self._save_config(k))
        page.connect_requested.connect(
            lambda on, k=key: self.connect_device(k) if on
            else self.disconnect_device(k))
        self.devices[key] = device
        self.pages[key] = page
        self.stack.addWidget(page)

    def _rebuild_column(self):
        self.column.rebuild(self.project.name,
                            str(self.project.output_path), self.devices)
        self.project_page.show_devices(self.devices)

    def _save_config(self, key: str) -> None:
        device = self.devices.get(key)
        if device is not None:
            self.project.set_device_config(key, device.config.to_dict())

    def _add_device(self, key: str) -> None:
        if key in self.devices or self._recording():
            return
        cls = DEVICE_TYPES[key]
        self.project.add_device(key, cls.CONFIG().to_dict())
        self._create_device(key)
        self._rebuild_column()
        self.column.select(key)

    def _remove_device(self, key: str) -> None:
        device = self.devices.get(key)
        if device is None or self._recording():
            return
        if device.state != IDLE:
            QMessageBox.information(self, "Remove device",
                                    f"Disconnect the {device.NAME} first.")
            return
        answer = QMessageBox.question(
            self, "Remove device",
            f"Remove the {device.NAME} from the project? Its configuration "
            f"is forgotten.")
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.project.remove_device(key)
        page = self.pages.pop(key)
        self.devices.pop(key)
        self.stack.removeWidget(page)
        page.deleteLater()
        self._rebuild_column()
        self.column.select(PROJECT)

    def _select(self, key: str) -> None:
        widget = self.project_page if key == PROJECT else self.pages.get(key)
        if widget is not None:
            self.stack.setCurrentWidget(widget)
            if isinstance(widget, DevicePage):
                widget.shown()

    def _project_changed(self):
        self.project_label.setText(self.project.name)
        self.setWindowTitle(f"{self.project.name} — {APP_NAME}")
        self.column.project_name = self.project.name
        card = self.column.cards.get(PROJECT)
        if card:
            card.title.setText(self.project.name)
            card.sub.setText(str(self.project.output_path))
        self._refresh_status()

    # -- connecting ------------------------------------------------------------

    def _in_thread(self, work: Callable[[], object],
                   then: Optional[Callable[[object], None]] = None) -> None:
        def run():
            try:
                result = work()
            except Exception as error:  # noqa: BLE001
                result = error
            if then is not None:
                self.relay.done.emit(then, result)
        threading.Thread(target=run, daemon=True).start()

    def connect_device(self, key: str) -> None:
        device = self.devices[key]
        if device.state != IDLE:
            return

        def done(result):
            if isinstance(result, Exception):
                self._pending_record = False
                QMessageBox.critical(
                    self, f"{device.NAME}",
                    f"Could not connect the {device.NAME}:\n\n{result}")
            self._after_connect()
        self._in_thread(device.start, done)

    def disconnect_device(self, key: str) -> None:
        device = self.devices[key]
        if device.recording:
            return
        self._in_thread(device.stop)

    def connect_all(self) -> None:
        """Every idle device into the session pipeline, which starts once
        with all of them."""
        idle = [d for d in self.devices.values() if d.state == IDLE]
        if not idle:
            return

        def done(result):
            if isinstance(result, Exception):
                self._pending_record = False
                device = getattr(result, "device", None)
                name = device.NAME if device is not None else "a device"
                QMessageBox.critical(
                    self, APP_NAME,
                    f"Could not connect {name}, so nothing was connected:"
                    f"\n\n{result}")
            self._after_connect()
        self._in_thread(lambda: self.pipeline.connect(idle), done)

    def _after_connect(self):
        if self._pending_record and not any(
                d.state == STARTING for d in self.devices.values()):
            self._pending_record = False
            if any(d.running for d in self.devices.values()):
                self._ask_and_record()

    def _on_state(self, key: str, state: str) -> None:
        page = self.pages.get(key)
        if page is not None:
            page.set_state(state)
        self.column.refresh()
        self._refresh_status()

    def _on_error(self, key: str, message: str) -> None:
        device = self.devices.get(key)
        name = device.NAME if device else key
        if self.session is not None and self.session.active and \
                not any(d.recording for d in self.devices.values()):
            self._finish_recording()
        QMessageBox.critical(
            self, name,
            f"The {name} stopped, and with it every device (they share one "
            f"g.Pype pipeline): connect again to go on.\n\n{message}")

    # -- recording -------------------------------------------------------------

    def _recording(self) -> bool:
        return self.session is not None and self.session.active

    def _record_clicked(self):
        if self._recording():
            self.stop_recording()
            return
        if not self.devices:
            QMessageBox.information(self, "Record", "The project has no "
                                    "devices: add one in the left column.")
            return
        idle = [k for k, d in self.devices.items() if d.state == IDLE]
        if idle:
            self._pending_record = True
            self.status_left.setText("Connecting the devices before "
                                     "recording…")
            for key in idle:
                self.connect_device(key)
            return
        if any(d.state == STARTING for d in self.devices.values()):
            self._pending_record = True
            return
        self._ask_and_record()

    def _ask_and_record(self):
        running = [d for d in self.devices.values() if d.running]
        if not running:
            return
        missing = [d.NAME for d in self.devices.values() if not d.running]
        if missing:
            answer = QMessageBox.question(
                self, "Record", f"Not connected: {', '.join(missing)}.\n"
                                f"Record without it?")
            if answer != QMessageBox.StandardButton.Yes:
                return
        gp3 = self.devices.get("gp3")
        if gp3 is not None and gp3.running and gp3.calibrated is False:
            answer = QMessageBox.question(
                self, "Record", "The GP3 is not calibrated: its gaze will "
                "not be on the screen.\nRecord anyway?")
            if answer != QMessageBox.StandardButton.Yes:
                self.column.select("gp3")
                return
        session = Session(self.project, list(self.devices.values()))
        dialog = RecordDialog(session.dataset, self.project.task,
                              [d.NAME for d in running], self,
                              tasks=self.project.tasks,
                              eeg=any(d.MODALITY == "eeg" for d in running))
        if dialog.exec() != RecordDialog.DialogCode.Accepted:
            return
        s = dialog.subject
        info = {k: v for k, v in (("TaskDescription", s.task_description),
                                  ("Instructions", s.instructions)) if v}
        if info and self.project.tasks.get(s.task) != info:
            self.project.tasks[s.task] = info
            self.project.save()
        try:
            session.start(dialog.subject)
        except Exception as error:  # noqa: BLE001
            QMessageBox.critical(self, "Record",
                                 f"Could not start recording:\n\n{error}")
            return
        self.session = session
        self.record_started = time.time()
        self._recording_ui(True)

    def stop_recording(self, then: Optional[Callable[[], None]] = None):
        session = self.session
        if session is None or not session.active:
            if then:
                then()
            return
        self.record_button.setEnabled(False)
        self.status_left.setText("Finishing the recording…")

        def done(result):
            self._finish_recording(result)
            if then:
                then()
        self._in_thread(session.stop, done)

    def _finish_recording(self, result=None):
        self.record_started = None
        self.record_button.setEnabled(True)
        self._recording_ui(False)
        if isinstance(result, Exception):
            QMessageBox.critical(self, "Record",
                                 f"The recording did not finish cleanly:"
                                 f"\n\n{result}")
        elif isinstance(result, RecordingResult):
            self.project_page.add_recording(result)
            text = f"Saved {result.entities.stem()} " \
                   f"({result.duration:.0f} s)"
            if result.problems:
                QMessageBox.warning(self, "Record", "\n".join(
                    result.problems))
            self.statusBar().showMessage(text, 8000)
        self.session = None

    def _recording_ui(self, on: bool):
        self.record_button.setIcon(icons.stop(theme.RECORDING) if on
                                   else icons.record(theme.RECORDING))
        self.record_button.setToolTip("Stop recording" if on else
                                      "Record from every connected device")
        self.record_time.setText("● REC 00:00" if on else "")
        self.column.locked = on
        self.project_page.lock(on)
        for key, page in self.pages.items():
            page.set_state(self.devices[key].state)
        self.column.refresh()

    def _stop_clicked(self):
        def disconnect():
            self._in_thread(self.pipeline.stop)
        self.stop_recording(then=disconnect)

    # -- timers ----------------------------------------------------------------

    def _refresh_view(self):
        widget = self.stack.currentWidget()
        if isinstance(widget, DevicePage):
            try:
                widget.refresh()
            except Exception:  # noqa: BLE001 - a frame is not worth a crash
                pass

    def _refresh_status(self):
        self.column.refresh()
        parts = [d.status() for d in self.devices.values()]
        if not self._pending_record:
            self.status_left.setText("   ·   ".join(parts) or
                                     "No devices in this project")
        self.status_right.setText(f"BIDS: {self.project.output_path}")
        if self.record_started is not None:
            seconds = int(time.time() - self.record_started)
            self.record_time.setText(
                f"● REC {seconds // 60:02d}:{seconds % 60:02d}")
        while self.icon_row.count():
            item = self.icon_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for page in self.pages.values():
            for icon, tip in page.status_icons():
                button = QToolButton()
                button.setIcon(icon)
                button.setIconSize(QSize(22, 22))
                button.setToolTip(tip)
                button.setAutoRaise(True)
                self.icon_row.addWidget(button)
        running = any(d.state != IDLE for d in self.devices.values())
        self.stop_button.setEnabled(running)
        self.connect_button.setEnabled(any(d.state == IDLE
                                           for d in self.devices.values()))

    # -- closing ---------------------------------------------------------------

    def _switch(self):
        if self._recording():
            QMessageBox.information(self, "Open another project",
                                    "Stop recording first.")
            return
        self._switching = True
        self.close()

    def closeEvent(self, event):
        if self._recording():
            answer = QMessageBox.question(
                self, APP_NAME, "A recording is running. Stop it and quit?")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                self._switching = False
                return
            self.session.stop()
        self.view_timer.stop()
        self.status_timer.stop()
        try:
            self.pipeline.stop()
        except Exception:  # noqa: BLE001
            pass
        for key, device in self.devices.items():
            self._save_config(key)
        event.accept()
        if getattr(self, "_switching", False):
            self.switch_project.emit()
