"""The project's page: its details, where its recordings go, its devices,
and the recordings made so far."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QComboBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QPlainTextEdit, QPushButton,
                               QScrollArea, QVBoxLayout, QWidget)

from ..bids import clean_label
from ..devices import DEVICE_TYPES
from ..project import Project
from ..recording import RecordingResult
from . import icons
from .forms import hint
from .project_dialog import FolderField


class ProjectPage(QScrollArea):
    """Edits the project in place; every change is saved."""

    #: The output folder or a project detail changed.
    changed = Signal()
    add_requested = Signal(str)
    remove_requested = Signal(str)

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.project = project
        self.setWidgetResizable(True)
        body = QWidget()
        column = QVBoxLayout(body)
        column.setContentsMargins(16, 12, 16, 12)

        details = QGroupBox("Project")
        form = QFormLayout(details)
        self.name = QLineEdit(project.name)
        self.name.editingFinished.connect(self._save)
        self.file = QLabel(str(project.path))
        self.file.setObjectName("Hint")
        self.task = QLineEdit(project.task)
        self.task.editingFinished.connect(self._save)
        self.authors = QLineEdit(", ".join(project.authors))
        self.authors.editingFinished.connect(self._save)
        self.line = QComboBox()
        self.line.addItems(["50 Hz", "60 Hz"])
        self.line.setCurrentText(f"{project.power_line_frequency} Hz")
        self.line.currentTextChanged.connect(lambda _: self._save())
        self.description = QPlainTextEdit(project.description)
        self.description.setFixedHeight(64)
        self.description.textChanged.connect(self._save)
        form.addRow("Name", self.name)
        form.addRow("File", self.file)
        form.addRow("Default task", self.task)
        form.addRow("Authors", self.authors)
        form.addRow("Mains frequency", self.line)
        form.addRow("Description", self.description)
        column.addWidget(details)

        bids = QGroupBox("Dataset metadata (BIDS)")
        form = QFormLayout(bids)
        self.license = QLineEdit(project.license)
        self.license.setToolTip("License of the dataset, e.g. CC0 or "
                                "CC-BY-4.0 (dataset_description.json)")
        self.institution = QLineEdit(project.institution_name)
        self.address = QLineEdit(project.institution_address)
        self.department = QLineEdit(project.institution_department)
        for edit in (self.license, self.institution, self.address,
                     self.department):
            edit.editingFinished.connect(self._save)
        form.addRow("License", self.license)
        form.addRow("Institution", self.institution)
        form.addRow("Address", self.address)
        form.addRow("Department", self.department)
        form.addRow("", hint("Written into the dataset and every sidecar. "
                             "dataset_description.json and the README are "
                             "written once, with the first recording."))
        column.addWidget(bids)

        output = QGroupBox("Recordings")
        form = QFormLayout(output)
        self.output = FolderField(project.output_folder,
                                  "Folder of the BIDS dataset")
        self.output.edit.editingFinished.connect(self._save)
        self.output.edit.textChanged.connect(self._path_hint)
        open_ = QPushButton(icons.folder(), "Open")
        open_.clicked.connect(self._open_output)
        form.addRow("BIDS dataset", self.output)
        self.resolved = hint("")
        form.addRow("", self.resolved)
        form.addRow("", open_)
        column.addWidget(output)

        devices = QGroupBox("Devices")
        self.device_box = QVBoxLayout(devices)
        column.addWidget(devices)

        recent = QGroupBox("Recorded in this session")
        box = QVBoxLayout(recent)
        self.recordings = QListWidget()
        self.recordings.setMinimumHeight(120)
        self.recordings.itemDoubleClicked.connect(self._reveal)
        box.addWidget(self.recordings)
        column.addWidget(recent)
        column.addStretch(1)
        self.setWidget(body)
        self._path_hint()
        self.show_devices({})

    def _path_hint(self):
        self.resolved.setText(f"→ {self.project.output_path}")

    def _save(self):
        p = self.project
        p.name = self.name.text().strip() or p.name
        p.task = clean_label(self.task.text()) or "rest"
        self.task.setText(p.task)
        p.authors = [a.strip() for a in self.authors.text().split(",")
                     if a.strip()]
        p.power_line_frequency = int(self.line.currentText().split()[0])
        p.description = self.description.toPlainText()
        p.license = self.license.text().strip()
        p.institution_name = self.institution.text().strip()
        p.institution_address = self.address.text().strip()
        p.institution_department = self.department.text().strip()
        if self.output.text():
            p.output_folder = self.output.text()
        p.save()
        self._path_hint()
        self.changed.emit()

    def lock(self, locked: bool) -> None:
        """No moving the output while recording."""
        self.output.setEnabled(not locked)

    def _open_output(self):
        path = self.project.output_path
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def show_devices(self, devices: dict) -> None:
        while self.device_box.count():
            item = self.device_box.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for key, cls in DEVICE_TYPES.items():
            line = QWidget()
            row = QHBoxLayout(line)
            row.setContentsMargins(0, 0, 0, 0)
            icon = QLabel()
            icon.setPixmap(icons.device(key).pixmap(20, 20))
            row.addWidget(icon)
            used = key in devices
            label = QLabel(f"<b>{cls.NAME}</b> — {cls.DESCRIPTION}")
            label.setEnabled(used)
            row.addWidget(label, 1)
            button = QPushButton("Remove" if used else "Add")
            button.clicked.connect(
                lambda _=False, k=key, u=used:
                (self.remove_requested if u else self.add_requested).emit(k))
            row.addWidget(button)
            self.device_box.addWidget(line)

    def add_recording(self, result: RecordingResult) -> None:
        when = datetime.fromtimestamp(result.started).strftime("%H:%M:%S")
        counts = ", ".join(f"{name} {n} samples"
                           for name, n in result.devices.items())
        item = QListWidgetItem(
            f"{when}  {result.entities.stem()}  ·  "
            f"{result.duration:.0f} s  ·  {counts}")
        item.setToolTip("\n".join(str(f) for f in result.files))
        folder = result.files[0].parent if result.files else None
        item.setData(256, str(folder) if folder else "")
        self.recordings.insertItem(0, item)

    def _reveal(self, item):
        folder = item.data(256)
        if folder and Path(folder).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))
