"""The start dialog: open a recent or existing project, or make a new one."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox,
                               QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem,
                               QMessageBox, QPushButton, QToolButton,
                               QVBoxLayout, QWidget)

from .. import APP_NAME, __version__
from .. import project as proj
from ..bids import clean_label
from ..devices import DEVICE_TYPES
from . import icons, theme


def brand(size: int = 56) -> QWidget:
    """The logo and the name, as the start dialog shows them."""
    box = QWidget()
    row = QHBoxLayout(box)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(12)
    logo = QLabel()
    logo.setPixmap(icons.app().pixmap(QSize(size, size)))
    row.addWidget(logo)
    column = QVBoxLayout()
    column.setSpacing(0)
    title = QLabel(
        f'<span style="font-size:24px;font-weight:700">br</span>'
        f'<span style="font-size:24px;font-weight:800;color:{theme.BRAND}">'
        f'AI</span><span style="font-size:24px;font-weight:700">n</span>'
        f'<span style="font-size:24px;color:{theme.MUTED}"> dump</span>')
    column.addWidget(title)
    sub = QLabel(f"Multi-device recording for the brAIn lab · "
                 f"v{__version__}")
    sub.setObjectName("Hint")
    column.addWidget(sub)
    row.addLayout(column)
    row.addStretch(1)
    return box


class FolderField(QWidget):
    """A path and a Browse button."""

    def __init__(self, path: str = "", caption: str = "Choose a folder",
                 parent=None):
        super().__init__(parent)
        self.caption = caption
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(path)
        row.addWidget(self.edit, 1)
        browse = QToolButton(text="Browse…")
        browse.clicked.connect(self._browse)
        row.addWidget(browse)

    def _browse(self):
        start = self.edit.text() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, self.caption, start)
        if folder:
            self.edit.setText(folder)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, text: str) -> None:
        self.edit.setText(text)


class NewProjectDialog(QDialog):
    """Name, place, output folder and devices of a new project."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New project")
        self.setMinimumWidth(560)
        self.project: Optional[proj.Project] = None
        column = QVBoxLayout(self)
        form = QFormLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. Uncanny faces EEG")
        self.location = FolderField(str(Path.home() / "brAIn stream"),
                                    "Where the project folder goes")
        self.output = FolderField("", "BIDS dataset folder")
        self.output.edit.setPlaceholderText(
            "default: the project folder's bids/")
        self.task = QLineEdit("rest")
        self.authors = QLineEdit()
        self.authors.setPlaceholderText("Comma-separated")
        form.addRow("Name", self.name)
        form.addRow("Location", self.location)
        form.addRow("Recordings (BIDS)", self.output)
        form.addRow("Default task", self.task)
        form.addRow("Authors", self.authors)
        column.addLayout(form)
        self.folder_hint = QLabel()
        self.folder_hint.setObjectName("Hint")
        self.folder_hint.setWordWrap(True)
        column.addWidget(self.folder_hint)
        column.addWidget(QLabel("Devices"))
        self.devices = {}
        for key, cls in DEVICE_TYPES.items():
            box = QCheckBox(f"{cls.NAME} — {cls.DESCRIPTION}")
            box.setChecked(True)
            self.devices[key] = box
            column.addWidget(box)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Create")
        buttons.accepted.connect(self._create)
        buttons.rejected.connect(self.reject)
        column.addWidget(buttons)
        self.name.textChanged.connect(self._hint)
        self.location.edit.textChanged.connect(self._hint)
        self._hint()

    def _folder(self) -> Path:
        return Path(self.location.text()).expanduser() / \
            (self.name.text().strip() or "…")

    def _hint(self):
        self.folder_hint.setText(f"The project goes in {self._folder()}")

    def _create(self):
        name = self.name.text().strip()
        if not name:
            QMessageBox.warning(self, "New project", "Give the project a name.")
            return
        task = clean_label(self.task.text()) or "rest"
        authors = [a.strip() for a in self.authors.text().split(",")
                   if a.strip()]
        try:
            project = proj.Project.create(
                self._folder(), name, output_folder=self.output.text()
                or "bids", task=task, authors=authors)
            for key, box in self.devices.items():
                if box.isChecked():
                    project.add_device(key, DEVICE_TYPES[key].CONFIG()
                                       .to_dict())
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "New project", str(error))
            return
        self.project = project
        self.accept()


class StartDialog(QDialog):
    """What brAIn stream opens with: pick the project to record for."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(icons.app())
        self.setMinimumSize(620, 460)
        self.project: Optional[proj.Project] = None
        column = QVBoxLayout(self)
        column.setContentsMargins(24, 24, 24, 24)
        column.setSpacing(14)
        column.addWidget(brand())
        title = QLabel("Choose a project")
        title.setObjectName("Big")
        column.addWidget(title)
        hint = QLabel("A project keeps its devices, their configuration and "
                      "where its recordings go.")
        hint.setObjectName("Hint")
        column.addWidget(hint)
        self.recent = QListWidget()
        self.recent.setIconSize(QSize(28, 28))
        self.recent.itemDoubleClicked.connect(lambda _: self._open_selected())
        self.recent.currentRowChanged.connect(self._selection)
        column.addWidget(self.recent, 1)
        row = QHBoxLayout()
        new = QPushButton(icons.plus(), "New project…")
        new.clicked.connect(self._new)
        browse = QPushButton(icons.folder(), "Open other…")
        browse.clicked.connect(self._browse)
        self.forget = QPushButton("Remove from list")
        self.forget.clicked.connect(self._forget)
        row.addWidget(new)
        row.addWidget(browse)
        row.addWidget(self.forget)
        row.addStretch(1)
        self.open = QPushButton("Open")
        self.open.setObjectName("Primary")
        self.open.setDefault(True)
        self.open.clicked.connect(self._open_selected)
        quit_ = QPushButton("Quit")
        quit_.clicked.connect(self.reject)
        row.addWidget(quit_)
        row.addWidget(self.open)
        column.addLayout(row)
        self._fill()

    def _fill(self):
        self.recent.clear()
        for path in proj.recent_projects():
            try:
                project = proj.Project.load(path)
            except Exception:  # noqa: BLE001
                continue
            devices = ", ".join(DEVICE_TYPES[t].NAME for t in project.devices
                                if t in DEVICE_TYPES) or "no devices"
            item = QListWidgetItem(icons.app(),
                                   f"{project.name}\n{devices} · {path.parent}")
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setToolTip(str(path))
            self.recent.addItem(item)
        if self.recent.count():
            self.recent.setCurrentRow(0)
        else:
            empty = QListWidgetItem("No recent projects: make a new one, or "
                                    "open one from disk.")
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            self.recent.addItem(empty)
        self._selection()

    def _selected_path(self) -> Optional[str]:
        item = self.recent.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _selection(self, *_):
        on = bool(self._selected_path())
        self.open.setEnabled(on)
        self.forget.setEnabled(on)

    def _load(self, path: str) -> None:
        try:
            self.project = proj.Project.load(path)
        except Exception as error:  # noqa: BLE001
            QMessageBox.critical(self, "Open project",
                                 f"Could not open {path}:\n{error}")
            return
        proj.remember(path)
        self.accept()

    def _open_selected(self):
        path = self._selected_path()
        if path:
            self._load(path)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open project", str(Path.home()),
            "brAIn stream projects (" + " ".join(
                f"*{e}" for e in (proj.EXTENSION, *proj.OLD_EXTENSIONS))
            + ")")
        if path:
            self._load(path)

    def _forget(self):
        path = self._selected_path()
        if path:
            proj.forget(path)
            self._fill()

    def _new(self):
        dialog = NewProjectDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.project:
            proj.remember(dialog.project.path)
            self.project = dialog.project
            self.accept()
