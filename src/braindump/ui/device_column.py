"""The left column: the project, its devices with their state, and the
supported devices it does not use yet."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QScrollArea,
                               QToolButton, QVBoxLayout, QWidget)

from ..devices import DEVICE_TYPES
from ..devices.base import IDLE, RUNNING, STARTING, STOPPING, Device
from . import icons, theme

PROJECT = "project"


class Card(QFrame):
    """One entry of the column."""

    clicked = Signal()

    def __init__(self, icon, title: str, sub: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("DeviceCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 8, 6, 8)
        row.setSpacing(8)
        self.icon = QLabel()
        self.icon.setPixmap(icon.pixmap(24, 24))
        row.addWidget(self.icon, 0, Qt.AlignmentFlag.AlignTop)
        text = QVBoxLayout()
        text.setSpacing(1)
        self.title = QLabel(title)
        self.title.setObjectName("CardTitle")
        self.sub = QLabel(sub)
        self.sub.setObjectName("CardSub")
        self.sub.setWordWrap(True)
        text.addWidget(self.title)
        text.addWidget(self.sub)
        row.addLayout(text, 1)
        self.dot = QLabel()
        row.addWidget(self.dot, 0, Qt.AlignmentFlag.AlignTop)
        self.button = QToolButton()
        self.button.setIconSize(QSize(14, 14))
        self.button.setAutoRaise(True)
        self.button.setVisible(False)
        row.addWidget(self.button, 0, Qt.AlignmentFlag.AlignTop)

    def set_selected(self, on: bool) -> None:
        self.setProperty("selected", "true" if on else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def set_dot(self, colour: str | None, tip: str = "") -> None:
        if colour is None:
            self.dot.clear()
        else:
            self.dot.setPixmap(icons.dot(colour).pixmap(12, 12))
        self.dot.setToolTip(tip)

    def mousePressEvent(self, event):
        self.clicked.emit()


def state_colour(device: Device) -> tuple[str | None, str]:
    if device.recording:
        return theme.RECORDING, "recording"
    return {IDLE: (theme.MUTED, "not connected"),
            STARTING: (theme.WARNING, "connecting"),
            RUNNING: (theme.GOOD, "connected"),
            STOPPING: (theme.WARNING, "stopping")}.get(device.state,
                                                      (None, ""))


class DeviceColumn(QWidget):
    selected = Signal(str)
    add_requested = Signal(str)
    remove_requested = Signal(str)

    WIDTH = 250

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DeviceColumn")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedWidth(self.WIDTH)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)
        self.body = QWidget()
        self.column = QVBoxLayout(self.body)
        self.column.setContentsMargins(0, 0, 0, 0)
        self.column.setSpacing(0)
        scroll.setWidget(self.body)
        self.cards: dict[str, Card] = {}
        self.current = PROJECT
        self.devices: dict[str, Device] = {}
        self.project_name = ""
        self.project_sub = ""
        self.locked = False

    @staticmethod
    def _title(text: str) -> QLabel:
        label = QLabel(text.upper())
        label.setObjectName("ColumnTitle")
        return label

    def rebuild(self, project_name: str, output: str,
                devices: dict[str, Device]) -> None:
        self.project_name, self.project_sub = project_name, output
        self.devices = devices
        while self.column.count():
            item = self.column.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.cards = {}
        self.column.addWidget(self._title("Project"))
        card = Card(icons.app(), project_name, output)
        card.clicked.connect(lambda: self.select(PROJECT))
        self.cards[PROJECT] = card
        self.column.addWidget(card)

        self.column.addWidget(self._title("Devices"))
        if not devices:
            note = QLabel("No devices: add one below.")
            note.setObjectName("Hint")
            note.setContentsMargins(12, 4, 12, 8)
            self.column.addWidget(note)
        for key, device in devices.items():
            card = Card(icons.device(key), device.NAME, "")
            card.clicked.connect(lambda k=key: self.select(k))
            card.button.setIcon(icons.cross())
            card.button.setToolTip(f"Remove the {device.NAME} from the "
                                   f"project")
            card.button.clicked.connect(
                lambda _=False, k=key: self.remove_requested.emit(k))
            card.button.setVisible(True)
            self.cards[key] = card
            self.column.addWidget(card)

        missing = [k for k in DEVICE_TYPES if k not in devices]
        if missing:
            self.column.addWidget(self._title("Supported devices"))
        for key in missing:
            cls = DEVICE_TYPES[key]
            card = Card(icons.device(key, theme.MUTED), cls.NAME,
                        cls.DESCRIPTION)
            card.setProperty("available", "true")
            card.button.setIcon(icons.plus(theme.ICON_ACTIVE))
            card.button.setToolTip(f"Add the {cls.NAME} to the project")
            card.button.setVisible(True)
            card.button.clicked.connect(
                lambda _=False, k=key: self.add_requested.emit(k))
            card.clicked.connect(lambda k=key: self.add_requested.emit(k))
            card.setToolTip(f"Add the {cls.NAME} to the project")
            self.column.addWidget(card)
        self.column.addStretch(1)
        if self.current not in self.cards:
            self.current = PROJECT
        self.select(self.current, emit=False)
        self.refresh()

    def select(self, key: str, emit: bool = True) -> None:
        self.current = key
        for k, card in self.cards.items():
            card.set_selected(k == key)
        if emit:
            self.selected.emit(key)

    def refresh(self) -> None:
        for key, device in self.devices.items():
            card = self.cards.get(key)
            if card is None:
                continue
            colour, text = state_colour(device)
            ident = device.identity()
            card.sub.setText(f"{ident}\n{text}" if ident else text)
            card.set_dot(colour, text)
            card.button.setEnabled(device.state == IDLE and not self.locked)
