"""What every device's page in the main window has: a header with the
device's state and its own Connect button, and tabs for its viewer and
its settings."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QScrollArea,
                               QTabWidget, QVBoxLayout, QWidget)

from ..devices.base import IDLE, RUNNING, STARTING, STOPPING, Device
from . import icons, theme
from .forms import hint


class DevicePage(QWidget):
    """Base of the Unicorn's and the GP3's pages."""

    #: The device's configuration was edited.
    config_changed = Signal()
    #: Ask the window to connect (True) or disconnect (False) this device.
    connect_requested = Signal(bool)

    def __init__(self, device: Device, parent=None):
        super().__init__(parent)
        self.device = device
        column = QVBoxLayout(self)
        column.setContentsMargins(10, 8, 10, 8)
        column.setSpacing(6)

        header = QHBoxLayout()
        self.icon = QLabel()
        self.icon.setPixmap(icons.device(device.TYPE).pixmap(26, 26))
        header.addWidget(self.icon)
        self.title = QLabel(f"<b>{device.NAME}</b>")
        header.addWidget(self.title)
        self.state_label = QLabel()
        self.state_label.setObjectName("Hint")
        header.addWidget(self.state_label)
        header.addStretch(1)
        self.controls = QHBoxLayout()
        header.addLayout(self.controls)
        self.connect_button = QPushButton()
        self.connect_button.setMinimumWidth(110)
        self.connect_button.clicked.connect(self._toggle)
        header.addWidget(self.connect_button)
        column.addLayout(header)

        self.tabs = QTabWidget()
        column.addWidget(self.tabs, 1)
        self.settings_note = hint("")
        self.set_state(device.state)

    def add_settings_tab(self, *forms: QWidget) -> None:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.addWidget(self.settings_note)
        for form in forms:
            layout.addWidget(form)
        layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(box)
        self.tabs.addTab(scroll, "Settings")

    def _toggle(self):
        self.connect_requested.emit(self.device.state == IDLE)

    def set_state(self, state: str) -> None:
        d = self.device
        text = {IDLE: "not connected", STARTING: "connecting…",
                RUNNING: "recording" if d.recording else "connected",
                STOPPING: "stopping…"}.get(state, state)
        ident = d.identity()
        self.state_label.setText(f"{ident} · {text}" if ident else text)
        self.connect_button.setText("Disconnect" if state == RUNNING
                                    else "Connect")
        self.connect_button.setObjectName("" if state == RUNNING
                                          else "Primary")
        self.connect_button.style().unpolish(self.connect_button)
        self.connect_button.style().polish(self.connect_button)
        self.connect_button.setEnabled(state in (IDLE, RUNNING)
                                       and not d.recording)
        running = state != IDLE
        self.settings_note.setText(
            "The device is connected: changes take effect when it next "
            "connects." if running else
            "Saved in the project as you change them.")
        self.settings_note.setStyleSheet(
            f"color: {theme.WARNING}" if running else "")

    def refresh(self) -> None:
        """Redraw the live views (called by the window's timer)."""

    def status_icons(self) -> list:
        return []

    def shown(self) -> None:
        """The page became visible."""


def placeholder(text: str) -> QWidget:
    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setObjectName("Hint")
    return label
