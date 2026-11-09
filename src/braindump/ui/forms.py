"""Small helpers for the configuration forms and the panels."""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QObject, QSize, Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QFormLayout, QFrame, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QSpinBox, QToolButton,
                               QVBoxLayout, QWidget)


class ConfigForm(QGroupBox):
    """A form whose fields edit a config dataclass's attributes in place,
    and say so through ``changed``."""

    changed = Signal(str)

    def __init__(self, title: str, config, parent=None):
        super().__init__(title, parent)
        self.config = config
        self.form = QFormLayout(self)
        self.form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.fields: dict[str, QWidget] = {}

    def _row(self, attr: str, label: str, widget: QWidget, tip: str):
        if tip:
            widget.setToolTip(tip)
        self.fields[attr] = widget
        self.form.addRow(label, widget)
        return widget

    def _set(self, attr: str, value) -> None:
        if getattr(self.config, attr) != value:
            setattr(self.config, attr, value)
            self.changed.emit(attr)

    def line(self, attr, label, tip="", placeholder=""):
        edit = QLineEdit(str(getattr(self.config, attr)))
        edit.setPlaceholderText(placeholder)
        edit.editingFinished.connect(
            lambda: self._set(attr, edit.text().strip()))
        return self._row(attr, label, edit, tip)

    def words(self, attr, label, tip=""):
        """A list of strings, comma-separated."""
        edit = QLineEdit(", ".join(getattr(self.config, attr)))
        edit.editingFinished.connect(lambda: self._set(
            attr, [w.strip() for w in edit.text().split(",")]))
        return self._row(attr, label, edit, tip)

    def integer(self, attr, label, low, high, tip="", suffix="",
                special=""):
        box = QSpinBox()
        box.setRange(low, high)
        box.setValue(int(getattr(self.config, attr)))
        box.setSuffix(suffix)
        if special:
            box.setSpecialValueText(special)
        box.valueChanged.connect(lambda v: self._set(attr, int(v)))
        return self._row(attr, label, box, tip)

    def number(self, attr, label, low, high, tip="", suffix="", decimals=1,
               special="", step=1.0):
        box = QDoubleSpinBox()
        box.setRange(low, high)
        box.setDecimals(decimals)
        box.setSingleStep(step)
        box.setValue(float(getattr(self.config, attr)))
        box.setSuffix(suffix)
        if special:
            box.setSpecialValueText(special)
        box.valueChanged.connect(lambda v: self._set(attr, float(v)))
        return self._row(attr, label, box, tip)

    def boolean(self, attr, label, text="", tip=""):
        box = QCheckBox(text)
        box.setChecked(bool(getattr(self.config, attr)))
        box.toggled.connect(lambda v: self._set(attr, bool(v)))
        return self._row(attr, label, box, tip)

    def choice(self, attr, label, options: dict, tip=""):
        """``options`` maps stored values to the text shown."""
        combo = QComboBox()
        for value, text in options.items():
            combo.addItem(text, value)
        index = combo.findData(getattr(self.config, attr))
        combo.setCurrentIndex(max(0, index))
        combo.currentIndexChanged.connect(
            lambda i: self._set(attr, combo.itemData(i)))
        return self._row(attr, label, combo, tip)

    def show_row(self, attr: str, visible: bool) -> None:
        self.form.setRowVisible(self.fields[attr], visible)


def panel(title: str, content: QWidget, corner: QWidget | None = None
          ) -> QGroupBox:
    """Content in a titled group box, as g.Pype and Rhino Recorder show
    their views."""
    from rhinorecorder.ui.main_window import _panel
    return _panel(title, content, corner)


def tool(icon, tip: str, size: int = 24) -> QToolButton:
    button = QToolButton()
    button.setIcon(icon)
    button.setIconSize(QSize(size, size))
    button.setToolTip(tip)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    return button


def divider() -> QFrame:
    line = QFrame()
    line.setObjectName("Divider")
    line.setFixedSize(1, 24)
    return line


def hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("Hint")
    label.setWordWrap(True)
    return label


class Relay(QObject):
    """Carries device events from worker threads to the GUI thread."""

    state = Signal(str, str)        # device type, state
    error = Signal(str, str)        # device type, message
    done = Signal(object, object)   # callback, result (run on GUI thread)


def row(*widgets, stretch_last: bool = False) -> QWidget:
    box = QWidget()
    layout = QHBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    for w in widgets:
        if w is None:
            layout.addStretch(1)
        else:
            layout.addWidget(w)
    return box


def column(*widgets) -> QWidget:
    box = QWidget()
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    for w in widgets:
        if w is None:
            layout.addStretch(1)
        else:
            layout.addWidget(w)
    return box


Callback = Callable[[], None]
