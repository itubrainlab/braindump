"""The Polar H10's page: heart rate, skin contact and battery, the ECG as
it comes, and the heart rate beat by beat."""
from __future__ import annotations

import threading
import time

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (QComboBox, QLabel, QPushButton, QSplitter,
                               QVBoxLayout, QWidget)

from ..devices.polar import (ACC_RANGES, ACC_RATES, ECG_RATE, PolarDevice,
                             find_straps)
from . import icons, theme
from .device_page import DevicePage
from .forms import ConfigForm, hint, panel, row

ECG_SECONDS = 8
TREND_SECONDS = 180


class _Found(QObject):
    found = Signal(object)


def _plot(left: str, bottom: str) -> tuple[pg.PlotWidget, pg.PlotItem]:
    widget = pg.PlotWidget()
    widget.setBackground(theme.WINDOW)
    item = widget.getPlotItem()
    item.setMenuEnabled(False)
    item.hideButtons()
    item.setMouseEnabled(False, False)
    item.showGrid(x=True, y=True, alpha=0.2)
    item.setLabel("left", left, color=theme.MUTED)
    item.setLabel("bottom", bottom, color=theme.MUTED)
    for name in ("left", "bottom"):
        item.getAxis(name).setPen(pg.mkPen(theme.MUTED))
        item.getAxis(name).setTextPen(pg.mkPen(theme.TEXT))
    return widget, item


class PolarPage(DevicePage):
    def __init__(self, device: PolarDevice, parent=None):
        super().__init__(device, parent)

        # Heart rate, big, as a monitor shows it.
        side = QWidget()
        column = QVBoxLayout(side)
        self.bpm = QLabel("—")
        self.bpm.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bpm.setStyleSheet(f"font-size: 64px; font-weight: 700; "
                               f"color: {theme.BRAND};")
        unit = QLabel("beats per minute")
        unit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        unit.setObjectName("Hint")
        self.contact = QLabel()
        self.contact.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rr = QLabel()
        self.rr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rr.setObjectName("Hint")
        self.battery_label = QLabel()
        self.battery_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.battery_label.setObjectName("Hint")
        for w in (self.bpm, unit, self.contact, self.rr, self.battery_label):
            column.addWidget(w)
        column.addStretch(1)

        ecg, self.ecg_item = _plot("ECG (mV)", "Time (s)")
        self.ecg_item.setXRange(-ECG_SECONDS, 0, padding=0)
        self.ecg_curve = self.ecg_item.plot(pen=pg.mkPen(theme.TEXT,
                                                         width=1.2))
        trend, self.trend_item = _plot("Heart rate (bpm)", "Time (s)")
        self.trend_item.setXRange(-TREND_SECONDS, 0, padding=0)
        self.trend_curve = self.trend_item.plot(
            pen=pg.mkPen(theme.BRAND, width=1.5), symbol="o", symbolSize=3,
            symbolBrush=theme.BRAND, symbolPen=None)

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(panel("ECG", ecg))
        right.addWidget(panel("Heart rate, beat by beat", trend))
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(panel("Heart", side))
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 4)
        self.tabs.addTab(split, icons.heart(), "Heart")
        self._build_settings()
        self._idle()

    # -- settings --------------------------------------------------------------------

    def _build_settings(self):
        cfg = self.device.config
        strap = ConfigForm("Strap", cfg)
        self.name = strap.line(
            "device", "Strap", "Its name (e.g. Polar H10 A1B2C3D4, printed "
            "on the sensor's back as its ID) or Bluetooth address. Empty "
            "connects to the first Polar H10 found.", "first found")
        self.find = QPushButton("Search…")
        self.find.setToolTip("Look for Polar H10 straps in range: wear the "
                             "strap, it switches on with skin contact")
        self.found = QComboBox()
        self.found.setVisible(False)
        self.found.activated.connect(self._pick)
        self.find.clicked.connect(self._search)
        strap.form.addRow("", row(self.find, self.found, None))
        strap.boolean("simulate", "Simulated", "A simulated strap (no "
                      "hardware)")
        strap.number("latency_ms", "Bluetooth delay", 0, 500,
                     "The link's minimum delay, taken off the time stamps "
                     "when known. The clock offset is measured "
                     "continuously; only this fixed part cannot be.",
                     " ms", 0)
        acc = ConfigForm("Accelerometer", cfg)
        acc.boolean("accelerometer", "Record", "The strap's acceleration, "
                    "as recording-accel")
        acc.choice("acc_rate", "Rate", {r: f"{r} Hz" for r in ACC_RATES})
        acc.choice("acc_range", "Range", {r: f"±{r} g" for r in ACC_RANGES})
        for form in (strap, acc):
            form.changed.connect(lambda _: self.config_changed.emit())
        self.add_settings_tab(strap, acc, hint(
            f"The ECG ({ECG_RATE:g} Hz, µV), the heart rate and the RR "
            "intervals are always recorded. On macOS, brAIn dump asks once "
            "for Bluetooth access."))

    def _search(self):
        self.find.setEnabled(False)
        self.find.setText("Searching…")
        relay = _Found(self)
        relay.found.connect(self._found)
        self._relay = relay
        seconds = self.device.config.scan_seconds

        def work():
            try:
                result = find_straps(seconds)
            except Exception as error:  # noqa: BLE001
                result = error
            relay.found.emit(result)
        threading.Thread(target=work, daemon=True).start()

    def _found(self, result):
        self.find.setEnabled(True)
        self.find.setText("Search again…")
        self.found.clear()
        if isinstance(result, Exception):
            self.found.addItem(f"Search failed: {result}")
        elif not result:
            self.found.addItem("No Polar H10 found")
        else:
            for name, address in result:
                self.found.addItem(f"{name}  ({address})", name)
        self.found.setVisible(True)

    def _pick(self, index: int):
        name = self.found.itemData(index)
        if name:
            self.name.setText(name)
            self.device.config.device = name
            self.config_changed.emit()

    # -- live ----------------------------------------------------------------------------

    def _idle(self):
        self.bpm.setText("—")
        self.contact.setText("")
        self.rr.setText("")
        self.battery_label.setText("")
        self.ecg_curve.setData([], [])
        self.trend_curve.setData([], [])

    def set_state(self, state):
        super().set_state(state)
        if state != "running" and hasattr(self, "bpm"):
            self._idle()

    def refresh(self):
        d = self.device
        graph = d.graph
        if graph is None:
            return
        self.bpm.setText(str(d.heart_rate) if d.heart_rate else "—")
        if d.contact is False:
            self.contact.setText("No skin contact: moisten the electrodes")
            self.contact.setStyleSheet(f"color: {theme.WARNING}")
        elif d.contact:
            self.contact.setText("Skin contact")
            self.contact.setStyleSheet(f"color: {theme.GOOD}")
        else:
            self.contact.setText("")
        beats = list(d.beats)
        if beats:
            self.rr.setText(f"last RR {beats[-1][1] * 1000:.0f} ms")
        battery = d.battery
        self.battery_label.setText(f"Battery {battery}%" if battery
                                   is not None else "")
        rows = graph.ecg_tap.latest(int(ECG_SECONDS * ECG_RATE))
        if len(rows):
            t = rows[:, 0] - rows[-1, 0]
            self.ecg_curve.setData(t, rows[:, 1] / 1000.0)
        if beats:
            now = time.time()
            times = np.array([b[0] for b in beats]) - now
            keep = times > -TREND_SECONDS
            self.trend_curve.setData(times[keep],
                                     np.array([b[2] for b in beats])[keep])

    def status_icons(self):
        d = self.device
        if not d.running:
            return []
        colour = theme.WARNING if d.contact is False else theme.ICON_ACTIVE
        tip = f"Polar H10: {d.heart_rate or '—'} bpm"
        if d.battery is not None:
            tip += f", battery {d.battery}%"
        return [(icons.heart(colour), tip)]
