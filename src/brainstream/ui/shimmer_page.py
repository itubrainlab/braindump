"""The Shimmer3 GSR+'s page: skin conductance now and over the last
minute, the ear clip's pulse, and the unit's settings."""
from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QLabel, QPushButton, QSplitter,
                               QVBoxLayout, QWidget)

from ..devices.shimmer import (RANGE_SPAN, RATES, ShimmerDevice, find_ports,
                               pairing_help)
from . import icons, theme
from .device_page import DevicePage
from .forms import ConfigForm, hint, panel, row
from .polar_page import _plot

GSR_SECONDS = 60
PPG_SECONDS = 8


class ShimmerPage(DevicePage):
    def __init__(self, device: ShimmerDevice, parent=None):
        super().__init__(device, parent)

        side = QWidget()
        column = QVBoxLayout(side)
        self.value = QLabel("—")
        self.value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value.setStyleSheet(f"font-size: 52px; font-weight: 700; "
                                 f"color: {theme.BRAND};")
        unit = QLabel("µS skin conductance")
        unit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        unit.setObjectName("Hint")
        self.resistance = QLabel()
        self.resistance.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.range = QLabel()
        self.range.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.range.setObjectName("Hint")
        self.battery_label = QLabel()
        self.battery_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.battery_label.setObjectName("Hint")
        for w in (self.value, unit, self.resistance, self.range,
                  self.battery_label):
            column.addWidget(w)
        column.addStretch(1)

        gsr, self.gsr_item = _plot("Conductance (µS)", "Time (s)")
        self.gsr_item.setXRange(-GSR_SECONDS, 0, padding=0)
        self.gsr_curve = self.gsr_item.plot(pen=pg.mkPen(theme.BRAND,
                                                         width=1.5))
        ppg, self.ppg_item = _plot("PPG (mV)", "Time (s)")
        self.ppg_item.setXRange(-PPG_SECONDS, 0, padding=0)
        self.ppg_curve = self.ppg_item.plot(pen=pg.mkPen(theme.TEXT,
                                                         width=1.2))
        self.ppg_panel = panel("Pulse (ear clip)", ppg)
        self.ppg_panel.setVisible(device.config.ppg)

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(panel("Skin conductance", gsr))
        right.addWidget(self.ppg_panel)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(panel("GSR", side))
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 4)
        self.tabs.addTab(split, icons.device("shimmer"), "GSR")
        self._build_settings()
        self._idle()

    # -- settings --------------------------------------------------------------------

    def _build_settings(self):
        cfg = self.device.config
        unit = ConfigForm("Unit", cfg)
        self.port = QComboBox()
        self.port.setEditable(True)
        self.port.setMinimumWidth(260)
        self.port.setToolTip("The paired unit's serial port. Empty uses the "
                             "port named after a Shimmer, or the only "
                             "Bluetooth serial port.")
        self.port.lineEdit().setPlaceholderText("automatic")
        self.port.setEditText(cfg.port)
        self.port.lineEdit().editingFinished.connect(self._port_changed)
        self.port.activated.connect(self._port_picked)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._fill_ports)
        unit.form.addRow("Serial port", row(self.port, refresh, None))
        unit.form.addRow("", hint(pairing_help()))
        unit.boolean("simulate", "Simulated", "A simulated unit (no "
                     "hardware)")
        unit.number("latency_ms", "Bluetooth delay", 0, 500,
                    "The link's minimum delay, taken off the time stamps "
                    "when known. The clock offset is measured "
                    "continuously; only this fixed part cannot be.",
                    " ms", 0)
        unit.choice("sampling_rate", "Sampling rate",
                    {r: f"{r:g} Hz" for r in RATES})
        unit.boolean("ppg", "Ear clip", "Record the ear clip's pulse (PPG, "
                     "internal ADC A13)")
        unit.changed.connect(self._changed)
        self.add_settings_tab(unit, hint(
            "The real unit needs pyshimmer, an optional extra under the "
            "GPL-3.0: pip install 'brainstream[shimmer]'. The packaged app "
            "installs it on first launch."))
        self._fill_ports()

    def _fill_ports(self):
        current = self.port.currentText()
        self.port.clear()
        try:
            ports = find_ports()
        except Exception:  # noqa: BLE001
            ports = []
        for device, desc in ports:
            self.port.addItem(f"{device}", device)
            self.port.setItemData(self.port.count() - 1, desc,
                                  Qt.ItemDataRole.ToolTipRole)
        self.port.setEditText(current)

    def _port_picked(self, index: int):
        self.port.setEditText(self.port.itemData(index) or "")
        self._port_changed()

    def _port_changed(self):
        value = self.port.currentText().strip()
        if value != self.device.config.port:
            self.device.config.port = value
            self.config_changed.emit()

    def _changed(self, attr: str):
        if attr == "ppg":
            self.ppg_panel.setVisible(self.device.config.ppg)
        self.config_changed.emit()

    # -- live ------------------------------------------------------------------------

    def _idle(self):
        self.value.setText("—")
        self.resistance.setText("")
        self.range.setText("")
        self.battery_label.setText("")
        self.gsr_curve.setData([], [])
        self.ppg_curve.setData([], [])

    def set_state(self, state):
        super().set_state(state)
        if state != "running" and hasattr(self, "value"):
            self._idle()

    def refresh(self):
        d = self.device
        graph = d.graph
        if graph is None:
            return
        if d.latest and d.latest[0] == d.latest[0]:
            us, kohm, band = d.latest
            self.value.setText(f"{us:.2f}")
            self.resistance.setText(f"{kohm:.0f} kΩ")
            self.range.setText(f"range {band} ({RANGE_SPAN[band]})")
        battery = d.battery
        self.battery_label.setText(f"Battery {battery:.0f}%" if battery
                                   is not None else "")
        rate = float(d.config.sampling_rate)
        rows = graph.tap.latest(int(GSR_SECONDS * rate))
        if len(rows):
            t = rows[:, 0] - rows[-1, 0]
            self.gsr_curve.setData(t, rows[:, 1])
            if d.config.ppg and rows.shape[1] > 4:
                keep = t > -PPG_SECONDS
                self.ppg_curve.setData(t[keep], rows[keep, 4])

    def status_icons(self):
        d = self.device
        if not d.running:
            return []
        tip = "Shimmer3 GSR+"
        if d.latest and d.latest[0] == d.latest[0]:
            tip += f": {d.latest[0]:.2f} µS"
        if d.battery is not None:
            tip += f", battery {d.battery:.0f}%"
        return [(icons.device("shimmer", theme.ICON_ACTIVE), tip)]
