"""The LSL page: which streams to record, the streams on the network, and
the events as they arrive."""
from __future__ import annotations

import threading
from datetime import datetime

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QHeaderView, QPushButton,
                               QSplitter, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from ..devices.lsl import LslDevice, find_streams
from . import icons, theme
from .device_page import DevicePage
from .forms import ConfigForm, hint, panel, row

FIELDS = ("name", "type", "source_id")


class _Found(QObject):
    found = Signal(object)


def _table(headers: list[str]) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setSectionResizeMode(
        QHeaderView.ResizeMode.Stretch)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    return table


class LslPage(DevicePage):
    def __init__(self, device: LslDevice, parent=None):
        super().__init__(device, parent)
        self._shown_events = 0
        self._filling = False

        # Streams to record
        box = QWidget()
        column = QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(hint(
            "Every stream matching a row is recorded; an empty field "
            "matches anything. Streams are found whenever they appear, "
            "also while recording."))
        self.selectors = _table(["Name", "Type", "Source id", "Connected"])
        self.selectors.itemChanged.connect(self._edited)
        column.addWidget(self.selectors, 1)
        add = QPushButton(icons.plus(), "Add")
        add.clicked.connect(lambda: self._add({"name": "", "type": "",
                                               "source_id": ""}))
        self.remove = QPushButton("Remove")
        self.remove.clicked.connect(self._remove)
        column.addWidget(row(add, self.remove, None))

        # Streams on the network
        net = QWidget()
        column2 = QVBoxLayout(net)
        column2.setContentsMargins(0, 0, 0, 0)
        self.network = _table(["Name", "Type", "Channels", "Rate", "Format",
                               "Host"])
        self.network.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.network.itemDoubleClicked.connect(lambda _: self._use())
        column2.addWidget(self.network, 1)
        self.search = QPushButton("Search the network")
        self.search.clicked.connect(self._search)
        use = QPushButton(icons.plus(), "Record this stream")
        use.clicked.connect(self._use)
        column2.addWidget(row(self.search, use, None))
        self._found: list[dict] = []

        # Events
        self.events = _table(["Time", "Stream", "Value", ""])
        self.events.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.events.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents)

        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(panel("Streams to record", box))
        top.addWidget(panel("On the network", net))
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(top)
        split.addWidget(panel("Events", self.events))
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        self.tabs.addTab(split, icons.device("lsl"), "Streams")

        form = ConfigForm("Lab Streaming Layer", device.config)
        form.boolean("simulate", "Simulated", "Send and record a test "
                     "marker stream, every second")
        form.number("resolve_timeout", "Search time", 0.2, 10,
                    "Seconds each search for streams waits for answers",
                    " s", 1, step=0.5)
        form.changed.connect(lambda _: self.config_changed.emit())
        self.add_settings_tab(form)
        self._fill()

    # -- streams to record ---------------------------------------------------------

    def _fill(self):
        self._filling = True
        self.selectors.setRowCount(0)
        for selector in self.device.config.streams:
            r = self.selectors.rowCount()
            self.selectors.insertRow(r)
            for c, key in enumerate(FIELDS):
                self.selectors.setItem(
                    r, c, QTableWidgetItem(str(selector.get(key, ""))))
            item = QTableWidgetItem("")
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.selectors.setItem(r, 3, item)
        self._filling = False
        self._connected()

    def _edited(self, _item):
        if self._filling:
            return
        streams = []
        for r in range(self.selectors.rowCount()):
            streams.append({key: (self.selectors.item(r, c).text().strip()
                                  if self.selectors.item(r, c) else "")
                            for c, key in enumerate(FIELDS)})
        self.device.config.streams = streams
        self.config_changed.emit()

    def _add(self, selector: dict):
        if selector in self.device.config.streams:
            return
        self.device.config.streams = self.device.config.streams + [selector]
        self.config_changed.emit()
        self._fill()

    def _remove(self):
        rows = sorted({i.row() for i in self.selectors.selectedIndexes()},
                      reverse=True)
        streams = list(self.device.config.streams)
        for r in rows:
            if 0 <= r < len(streams):
                streams.pop(r)
        self.device.config.streams = streams
        self.config_changed.emit()
        self._fill()

    def _connected(self):
        from ..devices.lsl import matches
        inlets = list(self.device.inlets.values())
        self._filling = True
        for r, selector in enumerate(self.device.config.streams):
            item = self.selectors.item(r, 3)
            if item is None:
                continue
            names = [i.name for i in inlets if matches(selector, i.info)]
            item.setText(", ".join(names) if names else
                         ("waiting…" if self.device.running else ""))
            item.setForeground(QColor(theme.GOOD if names else theme.MUTED))
        self._filling = False

    # -- the network ---------------------------------------------------------------------

    def _search(self):
        self.search.setEnabled(False)
        self.search.setText("Searching…")
        relay = _Found(self)
        relay.found.connect(self._show_found)
        self._relay = relay
        timeout = max(1.0, self.device.config.resolve_timeout)

        def work():
            try:
                result = find_streams(timeout)
            except Exception as error:  # noqa: BLE001
                result = error
            relay.found.emit(result)
        threading.Thread(target=work, daemon=True).start()

    def _show_found(self, result):
        self.search.setEnabled(True)
        self.search.setText("Search again")
        self.network.setRowCount(0)
        if isinstance(result, Exception):
            self.network.insertRow(0)
            self.network.setItem(0, 0, QTableWidgetItem(
                f"Search failed: {result}"))
            self._found = []
            return
        self._found = result
        for s in result:
            r = self.network.rowCount()
            self.network.insertRow(r)
            rate = "irregular" if not s["rate"] else f"{s['rate']:g} Hz"
            for c, text in enumerate((s["name"], s["type"],
                                      str(s["channels"]), rate, s["format"],
                                      s["host"])):
                item = QTableWidgetItem(text)
                item.setToolTip(f"source id: {s['source_id'] or '—'}")
                self.network.setItem(r, c, item)
        if not result:
            self.network.insertRow(0)
            self.network.setItem(0, 0, QTableWidgetItem("No streams found"))

    def _use(self):
        r = self.network.currentRow()
        if not 0 <= r < len(self._found):
            return
        s = self._found[r]
        self._add({"name": s["name"], "type": s["type"],
                   "source_id": s["source_id"]})

    # -- live ------------------------------------------------------------------------------

    def set_state(self, state):
        super().set_state(state)
        if hasattr(self, "selectors"):
            self._connected()

    def refresh(self):
        d = self.device
        self._connected()
        if d.received == self._shown_events:
            return
        new = d.received - self._shown_events
        self._shown_events = d.received
        latest = list(d.log)[-min(new, 200):]
        for ev in latest:
            self.events.insertRow(0)
            when = datetime.fromtimestamp(ev.time).strftime("%H:%M:%S.%f")[:-3]
            recorded = d.recording
            for c, text in enumerate((when, ev.trial_type, ev.value,
                                      "● REC" if recorded else "")):
                item = QTableWidgetItem(text)
                if c == 3:
                    item.setForeground(QColor(theme.RECORDING))
                self.events.setItem(0, c, item)
        while self.events.rowCount() > 300:
            self.events.removeRow(self.events.rowCount() - 1)

    def status_icons(self):
        d = self.device
        if not d.running:
            return []
        n = len(d.inlets)
        return [(icons.device("lsl", theme.ICON_ACTIVE if n
                              else theme.WARNING),
                 f"LSL: {n} stream(s) connected, {d.received} samples")]
