"""The g.HIamp's page: a multi-channel data viewer (a page of channels at
a time, as a high-density cap has too many to show at once) and the
amplifier's settings."""
from __future__ import annotations

import threading

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (QComboBox, QLabel, QPushButton, QVBoxLayout,
                               QWidget)

from ..devices.hiamp import (NO_FILTER, RATES, HIampDevice,
                             describe_amplifier_filter)
from .. import presets
from . import icons, theme
from .device_page import DevicePage
from .forms import ConfigForm, hint, panel, row, SourceToggle

SECONDS = 10
PAGE_SIZES = (8, 16, 32, 64)
#: Points drawn per trace, at most: enough for the eye, light to draw.
MAX_POINTS = 2000


class MultiScope(QWidget):
    """Stacked EEG traces of one page of channels, sweeping like the
    Unicorn's, with the trigger channel's onsets marked."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot = pg.PlotWidget()
        self.plot.setBackground(theme.WINDOW)
        item = self.plot.getPlotItem()
        item.setMenuEnabled(False)
        item.hideButtons()
        item.setMouseEnabled(False, False)
        item.showGrid(x=True, y=False, alpha=0.3)
        item.setXRange(0, SECONDS, padding=0)
        item.setLabel("bottom", "Time (s)", color=theme.MUTED)
        for name in ("left", "bottom"):
            item.getAxis(name).setPen(pg.mkPen(theme.MUTED))
            item.getAxis(name).setTextPen(pg.mkPen(theme.TEXT))
        item.getAxis("left").setWidth(56)
        self.item = item
        self.curves: list = []
        self.cursor = pg.InfiniteLine(0, angle=90,
                                      pen=pg.mkPen(theme.HIGHLIGHT, width=2))
        item.addItem(self.cursor)
        self.marks: list = []
        self.scale_uv = 50.0
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self.plot)
        self._names: list[str] = []

    def set_channels(self, names: list[str]) -> None:
        if names == self._names:
            return
        self._names = list(names)
        for curve in self.curves:
            self.item.removeItem(curve)
        pen = pg.mkPen(theme.TEXT, width=1)
        self.curves = [self.item.plot(pen=pen, connect="finite")
                       for _ in names]
        n = len(names)
        self.item.setYRange(-0.5, n, padding=0)
        self.item.getAxis("left").setTicks(
            [[(n - i - 0.5, name) for i, name in enumerate(names)]])

    def clear(self) -> None:
        for curve in self.curves:
            curve.setData([], [])
        for mark in self.marks:
            self.item.removeItem(mark)
        self.marks = []

    def show(self, buffer: np.ndarray, total: int, fs: float,
             columns: list[int], trigger) -> None:
        capacity = buffer.shape[0]
        if total == 0 or not columns:
            self.clear()
            return
        cursor = total % capacity
        filled = min(total, capacity)
        step = max(1, capacity // MAX_POINTS)
        x = np.arange(0, capacity, step) / fs
        n = len(columns)
        for row, (curve, column) in enumerate(zip(self.curves, columns)):
            y = buffer[::step, column] / (2 * self.scale_uv)
            y = (n - row - 0.5) + np.clip(y, -0.5, 0.5)
            if filled < capacity:
                y[filled // step:] = np.nan
            y[min(len(y) - 1, cursor // step)] = np.nan
            curve.setData(x, y)
        self.cursor.setValue(cursor / fs)
        for mark in self.marks:
            self.item.removeItem(mark)
        self.marks = []
        if trigger is not None:
            values = buffer[:filled, trigger]
            previous = np.concatenate(([0.0], values[:-1]))
            for slot in np.flatnonzero((values != 0) & (values != previous)):
                mark = pg.InfiniteLine(slot / fs, angle=90,
                                       pen=pg.mkPen(theme.WARNING, width=1))
                self.item.addItem(mark)
                self.marks.append(mark)


class _Loaded(QObject):
    loaded = Signal(object)


class HIampPage(DevicePage):
    def __init__(self, device: HIampDevice, parent=None):
        super().__init__(device, parent)
        cfg = device.config
        self.scaling = self._combo(presets.SCALINGS, cfg.view_scaling)
        self.filter = self._combo(presets.FILTERS, cfg.view_filter)
        self.notch = self._combo(presets.NOTCHES, cfg.view_notch)
        self.page_size = QComboBox()
        for n in PAGE_SIZES:
            self.page_size.addItem(f"{n} per page", n)
        self.page_size.setCurrentIndex(1)
        self.page = QComboBox()
        self.page.setMinimumWidth(110)
        for text, widget in (("Channels", self.page),
                             ("", self.page_size),
                             ("Scaling", self.scaling),
                             ("Filter", self.filter),
                             ("Notch", self.notch)):
            if text:
                label = QLabel(text)
                label.setObjectName("Hint")
                self.controls.addWidget(label)
            self.controls.addWidget(widget)
        self.scaling.currentTextChanged.connect(self._on_scaling)
        self.filter.currentTextChanged.connect(self._on_filter)
        self.notch.currentTextChanged.connect(self._on_notch)
        self.page_size.currentIndexChanged.connect(lambda _: self._pages())

        self.scope = MultiScope()
        self.scope.scale_uv = presets.SCALINGS.get(cfg.view_scaling, 50.0)
        self.source = SourceToggle(
            "Filtered: after the view's filter and notch (which are not "
            "recorded). Raw: as acquired and recorded, through the "
            "amplifier's filters (see Settings).")
        self.tabs.addTab(panel("Data Viewer", self.scope, self.source),
                         icons.wave(), "Data Viewer")
        self._build_settings()
        self._pages()

    @staticmethod
    def _combo(items, current: str) -> QComboBox:
        combo = QComboBox()
        combo.addItems(list(items))
        combo.setCurrentText(current)
        combo.setMinimumWidth(100)
        return combo

    def _build_settings(self):
        cfg = self.device.config
        amp = ConfigForm("Amplifier", cfg)
        amp.line("serial", "Serial", "The amplifier to open, e.g. "
                 "HA-2019.06.12. Empty opens the first one found.",
                 "first found")
        amp.boolean("simulate", "Simulated", "A simulated amplifier (no "
                    "hardware)")
        amp.choice("sampling_rate", "Sampling rate",
                   {r: f"{r} Hz" for r in RATES})
        amp.integer("channel_count", "Channels", 0, 256,
                    "Channels to record, from the first; 0 records every "
                    "channel the amplifier has.", special="all")
        amp.boolean("digital_input", "Digital input", "Record the trigger "
                    "input (TRIG); its rising edges become events")
        amp.number("latency_ms", "USB delay", 0, 200,
                   "Taken off the time stamps; the clock offset and rate "
                   "are measured, only this fixed part cannot be.", " ms",
                   0)
        cap = ConfigForm("Cap and montage", cfg)
        cap.words("electrodes", "Electrodes",
                  "Names of the channels in order, comma-separated (e.g. "
                  "Fp1, Fp2, F7 ...); the rest are Ch<n>")
        cap.words("channel_types", "Other channels",
                  "Channels that are not EEG, as number=type, e.g. 64=EOG "
                  "for a bipolar EOG lead on channel 64 (types: EOG, ECG, "
                  "EMG, MISC). Written into channels.tsv and eeg.json.")
        cap.line("reference", "Reference", "EEGReference")
        cap.line("ground", "Ground", "EEGGround")
        cap.line("placement", "Placement scheme", "EEGPlacementScheme")
        cap.line("cap_manufacturer", "Cap manufacturer")
        cap.line("cap_model", "Cap model")
        hardware = ConfigForm("Amplifier filters", cfg)
        self.amp_bandpass = hardware.choice(
            "amp_bandpass", "Band-pass", {NO_FILTER: "None"},
            "The g.HIamp's own filter, applied in the amplifier to every "
            "channel.")
        self.amp_notch = hardware.choice(
            "amp_notch", "Notch", {NO_FILTER: "None"},
            "The g.HIamp's own notch, applied in the amplifier to every "
            "channel.")
        self.load = QPushButton("Load from amplifier")
        self.load.setToolTip(
            "Ask the amplifier which filters it offers at the sampling "
            "rate. When it is not connected, it is opened for a moment, "
            "without recording.")
        self.load.clicked.connect(self._load)
        hardware.form.addRow("", row(self.load, None))
        self.amp_note = hint("")
        hardware.form.addRow("", self.amp_note)
        hardware.form.addRow("", hint(
            "Applied in the amplifier to the recorded EEG, and described in "
            "the recording's eeg.json (HardwareFilters) and channels.tsv. "
            "The Filter and Notch boxes above the data viewer change only "
            "what is shown."))
        self._forms = (amp, cap, hardware)
        self._refill_filters()
        for form in self._forms:
            form.changed.connect(self._changed)
        self.add_settings_tab(amp, cap, hardware, hint(
            "The g.HIamp needs g.tec's gtec-gds driver, which brAIn stream "
            "installs with g.Pype (Windows, Linux and macOS)."))

    @staticmethod
    def _refill(combo: QComboBox, options: dict, current) -> bool:
        """Fills ``combo`` with ``options`` (value -> text), keeping
        ``current`` if it is one; True if it had to fall back to the
        first."""
        combo.blockSignals(True)
        combo.clear()
        for value, text in options.items():
            combo.addItem(text, value)
        index = combo.findData(current)
        combo.setCurrentIndex(max(0, index))
        combo.blockSignals(False)
        return index < 0

    def _refill_filters(self) -> None:
        """The amplifier's filters at the sampling rate, as last loaded."""
        cfg = self.device.config
        fs = float(cfg.sampling_rate)
        changed = False
        known = False
        for kind, combo, attr in (("bandpass", self.amp_bandpass,
                                   "amp_bandpass"),
                                  ("notch", self.amp_notch, "amp_notch")):
            offered = cfg.known_filters(kind)
            known = known or bool(offered)
            options = {NO_FILTER: "None"}
            options.update({f["index"]: describe_amplifier_filter(
                f, notch=kind == "notch") for f in offered})
            if self._refill(combo, options, int(getattr(cfg, attr))):
                setattr(cfg, attr, NO_FILTER)
                changed = True
        self.amp_note.setText(
            f"The filters the amplifier offers at {fs:g} Hz." if known else
            f"Load the filters the amplifier offers at {fs:g} Hz (or "
            "connect it once).")
        if changed:
            self.config_changed.emit()

    def _load(self):
        self.load.setEnabled(False)
        self.load.setText("Loading…")
        relay = _Loaded(self)
        relay.loaded.connect(self._loaded)
        self._relay = relay
        device = self.device

        def work():
            try:
                result = device.load_filters()
            except Exception as error:  # noqa: BLE001
                result = error
            relay.loaded.emit(result)
        threading.Thread(target=work, daemon=True).start()

    def _loaded(self, result):
        self.load.setEnabled(True)
        self.load.setText("Load from amplifier")
        self._refill_filters()
        if isinstance(result, Exception):
            self.amp_note.setText(f"Could not load the filters: {result}")
            self.amp_note.setStyleSheet(f"color: {theme.WARNING}")
            return
        self.amp_note.setStyleSheet("")
        counts = ", ".join(f"{len(v)} {'band-pass' if k == 'bandpass' else k}"
                           for k, v in result.items())
        self.amp_note.setText(
            f"Loaded from the amplifier: {counts} filters at "
            f"{float(self.device.config.sampling_rate):g} Hz.")
        self.config_changed.emit()

    def _changed(self, attr: str):
        if attr == "electrodes":
            self._pages()
        if attr == "sampling_rate":
            self._refill_filters()
        self.config_changed.emit()

    def _pages(self):
        names = self.device.channel_names()
        size = self.page_size.currentData() or 16
        current = self.page.currentIndex()
        self.page.blockSignals(True)
        self.page.clear()
        for start in range(0, len(names), size):
            end = min(start + size, len(names))
            self.page.addItem(f"{names[start]} – {names[end - 1]}", start)
        self.page.setCurrentIndex(max(0, min(current, self.page.count() - 1)))
        self.page.blockSignals(False)

    def _on_scaling(self, name):
        self.device.config.view_scaling = name
        self.scope.scale_uv = presets.SCALINGS[name]
        self.config_changed.emit()

    def _on_filter(self, name):
        self.device.set_view_filter(name)
        self.config_changed.emit()

    def _on_notch(self, name):
        self.device.set_view_notch(name)
        self.config_changed.emit()

    def set_state(self, state):
        super().set_state(state)
        if hasattr(self, "amp_note") and state == "running":
            # Connecting lists the amplifier's filters (and saves them).
            self._refill_filters()
            self.config_changed.emit()
        if hasattr(self, "scope"):
            self._pages()
            if state != "running":
                self.scope.clear()

    def refresh(self):
        d = self.device
        graph = d.graph
        if graph is None:
            return
        raw = self.source.is_raw()
        tap = graph.raw if raw else graph.scope
        buffer, total = tap.sweep()
        columns = d.signal_columns()
        names = d.channel_names()
        if self.page.count() == 0:
            self._pages()
        start = self.page.currentData() or 0
        size = self.page_size.currentData() or 16
        shown = columns[start:start + size]
        self.scope.set_channels(names[start:start + len(shown)])
        filled = min(total, buffer.shape[0])
        if raw and filled and shown:
            buffer[:, shown] -= buffer[:filled, shown].mean(axis=0)
        self.scope.show(buffer, total, d.rate, shown, d.trigger_column())
