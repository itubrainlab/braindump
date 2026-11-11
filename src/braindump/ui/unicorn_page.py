"""The Unicorn's page: the data viewer, signal quality and power
spectrum, and the headset's settings."""
from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QLabel, QPushButton,
                               QSplitter, QWidget)

from ..devices.unicorn import (EEG, GROUPS, HIGHPASS, LOWPASS, NUM_EEG,
                               HybridBlack,
                               NYQUIST, SAMPLING_RATE, UnicornDevice,
                               find_headsets)
from .. import presets
from . import icons, quality, theme
from . import power_spectrum as spectrum
from .scope import Scope
from .spectrum_view import SpectrumView
from .device_page import DevicePage
from .forms import ConfigForm, hint, panel, row, SourceToggle


class _Found(QObject):
    found = Signal(object)


class UnicornPage(DevicePage):
    def __init__(self, device: UnicornDevice, parent=None):
        self.quality = quality.SignalQuality()
        self._quality_seen = 0
        super().__init__(device, parent)
        cfg = device.config

        # View controls, in the page's header.
        self.scaling = self._combo(presets.SCALINGS, cfg.view_scaling)
        self.filter = self._combo(presets.FILTERS, cfg.view_filter)
        self.notch = self._combo(presets.NOTCHES, cfg.view_notch)
        self.spectrum_box = QCheckBox("Spectrum")
        self.spectrum_box.setChecked(cfg.view_spectrum)
        for text, widget in (("Scaling", self.scaling),
                             ("Filter", self.filter),
                             ("Notch", self.notch)):
            label = QLabel(text)
            label.setObjectName("Hint")
            self.controls.addWidget(label)
            self.controls.addWidget(widget)
        self.controls.addWidget(self.spectrum_box)
        self.scaling.currentTextChanged.connect(self._on_scaling)
        self.filter.currentTextChanged.connect(self._on_filter)
        self.notch.currentTextChanged.connect(self._on_notch)
        self.spectrum_box.toggled.connect(self._on_spectrum)

        # The viewer
        self.scope = Scope()
        self.scope.set_scale(presets.SCALINGS.get(cfg.view_scaling, 50.0))
        self._name_channels()
        self.scope_source = SourceToggle(
            "Filtered: the EEG after the view's filter and notch (which are "
            "not recorded). Raw: as acquired, before the recording filter "
            "(see Settings).")
        self.spectrum = SpectrumView()
        self.spectrum_source = SourceToggle(
            "Filtered or raw EEG, averaged across the electrodes.")
        self.spectrum_panel = panel("Power Spectrum", self.spectrum,
                                    self.spectrum_source)
        self.spectrum_panel.setVisible(cfg.view_spectrum)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(panel("Data Viewer", self.scope,
                                 self.scope_source))
        splitter.addWidget(self.spectrum_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.tabs.addTab(splitter, icons.wave(), "Data Viewer")
        self._build_settings()

    @staticmethod
    def _combo(items, current: str) -> QComboBox:
        combo = QComboBox()
        combo.addItems(list(items))
        combo.setCurrentText(current)
        combo.setMinimumWidth(110)
        return combo

    def _name_channels(self):
        names = self.device.config.electrode_names()
        self.scope.plot_item.getAxis("left").setTicks([[
            (NUM_EEG - i - 0.5, name) for i, name in enumerate(names)]
            + [(-0.5, "TRIG")]])

    # -- settings ----------------------------------------------------------------

    def _build_settings(self):
        cfg = self.device.config
        head = ConfigForm("Headset", cfg)
        self.serial = head.line(
            "serial", "Serial", "The headset to open, e.g. UN-2024.10.18. "
            "Empty opens the first one found.", "first found")
        self.find = QPushButton("Search…")
        self.find.setToolTip("Look for headsets in range (paired ones on "
                             "Windows). Takes a few seconds.")
        self.found = QComboBox()
        self.found.setVisible(False)
        self.found.activated.connect(
            lambda i: (self.serial.setText(self.found.itemText(i)),
                       head._set("serial", self.found.itemText(i))))
        self.find.clicked.connect(self._search)
        head.form.addRow("", row(self.find, self.found, None))
        head.boolean("simulate", "Simulated", "Use a simulated headset "
                     "(no hardware)")
        head.number("latency_ms", "Bluetooth delay", 0, 500,
                    "Taken off the time stamps, so the EEG lines up with "
                    "the other devices (g.tec measured "
                    f"{HybridBlack.DEVICE_DELAY_MS} ms).", " ms", 0)
        head.boolean("test_signal", "Test signal",
                     "The headset's internal square wave instead of the "
                     "electrodes")

        channels = ConfigForm("Channels", cfg)
        channels.words("electrodes", "Electrodes",
                       "Names of EEG 1-8, as written into the BDF and "
                       "channels.tsv")
        for name, text in GROUPS.items():
            channels.boolean(name, "" if name != "accelerometer" else
                             "Also record", text)

        filters = ConfigForm("Recording filters", cfg)
        filters.choice(
            "record_highpass", "High-pass",
            {f: "None" if f == 0 else f"{f:g} Hz" for f in HIGHPASS},
            "Applied to the EEG channels as they are recorded.")
        filters.choice(
            "record_lowpass", "Low-pass",
            {f: f"{f:g} Hz (half the sampling rate: none)" if f >= NYQUIST
             else f"{f:g} Hz" for f in LOWPASS},
            "Applied to the EEG channels as they are recorded.")
        filters.choice("record_notch", "Notch",
                       {name: name for name in presets.NOTCHES},
                       "Removes the power line (and its harmonics, when "
                       "cascading) from the recorded EEG.")
        filters.form.addRow("", hint(
            "4th-order Butterworth filters and IIR notches, causal, as in "
            "Unicorn Recorder, described in the recording's eeg.json and "
            "channels.tsv. The Filter and Notch boxes above the data viewer "
            "change only what is shown, not what is recorded."))

        bids = ConfigForm("BIDS description", cfg)
        bids.line("reference", "Reference", "EEGReference")
        bids.line("ground", "Ground", "EEGGround")
        bids.line("placement", "Placement scheme", "EEGPlacementScheme")

        trigger = ConfigForm("Trigger input", cfg)
        trigger.integer("trigger_port", "UDP port", 0, 65535,
                        "Numbers sent here land in the TRIG channel and "
                        "events.tsv; 0 is off.", special="off")
        trigger.line("trigger_ip", "Listen on")
        for form in (head, channels, filters, bids, trigger):
            form.changed.connect(self._changed)
        self.add_settings_tab(head, channels, filters, bids, trigger)

    def _changed(self, attr: str):
        if attr == "electrodes":
            self._name_channels()
        self.config_changed.emit()

    def _search(self):
        self.find.setEnabled(False)
        self.find.setText("Searching…")
        relay = _Found(self)
        relay.found.connect(self._found)
        self._relay = relay

        def work():
            try:
                result = find_headsets(rescan=True)
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
            self.found.addItem("No headsets found")
        else:
            self.found.addItems(result)
        self.found.setVisible(True)

    # -- view controls -------------------------------------------------------------

    def _on_scaling(self, name):
        self.device.config.view_scaling = name
        self.scope.set_scale(presets.SCALINGS[name])
        self.config_changed.emit()

    def _on_filter(self, name):
        self.device.set_view_filter(name)
        self.config_changed.emit()

    def _on_notch(self, name):
        self.device.set_view_notch(name)
        self.config_changed.emit()

    def _on_spectrum(self, on):
        self.device.config.view_spectrum = on
        self.spectrum_panel.setVisible(on)
        self.config_changed.emit()

    # -- live ------------------------------------------------------------------------

    def set_state(self, state):
        super().set_state(state)
        if state != "running" and hasattr(self, "scope"):
            self.scope.clear()
            self.spectrum.clear()
            self.scope.set_quality([quality.UNKNOWN] * NUM_EEG)
            self.quality.reset()
            self._quality_seen = 0

    def refresh(self):
        graph = self.device.graph
        if graph is None:
            return
        raw = self.scope_source.is_raw()
        buffer, total = (graph.raw if raw else graph.scope).sweep()
        filled = min(total, buffer.shape[0])
        if raw and filled:
            buffer[:, EEG] -= buffer[:filled, EEG].mean(axis=0)
        self.scope.show_sweep(buffer, total, EEG, graph.trigger)
        # Signal quality, from the raw EEG not yet judged.
        seen = graph.raw.total
        new = seen - self._quality_seen
        if new > 0:
            rows = graph.raw.latest(min(new, graph.raw.capacity))
            self.quality.update(rows[:, EEG])
            self._quality_seen = seen
            self.scope.set_quality(self.quality.verdicts())
        if self.spectrum_panel.isVisible():
            tap = graph.raw if self.spectrum_source.is_raw() else graph.scope
            samples = int(spectrum.WINDOW_S * SAMPLING_RATE)
            self.spectrum.show_eeg(tap.latest(samples)[:, EEG])

    def status_icons(self):
        d = self.device
        if not d.running:
            return []
        battery = d.battery()
        lost = d.lost_scans
        return [(icons.battery(battery),
                 f"Unicorn battery: {battery:.0f}%" if battery is not None
                 else "Unicorn battery"),
                (icons.wifi(theme.WARNING if lost else theme.ICON_ACTIVE),
                 f"Unicorn wireless data loss: {lost} samples")]
