"""The GP3's page: the eyes as the camera sees them, the gaze on the
screen, the pupils over time, calibration with its quality, and the
tracker's settings."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QComboBox, QGridLayout, QLabel, QMessageBox,
                               QPushButton, QVBoxLayout, QWidget)

from ..devices.gp3 import COL, MODES, GP3Device
from ..devices.gp3_backends import (CalibrationDriver, CalibrationFailed,
                                    CalibrationQuality, EyeView)
from . import icons, theme
from .calibration_window import CalibrationWindow, screen_for
from .device_page import DevicePage
from .eye_widgets import EYE_COLOURS, EyeViewWidget, GazeScreen, QualityPlot
from .forms import ConfigForm, hint, panel, row

POINTS = {5: "5 points", 9: "9 points", 13: "13 points", 16: "16 points",
          25: "25 points"}
RATING = {"good": theme.GOOD, "fair": theme.WARNING, "poor": theme.BAD}


def _displays() -> dict[int, str]:
    try:
        from opengp3.platform import displays
        found = displays()
    except Exception:  # noqa: BLE001
        found = []
    if not found:
        return {0: "Main display"}
    return {i: f"{i}: {int(d.width)}×{int(d.height)}"
               f"{' (main)' if d.main else ''}"
            for i, d in enumerate(found)}


class GP3Page(DevicePage):
    def __init__(self, device: GP3Device, parent=None):
        super().__init__(device, parent)
        self.window_: Optional[CalibrationWindow] = None
        self.remote_run: Optional[CalibrationDriver] = None
        self.remote_timer = QTimer(self)
        self.remote_timer.setInterval(100)
        self.remote_timer.timeout.connect(self._poll_remote)

        grid = QWidget()
        layout = QGridLayout(grid)
        layout.setContentsMargins(0, 0, 0, 0)
        self.eyes = EyeViewWidget()
        self.eyes.mirror = device.config.mirror_view
        layout.addWidget(panel("Eyes", self.eyes), 0, 0)
        layout.addWidget(self._calibration_panel(), 0, 1)
        self.gaze = GazeScreen()
        layout.addWidget(panel("Gaze on screen", self.gaze), 1, 0)
        layout.addWidget(panel("Pupils", self._pupil_plot()), 1, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(0, 3)
        layout.setRowStretch(1, 2)
        self.tabs.addTab(grid, icons.eye(), "Eye Tracker")
        self._build_settings()
        self._show_quality()

    # -- panels ------------------------------------------------------------------

    def _calibration_panel(self) -> QWidget:
        cfg = self.device.config
        box = QWidget()
        column = QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        self.points = QComboBox()
        for n, text in POINTS.items():
            self.points.addItem(text, n)
        self.points.setCurrentIndex(max(0, self.points.findData(
            cfg.calibration_points)))
        self.points.currentIndexChanged.connect(self._points_changed)
        self.calibrate = QPushButton(icons.target(), "Calibrate")
        self.calibrate.setObjectName("Primary")
        self.calibrate.clicked.connect(self._calibrate)
        self.validate = QPushButton("Validate")
        self.validate.setToolTip("Measure accuracy, in degrees, on 8 "
                                 "targets the calibration did not use")
        self.validate.clicked.connect(self._validate)
        self.cancel = QPushButton("Cancel")
        self.cancel.clicked.connect(self._cancel_remote)
        self.cancel.setVisible(False)
        column.addWidget(row(self.points, self.calibrate, self.validate,
                             self.cancel, None))
        self.where = hint("")
        column.addWidget(self.where)
        self.badge = QLabel()
        self.badge.setObjectName("QualityBadge")
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        column.addWidget(row(self.badge, self.summary, None))
        self.validation = hint("")
        column.addWidget(self.validation)
        self.plot = QualityPlot()
        column.addWidget(self.plot, 1)
        return panel("Calibration", box)

    def _pupil_plot(self) -> QWidget:
        plot = pg.PlotWidget()
        plot.setBackground(theme.WINDOW)
        item = plot.getPlotItem()
        item.setMenuEnabled(False)
        item.hideButtons()
        item.setMouseEnabled(False, False)
        item.showGrid(x=True, y=True, alpha=0.2)
        item.setLabel("left", "Pupil (mm)", color=theme.MUTED)
        item.setLabel("bottom", "Time (s)", color=theme.MUTED)
        item.setYRange(1.5, 7.5)
        for name in ("left", "bottom"):
            item.getAxis(name).setPen(pg.mkPen(theme.MUTED))
            item.getAxis(name).setTextPen(pg.mkPen(theme.TEXT))
        self.pupil_curves = {
            side: item.plot(pen=pg.mkPen(EYE_COLOURS[side], width=1.5),
                            connect="finite", name=side)
            for side in ("L", "R")}
        self.pupil_item = item
        return plot

    # -- settings ---------------------------------------------------------------------

    def _build_settings(self):
        cfg = self.device.config
        conn = ConfigForm("Connection", cfg)
        conn.choice("mode", "GP3", MODES,
                    "On this computer, OpenGP3 drives the camera itself "
                    "(the GP3 or the GP3 HD; Windows needs UsbDk or "
                    "WinUSB, see OpenGP3). On another computer, it runs "
                    "opengp3 server or Gazepoint "
                    "Control, and brAIn dump reads it over the network.")
        conn.boolean("simulate", "Simulated", "No hardware: a simulated "
                     "eye tracker")
        conn.line("host", "Server", "The computer serving the Open Gaze "
                  "API", "e.g. 192.168.1.20")
        conn.integer("port", "Port", 1, 65535, "Open Gaze API port "
                     "(Gazepoint's default 4242)")
        conn.choice("remote_calibration_display", "Calibration targets",
                    {"tracker": "On the GP3 computer's screen",
                     "here": "On this computer's screen"},
                    "Where the participant looks during calibration")
        conn.boolean("hide_remote_calibration", "",
                     "Close the server's calibration window when done")

        screen = ConfigForm("Screen", cfg)
        screen.choice("display", "Display", _displays(),
                      "The screen the participant looks at")
        screen.number("screen_width_cm", "Screen width", 0, 300,
                      "Physical width, for accuracy in degrees. 0 asks the "
                      "operating system (or is unknown, remotely).", " cm",
                      1, special="automatic")
        screen.number("distance_cm", "Viewing distance", 30, 150,
                      suffix=" cm", decimals=0)
        screen.number("max_diagonal_in", "Calibration area", 0, 100,
                      "On larger screens, calibrate a centred area this "
                      "big (the GP3 is specified up to 22\").", "\"", 0,
                      special="whole screen")

        tracking = ConfigForm("Tracking", cfg)
        tracking.number("calibration_delay", "Target travel", 0.2, 3,
                        "Seconds each target takes to move into place",
                        " s", 2, step=0.1)
        tracking.number("calibration_timeout", "Target collection", 0.5, 5,
                        "Seconds of data collected at each target", " s",
                        2, step=0.1)
        tracking.choice("hd_rate", "GP3 HD rate",
                        {150: "150 Hz", 60: "60 Hz"},
                        "The GP3 HD's frame rate. The GP3 always runs at "
                        "60 Hz.")
        tracking.choice("monocular", "Eyes", {"": "Both",
                                              "L": "Left eye only",
                                              "R": "Right eye only"})
        tracking.boolean("auto_gain", "", "Sweep the camera gain while no "
                         "eyes are found (very bright or dim pupils)")
        tracking.boolean("smoothing", "", "Smooth the gaze (One-Euro "
                         "filter); raw gaze is recorded as well")
        tracking.boolean("mirror_view", "", "Mirror the eye view, so the "
                         "participant's left is on the left")
        self.forms = (conn, screen, tracking)
        for form in self.forms:
            form.changed.connect(self._changed)
        self.add_settings_tab(*self.forms)
        self._mode_rows()

    def _mode_rows(self):
        conn, screen, tracking = self.forms
        cfg = self.device.config
        remote = cfg.mode == "remote" and not cfg.simulate
        for attr in ("host", "port", "remote_calibration_display",
                     "hide_remote_calibration"):
            conn.show_row(attr, remote)
        screen.show_row("display", not remote)
        screen.show_row("max_diagonal_in", not remote)
        for attr in ("calibration_delay", "calibration_timeout", "hd_rate",
                     "monocular", "auto_gain", "smoothing"):
            tracking.show_row(attr, not remote)

    def _changed(self, attr: str):
        if attr in ("mode", "simulate"):
            self._mode_rows()
        if attr == "mirror_view":
            self.eyes.mirror = self.device.config.mirror_view
        self.config_changed.emit()
        self._update_controls()

    def _points_changed(self):
        self.device.config.calibration_points = self.points.currentData()
        self.config_changed.emit()

    # -- calibration -----------------------------------------------------------------

    def _busy(self) -> bool:
        return self.window_ is not None or self.remote_run is not None

    def _update_controls(self):
        d = self.device
        running = d.running
        busy = self._busy()
        self.calibrate.setEnabled(running and not busy)
        backend = d.backend
        self.validate.setEnabled(running and not busy and backend is not None
                                 and backend.can_validate
                                 and bool(backend.calibrated))
        self.validate.setVisible(not (d.config.mode == "remote"
                                      and not d.simulated))
        self.points.setEnabled(not busy)
        cfg = d.config
        if d.simulated:
            where = "Simulated eye tracker."
        elif cfg.mode == "remote":
            model = backend.model if running and backend else "GP3"
            where = (f"{model} at {cfg.host}:{cfg.port}. Targets on "
                     + ("this computer's screen."
                        if cfg.remote_calibration_display == "here"
                        else "the eye tracker computer's screen."))
        else:
            where = (f"{backend.model}, {backend.rate:g} Hz, on this "
                     "computer." if running and backend else
                     "GP3 or GP3 HD on this computer.")
        if running and backend is not None and backend.calibrated is False:
            where += f" <span style='color:{theme.WARNING}'>Not calibrated." \
                     "</span>"
        self.where.setText(where)

    def _calibrate(self):
        backend = self.device.backend
        if backend is None:
            return
        try:
            driver = backend.calibrate(self.points.currentData())
        except CalibrationFailed as error:
            QMessageBox.warning(self, "Calibration", str(error))
            return
        except Exception as error:  # noqa: BLE001
            QMessageBox.critical(self, "Calibration", str(error))
            return
        self._run(driver)

    def _validate(self):
        backend = self.device.backend
        if backend is None:
            return
        try:
            driver = backend.validate()
        except CalibrationFailed as error:
            QMessageBox.warning(self, "Validation", str(error))
            return
        self._run(driver)

    def _run(self, driver: CalibrationDriver):
        cfg = self.device.config
        if not driver.shows_targets:
            self.remote_run = driver
            self.cancel.setVisible(True)
            self.remote_timer.start()
            self._update_controls()
            return
        geometry = None
        if cfg.mode == "local" and not self.device.simulated:
            try:
                from opengp3.platform import displays
                d = displays()[cfg.display]
                geometry = (d.x, d.y, d.width, d.height)
            except Exception:  # noqa: BLE001
                geometry = None
        backend = self.device.backend
        window = CalibrationWindow(
            driver, backend.eye_view, screen_for(cfg.display, geometry),
            mirror=cfg.mirror_view)
        window.finished.connect(self._finished)
        window.failed.connect(self._failed)
        self.window_ = window
        self._update_controls()

    def _poll_remote(self):
        run = self.remote_run
        if run is None:
            self.remote_timer.stop()
            return
        done, total = run.progress()
        self.summary.setText(f"Calibrating on the GP3 computer… point "
                             f"{min(done, total)} of {total}")
        if not run.tick():
            return
        self.remote_timer.stop()
        self.remote_run = None
        self.cancel.setVisible(False)
        try:
            quality = run.result()
        except Exception as error:  # noqa: BLE001
            self._failed(str(error))
            return
        self._finished(quality)

    def _cancel_remote(self):
        run, self.remote_run = self.remote_run, None
        self.remote_timer.stop()
        self.cancel.setVisible(False)
        if run is not None:
            run.cancel()
        self._show_quality()
        self._update_controls()

    def _finished(self, quality: CalibrationQuality):
        self.window_ = None
        self.device.calibration_done(quality)
        self.config_changed.emit()
        self._show_quality()
        self._update_controls()

    def _failed(self, message: str):
        self.window_ = None
        self._update_controls()
        self._show_quality()
        if message:
            QMessageBox.warning(self, "Calibration", message)

    def _show_quality(self):
        cal = self.device.last_calibration
        self.plot.show_quality(cal)
        if cal is None:
            self.badge.setText("")
            self.badge.setStyleSheet("")
            self.summary.setText("Not calibrated yet.")
        else:
            rating = cal.rating()
            self.badge.setText(rating.upper())
            self.badge.setStyleSheet(
                f"background:{RATING[rating]}; color:#111;")
            from datetime import datetime
            when = datetime.fromtimestamp(cal.created).strftime("%H:%M")
            current = "" if self.device.calibrations else \
                " (from an earlier session)"
            self.summary.setText(f"{cal.summary()} · {when}{current}")
        val = self.device.last_validation
        self.validation.setText(f"Validation: {val.summary()}"
                                if val else "")

    # -- live ---------------------------------------------------------------------------

    def set_state(self, state):
        super().set_state(state)
        if hasattr(self, "calibrate"):
            self._update_controls()
            if state != "running":
                self.eyes.show_view(EyeView(), connected=False)
                self.gaze.show_gaze(np.empty((0, 2)), None,
                                    self.gaze.screen_size)
                for curve in self.pupil_curves.values():
                    curve.setData([], [])
            self._show_quality()

    def refresh(self):
        d = self.device
        backend, graph = d.backend, d.graph
        if backend is None or graph is None:
            return
        self.eyes.show_view(backend.eye_view())
        rows = graph.tap.latest(600)
        if len(rows) == 0:
            return
        recent = rows[-30:]
        ok = recent[:, COL["BPOGV"]] > 0
        trail = recent[ok][:, [COL["BPOGX"], COL["BPOGY"]]]
        last = rows[-1]
        fixation = ((last[COL["FPOGX"]], last[COL["FPOGY"]])
                    if last[COL["FPOGV"]] > 0 else None)
        screen = (backend.screen.width, backend.screen.height)
        self.gaze.show_gaze(trail, fixation, screen)
        t = rows[:, COL["TIME"]]
        t = t - t[-1]
        for side, curve in self.pupil_curves.items():
            y = rows[:, COL[f"{side}PMM"]].astype(float)
            y[rows[:, COL[f"{side}PMMV"]] <= 0] = np.nan
            curve.setData(t, y)
        self.pupil_item.setXRange(-10, 0, padding=0)

    def status_icons(self):
        d = self.device
        if not d.running:
            return []
        backend = d.backend
        cal = backend.calibrated if backend else None
        colour = theme.GOOD if cal else theme.WARNING if cal is False \
            else theme.MUTED
        text = {True: "GP3 calibrated", False: "GP3 not calibrated",
                None: "GP3 calibration unknown"}[cal]
        return [(icons.eye(colour), text)]

    def shown(self):
        self._update_controls()
