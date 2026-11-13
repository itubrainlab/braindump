"""The full-screen calibration (and validation) window.

It first shows the participant's eyes so they can settle in, then the
targets one by one as the driver times them, then the result. Space
starts, Esc cancels. The drawing is Qt's; the timing and the labelling
of samples are OpenGP3's (or the remote server's).
"""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..devices.gp3_backends import (CalibrationDriver, CalibrationFailed,
                                    CalibrationQuality, EyeView)
from . import theme
from .eye_widgets import EyeViewWidget, QualityPlot, eye_status

BACKGROUND = QColor(25, 25, 25)
RESULT_MS = 4000


def screen_for(display: int, geometry: Optional[tuple] = None):
    """The QScreen at ``geometry`` (x, y, w, h), else display number
    ``display``, else the primary one."""
    screens = QGuiApplication.screens()
    if geometry is not None:
        for s in screens:
            g = s.geometry()
            if (g.x(), g.y(), g.width(), g.height()) == tuple(
                    int(v) for v in geometry):
                return s
    if 0 <= display < len(screens):
        return screens[display]
    return QGuiApplication.primaryScreen()


class CalibrationWindow(QWidget):
    """Runs ``driver`` full screen; emits ``finished`` with the quality,
    or ``failed`` with a message (empty when cancelled)."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, driver: CalibrationDriver,
                 eye_view: Callable[[], EyeView], screen=None,
                 mirror: bool = True, wait_for_start: bool = True):
        super().__init__(None, Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle("brAIn stream calibration")
        self.setCursor(Qt.CursorShape.BlankCursor)
        self.driver = driver
        self.eye_view = eye_view
        self.phase = "position" if wait_for_start else "run"
        self.quality: Optional[CalibrationQuality] = None
        self._done = False
        self.eyes = EyeViewWidget(self)
        self.eyes.mirror = mirror
        self.eyes.setVisible(wait_for_start)
        self.plot = QualityPlot(self)
        self.plot.setVisible(False)
        self.timer = QTimer(self)
        self.timer.setInterval(15)
        self.timer.timeout.connect(self._tick)
        screen = screen or QGuiApplication.primaryScreen()
        self.setGeometry(screen.geometry())
        self.show()
        self.windowHandle().setScreen(screen)
        self.showFullScreen()
        self.activateWindow()
        self.raise_()
        self.setFocus()
        self.timer.start()

    # -- flow -------------------------------------------------------------------

    def _start(self):
        self.phase = "run"
        self.eyes.setVisible(False)
        self.update()

    def _tick(self):
        if self.phase == "position":
            self.eyes.show_view(self.eye_view())
        elif self.phase == "run":
            try:
                done = self.driver.tick()
            except Exception as error:  # noqa: BLE001
                self._fail(str(error))
                return
            if done:
                self._result()
        self.update()

    def _result(self):
        self.phase = "result"
        try:
            self.quality = self.driver.result()
        except CalibrationFailed as error:
            self._fail(str(error))
            return
        except Exception as error:  # noqa: BLE001
            self._fail(f"{type(error).__name__}: {error}")
            return
        self.plot.show_quality(self.quality)
        self.plot.setVisible(True)
        self._layout()
        QTimer.singleShot(RESULT_MS, self._close_ok)

    def _close_ok(self):
        if self._done:
            return
        self._done = True
        self.timer.stop()
        self.close()
        self.finished.emit(self.quality)

    def _fail(self, message: str):
        if self._done:
            return
        self._done = True
        self.timer.stop()
        if message == "":
            try:
                self.driver.cancel()
            except Exception:  # noqa: BLE001
                pass
        self.close()
        self.failed.emit(message)

    # -- input ---------------------------------------------------------------------

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Escape:
            if self.phase == "result":
                self._close_ok()
            else:
                self._fail("")
        elif key in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.phase == "position":
                self._start()
            elif self.phase == "result":
                self._close_ok()

    def mousePressEvent(self, event):
        if self.phase == "position":
            self._start()
        elif self.phase == "result":
            self._close_ok()

    # -- drawing ------------------------------------------------------------------

    def resizeEvent(self, event):
        self._layout()

    def _layout(self):
        w, h = self.width(), self.height()
        box = QRectF(0, 0, w * 0.42, h * 0.42)
        box.moveCenter(QPointF(w / 2, h * 0.52))
        self.eyes.setGeometry(box.toRect())
        plot = QRectF(0, 0, w * 0.6, h * 0.6)
        plot.moveCenter(QPointF(w / 2, h * 0.55))
        self.plot.setGeometry(plot.toRect())

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BACKGROUND)
        w, h = self.width(), self.height()
        font = QFont(self.font())
        if self.phase == "position":
            font.setPointSize(22)
            p.setFont(font)
            p.setPen(QColor(theme.TEXT))
            kind = "Validation" if self.driver.kind == "validation" \
                else "Calibration"
            done, total = self.driver.progress()
            p.drawText(QRectF(0, h * 0.12, w, 40),
                       Qt.AlignmentFlag.AlignCenter,
                       f"{kind}: {total} points")
            font.setPointSize(15)
            p.setFont(font)
            text, colour = eye_status(self.eye_view())
            p.setPen(QColor(colour))
            p.drawText(QRectF(0, h * 0.2, w, 30),
                       Qt.AlignmentFlag.AlignCenter, text)
            p.setPen(QColor(theme.MUTED))
            p.drawText(QRectF(0, h * 0.8, w, 60),
                       Qt.AlignmentFlag.AlignCenter,
                       "Sit comfortably and keep your head still.\n"
                       "Follow the dot with your eyes. "
                       "Press Space to start, Esc to cancel.")
            return
        if self.phase == "run":
            x, y, aw, ah = self.driver.area()
            if aw < 0.999 or ah < 0.999:
                p.setPen(QPen(QColor(50, 50, 50), 1))
                p.drawRect(QRectF(x * w, y * h, aw * w, ah * h))
            target = self.driver.target()
            if target is not None:
                fx, fy, progress = target
                centre = QPointF(fx * w, fy * h)
                r = 26 - 16 * min(1.0, max(0.0, progress))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(255, 255, 255))
                p.drawEllipse(centre, r, r)
                p.setBrush(QColor(theme.BRAND))
                p.drawEllipse(centre, 4, 4)
            return
        if self.phase == "result" and self.quality is not None:
            font.setPointSize(20)
            p.setFont(font)
            p.setPen(QColor(theme.TEXT))
            q = self.quality
            kind = "Validation" if q.kind == "validation" else "Calibration"
            p.drawText(QRectF(0, h * 0.1, w, 40),
                       Qt.AlignmentFlag.AlignCenter, f"{kind} done")
            font.setPointSize(14)
            p.setFont(font)
            p.setPen(QColor(theme.MUTED))
            p.drawText(QRectF(0, h * 0.16, w, 30),
                       Qt.AlignmentFlag.AlignCenter, q.summary())
