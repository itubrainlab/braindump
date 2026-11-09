"""Widgets for the eye tracker: the camera's view of the eyes, the gaze
on the screen, and a calibration's quality."""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QFont, QImage, QPainter, QPen,
                           QPixmap)
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..devices.gp3_backends import CalibrationQuality, EyeView
from . import theme

EYE_COLOURS = {"L": theme.LEFT_EYE, "R": theme.RIGHT_EYE, "B": theme.GOOD}
#: Comfortable distance from the GP3, in cm.
IDEAL_CM = (55, 75)
RANGE_CM = (45, 85)


def eye_status(view: EyeView) -> tuple[str, str]:
    """What to tell the operator about the eyes, and in what colour."""
    eyes = view.eyes
    if len(eyes) == 2:
        text, colour = "Both eyes", theme.GOOD
    elif eyes:
        text = f"Only the {'left' if 'L' in eyes else 'right'} eye"
        colour = theme.WARNING
    else:
        return "No eyes found", theme.BAD
    xs = [e[0] for e in eyes.values()]
    ys = [e[1] for e in eyes.values()]
    m, hints = 0.08, []
    # In the camera image, the subject's right is on the left.
    if min(xs) < m:
        hints.append("move to your left")
    if max(xs) > 1 - m:
        hints.append("move to your right")
    if min(ys) < m:
        hints.append("lower your head")
    if max(ys) > 1 - m:
        hints.append("raise your head")
    if view.distance_m:
        cm = view.distance_m * 100
        if cm < IDEAL_CM[0]:
            hints.append("move back")
        elif cm > IDEAL_CM[1]:
            hints.append("move closer")
    if hints:
        text += ": " + ", ".join(hints)
        colour = theme.WARNING
    return text, colour


class EyeViewWidget(QWidget):
    """The GP3's camera image (or, over the Open Gaze API, where its
    pupils are in it), with the pupils marked, the operator's hints and
    the head-distance bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = EyeView()
        self.mirror = True
        self.connected = False
        self.setMinimumSize(320, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

    def show_view(self, view: EyeView, connected: bool = True) -> None:
        self.view = view
        self.connected = connected
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(theme.DARK))
        view = self.view
        w, h = view.image_size
        bar = 34
        area = QRectF(self.rect()).adjusted(4, 4, -4, -4 - bar)
        scale = min(area.width() / w, area.height() / h)
        image = QRectF(0, 0, w * scale, h * scale)
        image.moveCenter(area.center())
        if not self.connected:
            p.setPen(QColor(theme.MUTED))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Connect the GP3 to see the eyes")
            return
        if view.image is not None:
            img = np.ascontiguousarray(view.image)
            if self.mirror:
                img = np.ascontiguousarray(img[:, ::-1])
            qimg = QImage(img.data, img.shape[1], img.shape[0],
                          img.strides[0], QImage.Format.Format_Grayscale8)
            p.drawPixmap(image, QPixmap.fromImage(qimg),
                         QRectF(0, 0, img.shape[1], img.shape[0]))
        else:
            p.fillRect(image, QColor("#151515"))
            p.setPen(QPen(QColor(theme.ALTERNATE), 1, Qt.PenStyle.DashLine))
            p.drawRect(image)
            p.setPen(QColor(theme.MUTED))
            p.drawText(image.adjusted(8, 8, -8, -8),
                       Qt.AlignmentFlag.AlignBottom
                       | Qt.AlignmentFlag.AlignRight,
                       "camera image not sent over the Open Gaze API")
        for side, (cx, cy, d) in view.eyes.items():
            x = 1 - cx if self.mirror else cx
            centre = QPointF(image.left() + x * image.width(),
                             image.top() + cy * image.height())
            r = max(4.0, d * image.width() / 2)
            colour = QColor(EYE_COLOURS[side])
            p.setPen(QPen(colour, 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(centre, r + 3, r + 3)
            if view.image is None:
                p.setBrush(QBrush(QColor(20, 20, 20)))
                p.setPen(QPen(QColor(200, 200, 200), 1))
                p.drawEllipse(centre, r * 2.6, r * 1.5)
                p.setBrush(QBrush(colour))
                p.drawEllipse(centre, r, r)
            p.setPen(colour)
            p.drawText(QPointF(centre.x() - 4, centre.y() - r - 8), side)
        text, colour = eye_status(view)
        p.setPen(QColor(colour))
        font = QFont(p.font())
        font.setBold(True)
        p.setFont(font)
        p.drawText(QRectF(image.left() + 8, image.top() + 6,
                          image.width() - 16, 20),
                   Qt.AlignmentFlag.AlignLeft, text)
        font.setBold(False)
        p.setFont(font)
        p.setPen(QColor(theme.MUTED))
        p.drawText(QRectF(image.left() + 8, image.top() + 6,
                          image.width() - 16, 20),
                   Qt.AlignmentFlag.AlignRight, f"{view.fps:.0f} Hz")
        self._distance_bar(p, QRectF(image.left(), image.bottom() + 8,
                                     image.width(), bar - 10))

    def _distance_bar(self, p: QPainter, r: QRectF):
        lo, hi = RANGE_CM
        x0, x1 = r.left() + 50, r.right() - 70
        y = r.center().y()
        p.setPen(QColor(theme.MUTED))
        p.drawText(QRectF(r.left(), r.top(), 46, r.height()),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                   "close")
        p.drawText(QRectF(x1 + 6, r.top(), 64, r.height()),
                   Qt.AlignmentFlag.AlignVCenter, "far")
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(theme.ALTERNATE))
        p.drawRoundedRect(QRectF(x0, y - 4, x1 - x0, 8), 4, 4)
        g0 = x0 + (IDEAL_CM[0] - lo) / (hi - lo) * (x1 - x0)
        g1 = x0 + (IDEAL_CM[1] - lo) / (hi - lo) * (x1 - x0)
        p.setBrush(QColor(46, 204, 64, 70))
        p.drawRect(QRectF(g0, y - 4, g1 - g0, 8))
        cm = self.view.distance_m * 100 if self.view.distance_m else None
        if cm:
            f = min(1.0, max(0.0, (cm - lo) / (hi - lo)))
            good = IDEAL_CM[0] <= cm <= IDEAL_CM[1]
            p.setBrush(QColor(theme.GOOD if good else theme.WARNING))
            p.drawEllipse(QPointF(x0 + f * (x1 - x0), y), 7, 7)
            p.setPen(QColor(theme.TEXT))
            p.drawText(QRectF(x0, r.top() - 4, x1 - x0, 12),
                       Qt.AlignmentFlag.AlignCenter, f"{cm:.0f} cm")


class _ScreenWidget(QWidget):
    """Something drawn on a picture of the screen."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.screen_size = (1920, 1080)
        self.setMinimumSize(240, 150)

    def screen_rect(self) -> QRectF:
        w, h = self.screen_size or (16, 9)
        w, h = max(1, w), max(1, h)
        area = QRectF(self.rect()).adjusted(8, 8, -8, -8)
        scale = min(area.width() / w, area.height() / h)
        r = QRectF(0, 0, w * scale, h * scale)
        r.moveCenter(area.center())
        return r

    @staticmethod
    def point(r: QRectF, fx: float, fy: float) -> QPointF:
        return QPointF(r.left() + fx * r.width(), r.top() + fy * r.height())


class GazeScreen(_ScreenWidget):
    """Where the participant looks, live: the last half second of gaze,
    and the current fixation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.trail = np.empty((0, 2))
        self.fixation: Optional[tuple[float, float]] = None

    def show_gaze(self, trail: np.ndarray, fixation, screen) -> None:
        self.trail, self.fixation, self.screen_size = trail, fixation, screen
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.screen_rect()
        p.fillRect(r, QColor("#1b1b1b"))
        p.setPen(QPen(QColor(theme.ALTERNATE), 1))
        p.drawRect(r)
        for fx in (1 / 3, 2 / 3):
            p.drawLine(self.point(r, fx, 0), self.point(r, fx, 1))
            p.drawLine(self.point(r, 0, fx), self.point(r, 1, fx))
        n = len(self.trail)
        for i, (x, y) in enumerate(self.trail):
            a = int(40 + 180 * (i + 1) / max(1, n))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(79, 195, 247, a))
            p.drawEllipse(self.point(r, x, y), 3, 3)
        if self.fixation is not None:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(theme.GOOD), 2))
            p.drawEllipse(self.point(r, *self.fixation), 12, 12)
        p.setPen(QColor(theme.MUTED))
        w, h = self.screen_size
        p.drawText(r.adjusted(6, 4, -6, -4), Qt.AlignmentFlag.AlignBottom
                   | Qt.AlignmentFlag.AlignRight, f"{w}×{h}")


class QualityPlot(_ScreenWidget):
    """A calibration's targets, and where each eye's gaze landed on them."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.quality: Optional[CalibrationQuality] = None
        self.setMinimumSize(240, 150)

    def show_quality(self, quality: Optional[CalibrationQuality]) -> None:
        self.quality = quality
        if quality is not None and quality.screen[0]:
            self.screen_size = quality.screen
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.screen_rect()
        p.fillRect(r, QColor("#1b1b1b"))
        p.setPen(QPen(QColor(theme.ALTERNATE), 1))
        p.drawRect(r)
        q = self.quality
        if q is None:
            p.setPen(QColor(theme.MUTED))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "Not calibrated")
            return
        ax, ay, aw, ah = q.area
        if aw < 0.999 or ah < 0.999:
            p.setPen(QPen(QColor(theme.MUTED), 1, Qt.PenStyle.DashLine))
            p.drawRect(QRectF(self.point(r, ax, ay),
                              self.point(r, ax + aw, ay + ah)))
        for point in q.points:
            target = self.point(r, point.x, point.y)
            for eye, (gx, gy) in point.gaze.items():
                colour = QColor(EYE_COLOURS.get(eye, theme.GOOD))
                g = self.point(r, gx, gy)
                p.setPen(QPen(colour, 1))
                p.drawLine(target, g)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(colour)
                p.drawEllipse(g, 3.5, 3.5)
            p.setBrush(Qt.BrushStyle.NoBrush)
            ok = bool(point.gaze)
            p.setPen(QPen(QColor(theme.TEXT if ok else theme.BAD), 1.5))
            p.drawEllipse(target, 7, 7)
            if not ok:
                p.drawLine(target + QPointF(-5, -5), target + QPointF(5, 5))
                p.drawLine(target + QPointF(-5, 5), target + QPointF(5, -5))
            elif point.accuracy_deg is not None:
                p.setPen(QColor(theme.MUTED))
                p.drawText(target + QPointF(9, -6),
                           f"{point.accuracy_deg:.1f}°")
        y = r.top() + 4
        for eye, text in (("L", "left eye"), ("R", "right eye"),
                          ("B", "both eyes")):
            if any(eye in pt.gaze for pt in q.points):
                p.setPen(QColor(EYE_COLOURS[eye]))
                p.drawText(QRectF(r.left() + 6, y, 120, 16),
                           Qt.AlignmentFlag.AlignLeft, f"● {text}")
                y += 15
