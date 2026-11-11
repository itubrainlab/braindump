"""Icons, drawn: the toolbar's and status bar's (:mod:`.glyphs`), and
brAIn dump's own, drawn in the same way."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainterPath, QPen
from .glyphs import (_icon, battery, gear, network, play, question,
                     record, stop, wifi)

from . import theme

__all__ = ["battery", "gear", "play", "question", "record", "stop", "wifi",
           "eye", "wave", "plus", "cross", "folder", "app", "dot",
           "target", "check", "connect", "disconnect"]


def app() -> QIcon:
    return QIcon(str(theme.APP_ICON))


def eye(color: str = theme.ICON) -> QIcon:
    def draw(p, c, s):
        p.setPen(QPen(c, s * 0.08))
        path = QPainterPath()
        path.moveTo(s * 0.08, s * 0.5)
        path.quadTo(s * 0.5, s * 0.08, s * 0.92, s * 0.5)
        path.quadTo(s * 0.5, s * 0.92, s * 0.08, s * 0.5)
        p.drawPath(path)
        p.setBrush(c)
        p.drawEllipse(QPointF(s * 0.5, s * 0.5), s * 0.14, s * 0.14)
    return _icon(draw, color)


def wave(color: str = theme.ICON) -> QIcon:
    def draw(p, c, s):
        p.setPen(QPen(c, s * 0.08))
        path = QPainterPath()
        path.moveTo(s * 0.06, s * 0.5)
        xs = [0.18, 0.28, 0.38, 0.5, 0.62, 0.72, 0.82, 0.94]
        ys = [0.5, 0.2, 0.75, 0.35, 0.85, 0.15, 0.6, 0.5]
        for x, y in zip(xs, ys):
            path.lineTo(s * x, s * y)
        p.drawPath(path)
    return _icon(draw, color)


def heart(color: str = theme.ICON) -> QIcon:
    def draw(p, c, s):
        p.setPen(QPen(c, s * 0.08, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        path = QPainterPath()
        path.moveTo(s * 0.5, s * 0.84)
        path.cubicTo(s * 0.12, s * 0.58, s * 0.04, s * 0.36, s * 0.16,
                     s * 0.22)
        path.cubicTo(s * 0.28, s * 0.08, s * 0.46, s * 0.14, s * 0.5,
                     s * 0.3)
        path.cubicTo(s * 0.54, s * 0.14, s * 0.72, s * 0.08, s * 0.84,
                     s * 0.22)
        path.cubicTo(s * 0.96, s * 0.36, s * 0.88, s * 0.58, s * 0.5,
                     s * 0.84)
        p.drawPath(path)
    return _icon(draw, color)


def hand(color: str = theme.ICON) -> QIcon:
    """A drop of sweat: skin conductance."""
    def draw(p, c, s):
        p.setPen(QPen(c, s * 0.08, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        path = QPainterPath()
        path.moveTo(s * 0.5, s * 0.1)
        path.cubicTo(s * 0.5, s * 0.1, s * 0.2, s * 0.48, s * 0.2, s * 0.62)
        path.cubicTo(s * 0.2, s * 0.8, s * 0.34, s * 0.9, s * 0.5, s * 0.9)
        path.cubicTo(s * 0.66, s * 0.9, s * 0.8, s * 0.8, s * 0.8, s * 0.62)
        path.cubicTo(s * 0.8, s * 0.48, s * 0.5, s * 0.1, s * 0.5, s * 0.1)
        p.drawPath(path)
        p.drawLine(QPointF(s * 0.3, s * 0.64), QPointF(s * 0.38, s * 0.64))
        p.drawLine(QPointF(s * 0.38, s * 0.64), QPointF(s * 0.44, s * 0.5))
        p.drawLine(QPointF(s * 0.44, s * 0.5), QPointF(s * 0.52, s * 0.76))
        p.drawLine(QPointF(s * 0.52, s * 0.76), QPointF(s * 0.6, s * 0.64))
        p.drawLine(QPointF(s * 0.6, s * 0.64), QPointF(s * 0.7, s * 0.64))
    return _icon(draw, color)


def _plug(p, c, s, gap: float) -> None:
    """A plug (left, prongs out) and its socket (right), ``gap`` apart
    (a fraction of the icon): 0 is plugged in."""
    shift = gap / 2
    p.setPen(QPen(c, s * 0.09, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    p.setBrush(c)
    # Plug: cable, solid body, two prongs.
    right = 0.44 - shift
    p.drawLine(QPointF(s * 0.04, s * 0.5), QPointF(s * (right - 0.2), s * 0.5))
    p.drawRoundedRect(QRectF(s * (right - 0.2), s * 0.24, s * 0.2, s * 0.52),
                      s * 0.06, s * 0.06)
    p.setPen(QPen(c, s * 0.08, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
    for y in (0.38, 0.62):
        p.drawLine(QPointF(s * right, s * y), QPointF(s * (right + 0.12), s * y))
    # Socket: solid body (the prongs disappear into it), cable.
    left = 0.56 + shift
    p.setPen(QPen(c, s * 0.09, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    p.drawRoundedRect(QRectF(s * left, s * 0.24, s * 0.2, s * 0.52),
                      s * 0.06, s * 0.06)
    p.drawLine(QPointF(s * (left + 0.2), s * 0.5), QPointF(s * 0.96, s * 0.5))


def connect(color: str = theme.ICON) -> QIcon:
    """Connect: the plug in its socket."""
    def draw(p, c, s):
        _plug(p, c, s, 0.06)
    return _icon(draw, color)


def disconnect(color: str = theme.ICON) -> QIcon:
    """Disconnect: plug and socket pulled apart, with a break between."""
    def draw(p, c, s):
        _plug(p, c, s, 0.22)
        p.setPen(QPen(c, s * 0.07, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap))
        for (x0, y0), (x1, y1) in (((0.55, 0.04), (0.5, 0.15)),
                                   ((0.45, 0.96), (0.5, 0.85))):
            p.drawLine(QPointF(s * x0, s * y0), QPointF(s * x1, s * y1))
    return _icon(draw, color)


def camera(color: str = theme.ICON) -> QIcon:
    """A video camera."""
    def draw(p, c, s):
        p.setPen(QPen(c, s * 0.08, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(QRectF(s * 0.08, s * 0.28, s * 0.58, s * 0.44),
                          s * 0.07, s * 0.07)
        path = QPainterPath()
        path.moveTo(s * 0.66, s * 0.44)
        path.lineTo(s * 0.92, s * 0.3)
        path.lineTo(s * 0.92, s * 0.7)
        path.lineTo(s * 0.66, s * 0.56)
        p.drawPath(path)
    return _icon(draw, color)


def plus(color: str = theme.ICON) -> QIcon:
    def draw(p, c, s):
        p.setPen(QPen(c, s * 0.1, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(s * 0.5, s * 0.2), QPointF(s * 0.5, s * 0.8))
        p.drawLine(QPointF(s * 0.2, s * 0.5), QPointF(s * 0.8, s * 0.5))
    return _icon(draw, color)


def cross(color: str = theme.MUTED) -> QIcon:
    def draw(p, c, s):
        p.setPen(QPen(c, s * 0.1, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(s * 0.25, s * 0.25), QPointF(s * 0.75, s * 0.75))
        p.drawLine(QPointF(s * 0.75, s * 0.25), QPointF(s * 0.25, s * 0.75))
    return _icon(draw, color)


def check(color: str = theme.GOOD) -> QIcon:
    def draw(p, c, s):
        p.setPen(QPen(c, s * 0.12, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        path = QPainterPath()
        path.moveTo(s * 0.2, s * 0.52)
        path.lineTo(s * 0.42, s * 0.74)
        path.lineTo(s * 0.8, s * 0.28)
        p.drawPath(path)
    return _icon(draw, color)


def folder(color: str = theme.ICON) -> QIcon:
    def draw(p, c, s):
        p.setPen(QPen(c, s * 0.07))
        path = QPainterPath()
        path.moveTo(s * 0.1, s * 0.25)
        path.lineTo(s * 0.4, s * 0.25)
        path.lineTo(s * 0.48, s * 0.35)
        path.lineTo(s * 0.9, s * 0.35)
        path.lineTo(s * 0.9, s * 0.8)
        path.lineTo(s * 0.1, s * 0.8)
        path.closeSubpath()
        p.drawPath(path)
    return _icon(draw, color)


def target(color: str = theme.ICON) -> QIcon:
    def draw(p, c, s):
        p.setPen(QPen(c, s * 0.08))
        p.drawEllipse(QRectF(s * 0.15, s * 0.15, s * 0.7, s * 0.7))
        p.setBrush(c)
        p.drawEllipse(QPointF(s * 0.5, s * 0.5), s * 0.1, s * 0.1)
    return _icon(draw, color)


def dot(color: str, size: int = 12) -> QIcon:
    def draw(p, c, s):
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(c)
        p.drawEllipse(QRectF(s * 0.15, s * 0.15, s * 0.7, s * 0.7))
    return _icon(draw, color, size)


def device(device_type: str, color: str = theme.ICON) -> QIcon:
    if device_type == "gp3":
        return eye(color)
    if device_type == "lsl":
        return network(color)
    if device_type == "polar":
        return heart(color)
    if device_type == "shimmer":
        return hand(color)
    if device_type in ("webcam", "obs"):
        return camera(color)
    return wave(color)


def colour(name: str) -> QColor:
    return QColor(name)
