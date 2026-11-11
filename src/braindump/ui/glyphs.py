"""The toolbar and status icons, drawn rather than shipped as images.

(From Rhino Recorder, the ITU brAIn lab's Unicorn recorder, MIT.)
"""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QIcon, QPainter, QPainterPath,
                           QPen, QPixmap, QPolygonF)

from . import theme

SIZE = 64


def _icon(draw, color: str = theme.ICON, size: int = SIZE) -> QIcon:
    """An icon from a function that paints on a size x size canvas,
    with a greyed version for the disabled state."""
    icon = QIcon()
    for mode, tint in ((QIcon.Mode.Normal, color),
                       (QIcon.Mode.Disabled, theme.ICON_DISABLED)):
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        draw(painter, QColor(tint), size)
        painter.end()
        icon.addPixmap(pixmap, mode)
    return icon


def play(color: str = theme.ICON) -> QIcon:
    def draw(p, c, s):
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(c)
        p.drawPolygon(QPolygonF([QPointF(s * .25, s * .15),
                                 QPointF(s * .85, s * .5),
                                 QPointF(s * .25, s * .85)]))
    return _icon(draw, color)


def stop(color: str = theme.ICON) -> QIcon:
    def draw(p, c, s):
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(c)
        p.drawRect(QRectF(s * .2, s * .2, s * .6, s * .6))
    return _icon(draw, color)


def record(color: str = theme.ICON) -> QIcon:
    def draw(p, c, s):
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(c)
        p.drawEllipse(QRectF(s * .18, s * .18, s * .64, s * .64))
    return _icon(draw, color)


def gear(color: str = theme.ICON) -> QIcon:
    def draw(p, c, s):
        centre = QPointF(s / 2, s / 2)
        teeth, outer, inner = 8, s * .42, s * .32
        path = QPainterPath()
        points = []
        for i in range(teeth * 4):
            angle = 2 * math.pi * i / (teeth * 4)
            r = outer if (i % 4) in (1, 2) else inner
            points.append(QPointF(centre.x() + r * math.cos(angle),
                                  centre.y() + r * math.sin(angle)))
        path.addPolygon(QPolygonF(points))
        path.closeSubpath()
        hole = QPainterPath()
        hole.addEllipse(centre, s * .13, s * .13)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(c)
        p.drawPath(path.subtracted(hole))
    return _icon(draw, color)


def question(color: str = theme.ICON) -> QIcon:
    def draw(p, c, s):
        font = p.font()
        font.setBold(True)
        font.setPixelSize(int(s * .8))
        p.setFont(font)
        p.setPen(c)
        p.drawText(QRectF(0, 0, s, s), Qt.AlignmentFlag.AlignCenter, "?")
    return _icon(draw, color)


def wifi(color: str = theme.ICON_ACTIVE) -> QIcon:
    def draw(p, c, s):
        pen = QPen(c, s * .09, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        centre = QPointF(s / 2, s * .8)
        for r in (s * .2, s * .38, s * .56):
            rect = QRectF(centre.x() - r, centre.y() - r, 2 * r, 2 * r)
            p.drawArc(rect, 45 * 16, 90 * 16)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(c)
        p.drawEllipse(centre, s * .06, s * .06)
    return _icon(draw, color)


def network(color: str = theme.ICON_ACTIVE) -> QIcon:
    def draw(p, c, s):
        pen = QPen(c, s * .07)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        nodes = [QPointF(s * .5, s * .2), QPointF(s * .2, s * .78),
                 QPointF(s * .8, s * .78)]
        for a in nodes:
            for b in nodes:
                p.drawLine(a, b)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(c)
        for n in nodes:
            p.drawEllipse(n, s * .12, s * .12)
    return _icon(draw, color)


def battery(level: float | None, color: str = theme.ICON_ACTIVE) -> QIcon:
    """``level`` from 0 to 100, or None when unknown."""
    def draw(p, c, s):
        body = QRectF(s * .08, s * .3, s * .76, s * .4)
        p.setPen(QPen(c, s * .06))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(body, s * .06, s * .06)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(c)
        p.drawRect(QRectF(s * .86, s * .42, s * .07, s * .16))
        if level is not None:
            fill = body.adjusted(s * .07, s * .07, -s * .07, -s * .07)
            fill.setWidth(fill.width() * max(0.0, min(level, 100.0)) / 100)
            tint = c if level > 20 else QColor(theme.BAD)
            p.setBrush(QBrush(tint))
            p.drawRect(fill)
    return _icon(draw, color)


def link(color: str = theme.ICON_ACTIVE) -> QIcon:
    def draw(p, c, s):
        p.setPen(QPen(c, s * .09, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.save()
        p.translate(s / 2, s / 2)
        p.rotate(-45)
        w, h = s * .38, s * .22
        p.drawRoundedRect(QRectF(-w + s * .04, -h / 2, w, h), h / 2, h / 2)
        p.drawRoundedRect(QRectF(-s * .04, -h / 2, w, h), h / 2, h / 2)
        p.restore()
    return _icon(draw, color)
