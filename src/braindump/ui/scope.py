"""The data viewer: eight EEG channels, ten seconds, sweeping left to
right with a cursor at the newest sample, as in Unicorn Recorder, drawn
the way g.Pype's TimeSeriesScope draws: light traces on the window colour,
a faint grid, channel names on the left axis.

Below the channels, a TRIG row shows each trigger as a marker with its
value, and a tick in the same colour crosses all channels at that
moment. Each trigger value has its own colour (:func:`trigger_colour`).

To the right of the traces, a column shows each channel's signal quality
on that channel's row.

(From Rhino Recorder, the ITU brAIn lab's Unicorn recorder, MIT.)
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QVBoxLayout, QWidget

from . import quality
from . import theme

QUALITY_COLOURS = {quality.GOOD: theme.GOOD, quality.BAD: theme.BAD,
                   quality.UNKNOWN: theme.UNKNOWN}
QUALITY_TEXT = {quality.GOOD: "good", quality.BAD: "bad",
                quality.UNKNOWN: "settling"}
#: Height of both bottom axes, so the two plots' rows line up.
BOTTOM_AXIS_HEIGHT = 40
#: Height of the TRIG row, in channel rows; it spans y from -this to 0.
TRIGGER_ROW = 0.7
#: Golden-ratio conjugate: successive multiples spread hues evenly.
_GOLDEN = 0.6180339887498949


def trigger_colour(value: float) -> QColor:
    """The colour of a trigger value: the same value always gets the same
    colour, and neighbouring values get clearly different ones."""
    hue = (0.11 + _GOLDEN * int(round(value))) % 1.0
    return QColor.fromHsvF(hue, 0.6, 1.0)


class _TimeAxis(pg.AxisItem):
    """Labels each second of the sweep with the time of the sample now
    shown there, so the labels wrap where the cursor is."""

    def __init__(self):
        super().__init__("bottom")
        self.window = 10.0
        self.cursor = 0.0
        self.sweep_start = 0.0

    def tickStrings(self, values, scale, spacing):
        labels = []
        for x in values:
            t = self.sweep_start + x
            if x > self.cursor:
                t -= self.window
            labels.append(f"{t:.0f}" if t >= 0 else "")
        return labels


class Scope(QWidget):
    """Shows the EEG columns of a SweepTap's buffer."""

    def __init__(self, seconds: float = 10.0, parent=None):
        super().__init__(parent)
        self.seconds = seconds
        self.fs = quality.SAMPLING_RATE
        self.scale_uv = 50.0
        self.channels = quality.NUM_EEG

        self.axis = _TimeAxis()
        self.axis.window = seconds
        self.plot = pg.GraphicsLayoutWidget()
        self.plot.setBackground(theme.WINDOW)
        self.plot.ci.layout.setHorizontalSpacing(0)
        item = self.plot.addPlot(row=0, col=0,
                                 axisItems={"bottom": self.axis})
        self.plot_item = item
        item.setMenuEnabled(False)
        item.hideButtons()
        item.setMouseEnabled(False, False)
        item.vb.setDefaultPadding(0)
        item.setXRange(0, seconds, padding=0)
        item.setYRange(-TRIGGER_ROW, self.channels, padding=0)
        item.showGrid(x=True, y=False, alpha=0.3)
        muted = pg.mkPen(theme.MUTED)
        for name in ("left", "bottom"):
            axis = item.getAxis(name)
            axis.setPen(muted)
            axis.setTextPen(pg.mkPen(theme.TEXT))
        self.axis.setTickSpacing(major=1.0, minor=1.0)
        item.getAxis("left").setTicks([[
            (self.channels - i - 0.5, f"CH{i + 1}")
            for i in range(self.channels)] + [(-TRIGGER_ROW / 2, "TRIG")]])
        item.getAxis("left").setWidth(48)
        self.axis.setHeight(BOTTOM_AXIS_HEIGHT)
        item.setLabel("bottom", "Time (s)", color=theme.MUTED)
        self._set_left_label()

        # Row separators, fainter than the grid, as in g.Pype's scope.
        separator = pg.mkPen(255, 255, 255, 40)
        for row in range(1, self.channels):
            item.addItem(pg.InfiniteLine(row, angle=0, pen=separator))
        item.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen(theme.MUTED)))

        self.x = np.arange(int(seconds * self.fs)) / self.fs
        trace = pg.mkPen(theme.TEXT, width=1)
        self.curves = []
        for _ in range(self.channels):
            curve = pg.PlotDataItem(pen=trace, connect="finite",
                                    skipFiniteCheck=False)
            item.addItem(curve)
            self.curves.append(curve)
        #: Per trigger value: (ticks across the channels, marker in TRIG).
        self._trigger_items: dict[int, tuple] = {}
        #: Value labels in the TRIG row, reused from frame to frame.
        self._trigger_labels: list[pg.TextItem] = []
        self._shown_triggers: tuple = ()
        self.cursor = pg.InfiniteLine(0, angle=90,
                                      pen=pg.mkPen(theme.CURSOR, width=2))
        self.cursor.hide()
        item.addItem(self.cursor)

        self._quality_column(item)

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self.plot)

    def _quality_column(self, scope) -> None:
        """A narrow plot beside the traces, its rows linked to theirs,
        with one quality indicator per channel."""
        item = self.plot.addPlot(row=0, col=1)
        item.setMenuEnabled(False)
        item.hideButtons()
        item.setMouseEnabled(False, False)
        item.hideAxis("left")
        item.setYLink(scope)
        item.setXRange(0, 1, padding=0)
        item.setFixedWidth(64)
        bottom = item.getAxis("bottom")
        bottom.setHeight(BOTTOM_AXIS_HEIGHT)
        bottom.setTicks([[]])
        bottom.setPen(pg.mkPen(theme.MUTED))
        item.setLabel("bottom", "Quality", color=theme.MUTED)
        separator = pg.mkPen(255, 255, 255, 40)
        for row in range(1, self.channels):
            item.addItem(pg.InfiniteLine(row, angle=0, pen=separator))
        item.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen(theme.MUTED)))
        item.addItem(pg.InfiniteLine(0, angle=90, pen=pg.mkPen(theme.MUTED)))
        self.quality_dots = pg.ScatterPlotItem(
            x=[0.5] * self.channels,
            y=[self.channels - i - 0.5 for i in range(self.channels)],
            size=16, pen=pg.mkPen(theme.DARK, width=1),
            brush=[pg.mkBrush(theme.UNKNOWN)] * self.channels)
        item.addItem(self.quality_dots)
        self.quality_plot = item
        self.set_quality([quality.UNKNOWN] * self.channels)

    def set_quality(self, verdicts: list[str],
                    deviations=None) -> None:
        """Colour each channel's indicator by its quality verdict."""
        self.quality_dots.setBrush(
            [pg.mkBrush(QUALITY_COLOURS[v]) for v in verdicts])
        lines = []
        for i, verdict in enumerate(verdicts):
            line = f"CH{i + 1}: {QUALITY_TEXT[verdict]}"
            if deviations is not None and verdict != quality.UNKNOWN:
                line += f" ({deviations[i]:.1f} µV)"
            lines.append(line)
        low, high = quality.GOOD_RANGE_UV
        self.plot.setToolTip(
            "Signal quality: green while the raw EEG, filtered to "
            f"{quality.BAND[0]:g}-{quality.BAND[1]:g} Hz, varies by "
            f"{low:g}-{high:g} µV over the last {quality.WINDOW_S:g} s and "
            f"the raw signal swings by at most "
            f"{quality.RAW_SWING_LIMIT_UV / 1000:g} mV; red when flat, too "
            "noisy, drifting, or carrying only the noise common to all "
            "channels (no skin contact); grey while settling.\n\n"
            + "\n".join(lines))

    def _set_left_label(self) -> None:
        self.plot_item.setLabel(
            "left", f"EEG Amplitudes (−{self.scale_uv:g} … "
            f"+{self.scale_uv:g} µV)", color=theme.MUTED)

    def set_scale(self, half_range_uv: float) -> None:
        self.scale_uv = half_range_uv
        self._set_left_label()

    def clear(self) -> None:
        for curve in self.curves:
            curve.setData([], [])
        self.show_triggers([], [])
        self.cursor.hide()

    def _items_for(self, value: int) -> tuple:
        """The tick and marker items of one trigger value, made once."""
        if value not in self._trigger_items:
            colour = trigger_colour(value)
            tick_colour = QColor(colour)
            tick_colour.setAlpha(200)
            # A whole pixel width: at 1.5 px, ticks at some positions were
            # drawn thin and washed out.
            ticks = pg.PlotDataItem(pen=pg.mkPen(tick_colour, width=2),
                                    connect="finite")
            marker = pg.PlotDataItem(pen=pg.mkPen(colour, width=4),
                                     connect="finite")
            self.plot_item.addItem(ticks)
            self.plot_item.addItem(marker)
            self._trigger_items[value] = (ticks, marker)
        return self._trigger_items[value]

    def show_triggers(self, times, values) -> None:
        """Mark triggers at ``times`` (s, in sweep position) with their
        ``values``: a tick across the channels, a marker and the value in
        the TRIG row, all in the value's colour."""
        shown = tuple(zip(np.round(times, 4).tolist(),
                          np.asarray(values).astype(int).tolist()))
        if shown == self._shown_triggers:
            return
        self._shown_triggers = shown
        by_value: dict[int, list[float]] = {}
        for t, value in shown:
            by_value.setdefault(value, []).append(t)
        for value in by_value:
            self._items_for(value)
        for value, (ticks, marker) in self._trigger_items.items():
            ts = np.array(by_value.get(value, []))
            xs = np.repeat(ts, 3)
            xs[2::3] = np.nan
            ticks.setData(xs, np.tile([0.0, float(self.channels), np.nan],
                                      len(ts)))
            marker.setData(xs, np.tile([-TRIGGER_ROW * 0.9,
                                        -TRIGGER_ROW * 0.1, np.nan], len(ts)))
        while len(self._trigger_labels) < len(shown):
            label = pg.TextItem(anchor=(0, 0.5))
            self.plot_item.addItem(label)
            self._trigger_labels.append(label)
        # Labels of triggers close together would overprint: such a label
        # takes the TRIG row's second line.
        seconds_per_px = self.seconds / max(self.plot_item.vb.width(), 1.0)
        lane_of, lane_end = [], [-np.inf, -np.inf]
        for t, value in shown:
            lane = 0 if t >= lane_end[0] else 1
            lane_of.append(lane)
            lane_end[lane] = t + (8 + 7 * len(str(value))) * seconds_per_px
        if 1 in lane_of:
            heights = (-TRIGGER_ROW * 0.3, -TRIGGER_ROW * 0.72)
        else:
            heights = (-TRIGGER_ROW / 2, -TRIGGER_ROW / 2)
        for i, label in enumerate(self._trigger_labels):
            if i < len(shown):
                t, value = shown[i]
                label.setText(f"{value}", color=trigger_colour(value))
                label.setPos(t, heights[lane_of[i]])
                label.show()
            else:
                label.hide()

    def show_sweep(self, buffer: np.ndarray, total: int,
                   eeg: list[int], status: int | None) -> None:
        """Draw a SweepTap buffer that has seen ``total`` samples."""
        capacity = buffer.shape[0]
        if total == 0 or capacity != len(self.x):
            self.clear()
            return
        cursor = total % capacity
        filled = min(total, capacity)
        for row, column in enumerate(eeg[:self.channels]):
            values = buffer[:, column] / (2 * self.scale_uv)
            # Each trace stays inside its own row.
            y = (self.channels - row - 0.5) + np.clip(values, -0.5, 0.5)
            if filled < capacity:
                y[filled:] = np.nan
            # Break the line at the cursor, where new meets old.
            y[cursor] = np.nan
            self.curves[row].setData(self.x, y)
        if status is not None:
            slots = np.flatnonzero(buffer[:filled, status])
            self.show_triggers(self.x[slots], buffer[slots, status])
        self.cursor.setValue(cursor / self.fs)
        self.cursor.show()
        self.axis.cursor = cursor / self.fs
        self.axis.sweep_start = (total - cursor) / self.fs
        self.axis.picture = None
        self.axis.update()
