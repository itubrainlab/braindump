"""The power spectrum beside the data viewer: the EEG's power spectral
density averaged across the electrodes (see :mod:`.power_spectrum`),
in decibels, drawn like the data viewer.

(From Rhino Recorder, the ITU brAIn lab's Unicorn recorder, MIT.)
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget

from . import power_spectrum as spectrum
from . import theme
from .scope import BOTTOM_AXIS_HEIGHT

#: Room above the curve for the band labels, as a share of its span.
HEADROOM = 0.15
#: Fill of the frequency bands, alternating so neighbours stand apart.
BAND_FILLS = ((255, 255, 255, 34), (255, 255, 255, 16))

TOOLTIP = (
    "Power spectral density of the EEG, filtered or raw as chosen above, "
    "averaged across the "
    f"eight electrodes: Welch's estimate over the last "
    f"{spectrum.WINDOW_S:g} s, in {spectrum.SEGMENT_S:g} s segments "
    f"({1 / spectrum.SEGMENT_S:g} Hz resolution), up to "
    f"{spectrum.MAX_FREQUENCY:g} Hz. Raw shows the whole band, mains "
    "interference included. Power is in dB "
    "relative to 1 µV²/Hz: 10 dB is ten "
    "times the power.\n\nShaded bands: "
    + ", ".join(f"{symbol} {name} {low:g}–{high:g} Hz"
                for symbol, name, low, high in spectrum.BANDS) + ".")


class SpectrumView(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot = pg.PlotWidget(background=theme.WINDOW)
        item = self.plot.getPlotItem()
        item.setMenuEnabled(False)
        item.hideButtons()
        item.setMouseEnabled(False, False)
        item.setXRange(0, spectrum.MAX_FREQUENCY, padding=0)
        item.enableAutoRange(x=False, y=False)
        item.showGrid(x=True, y=True, alpha=0.3)
        muted = pg.mkPen(theme.MUTED)
        for name in ("left", "bottom"):
            axis = item.getAxis(name)
            axis.setPen(muted)
            axis.setTextPen(pg.mkPen(theme.TEXT))
        item.getAxis("bottom").setTickSpacing(major=20.0, minor=10.0)
        item.getAxis("bottom").setHeight(BOTTOM_AXIS_HEIGHT)
        item.setLabel("bottom", "Frequency (Hz)", color=theme.MUTED)
        item.setLabel("left", "Power (dB µV²/Hz)", color=theme.MUTED)
        self._bands(item)
        self.curve = pg.PlotDataItem(pen=pg.mkPen(theme.TEXT, width=1))
        item.addItem(self.curve)
        self.plot.setToolTip(TOOLTIP)

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self.plot)

    def _bands(self, item) -> None:
        """Grey bands behind the curve, one per EEG frequency band, each
        labelled with its symbol near the top."""
        self.band_regions = []
        for i, (symbol, name, low, high) in enumerate(spectrum.BANDS):
            region = pg.LinearRegionItem(
                (low, high), movable=False,
                brush=pg.mkBrush(*BAND_FILLS[i % 2]),
                pen=pg.mkPen(None))
            region.setZValue(-10)
            item.addItem(region)
            # An invisible line carries the label, which keeps its place
            # near the top however the power axis scales.
            line = pg.InfiniteLine((low + high) / 2, angle=90,
                                   pen=pg.mkPen(None), label=symbol,
                                   labelOpts={"position": 0.96,
                                              "color": theme.MUTED,
                                              # Centred on the band.
                                              "anchors": [(0.5, 0.5),
                                                          (0.5, 0.5)]})
            line.setZValue(-9)
            item.addItem(line)
            self.band_regions.append(region)

    def show_eeg(self, eeg: np.ndarray) -> None:
        """Show the spectrum of (samples, channels) EEG in microvolts."""
        freqs, psd = spectrum.average_psd(eeg)
        if len(freqs) == 0:
            self.clear()
            return
        db = spectrum.decibels(psd)
        self.curve.setData(freqs, db)
        low, high = float(db.min()), float(db.max())
        span = max(high - low, 1.0)
        self.plot.setYRange(low - 0.05 * span, high + HEADROOM * span,
                            padding=0)

    def clear(self) -> None:
        self.curve.setData([], [])
