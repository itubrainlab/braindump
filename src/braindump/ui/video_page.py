"""The webcam's and the screen capture's page: what the camera (or OBS) sends,
live, and its settings."""
from __future__ import annotations

import threading

import numpy as np
from PySide6.QtCore import QObject, QSize, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QComboBox, QLabel, QPushButton, QSizePolicy,
                               QVBoxLayout, QWidget)

from ..devices.video import (ObsDevice, VideoDevice, WebcamDevice,
                             find_cameras, find_ndi_sources)
from . import icons, theme
from .device_page import DevicePage
from .forms import ConfigForm, hint, panel, row


class _Found(QObject):
    found = Signal(object)


class Preview(QLabel):
    """The latest frame, scaled to fit."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(QSize(320, 180))
        self.setSizePolicy(QSizePolicy.Policy.Ignored,
                           QSizePolicy.Policy.Ignored)
        self.setStyleSheet(f"background: {theme.DARK}; color: {theme.MUTED};")
        self._shown = None

    def show_image(self, image: np.ndarray, key) -> None:
        if key is self._shown:
            return
        self._shown = key
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        h, w = rgb.shape[:2]
        qimage = QImage(rgb.data, w, h, rgb.strides[0],
                        QImage.Format.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimage).scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def blank(self, text: str) -> None:
        self._shown = None
        self.clear()
        self.setText(text)


class VideoPage(DevicePage):
    def __init__(self, device: VideoDevice, parent=None):
        super().__init__(device, parent)
        box = QWidget()
        column = QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        self.preview = Preview()
        column.addWidget(self.preview, 1)
        self.info = QLabel()
        self.info.setObjectName("Hint")
        column.addWidget(self.info)
        self.tabs.addTab(panel("Video", box),
                         icons.device(device.TYPE), "Video")
        self._build_settings()
        self.set_state(device.state)

    def _build_settings(self):
        cfg = self.device.config
        source = ConfigForm("Source", cfg)
        if isinstance(self.device, WebcamDevice):
            self.choices = QComboBox()
            self.choices.setMinimumWidth(240)
            self.choices.activated.connect(self._pick_camera)
            search = QPushButton("Find cameras")
            search.clicked.connect(self._search)
            source.integer("camera", "Camera", 0, 16, "The camera's number: "
                           "0 is the first (built-in) one")
            source.form.addRow("", row(search, self.choices, None))
            source.integer("width", "Width", 0, 7680, suffix=" px")
            source.integer("height", "Height", 0, 4320, suffix=" px")
            source.number("fps", "Frame rate", 1, 240, "Asked of the "
                          "camera; it may choose another", " fps", 1)
        else:
            self.choices = QComboBox()
            self.choices.setMinimumWidth(240)
            self.choices.activated.connect(self._pick_ndi)
            search = QPushButton("Find NDI sources")
            search.clicked.connect(self._search)
            self.ndi_field = source.line(
                "source", "NDI source", "The source's name, or part of it; "
                "empty takes the first whose name says OBS.", "OBS")
            source.form.addRow("", row(search, self.choices, None))
            source.form.addRow("", hint(
                "In OBS, install the DistroAV plugin and turn on Tools → NDI "
                "Output Settings → Main Output. OBS can run on this computer "
                "or another one on the network."))
        source.boolean("simulate", "Simulated", "A simulated source (no "
                       "hardware)")
        self.choices.setVisible(False)
        bids = ConfigForm("BIDS", cfg)
        bids.line("label", "Recording label", "The recording-<label> of the "
                  "files, telling this video from the others (letters and "
                  "digits)")
        bids.line("position", "Position", "Where it sees from, for the "
                  "sidecar (DevicePosition), e.g. front, overhead")
        bids.number("latency_ms", "Delay", 0, 1000, "How long after it was "
                    "taken a frame arrives, taken off its time stamp when "
                    "known (camera exposure and transfer; OBS's encoding).",
                    " ms", 0)
        for form in (source, bids):
            form.changed.connect(lambda _: self.config_changed.emit())
        self.add_settings_tab(source, bids, hint(
            "Stored as BIDS behavioural video (proposal BEP047): beh/…"
            "_recording-<label>_video.mp4 with its sidecar, and every "
            "frame's time in …_recording-<label>_physio.tsv.gz."))

    # -- finding sources ---------------------------------------------------------

    def _search(self):
        relay = _Found(self)
        relay.found.connect(self._found)
        self._relay = relay
        webcam = isinstance(self.device, WebcamDevice)

        def work():
            try:
                result = find_cameras() if webcam else find_ndi_sources()
            except Exception as error:  # noqa: BLE001
                result = error
            relay.found.emit(result)
        threading.Thread(target=work, daemon=True).start()
        self.choices.clear()
        self.choices.addItem("Searching…")
        self.choices.setVisible(True)

    def _found(self, result):
        self.choices.clear()
        if isinstance(result, Exception):
            self.choices.addItem(f"Search failed: {result}")
        elif not result:
            self.choices.addItem("None found")
        elif isinstance(self.device, WebcamDevice):
            for index, name in result:
                self.choices.addItem(f"{index}: {name}", index)
        else:
            for name in result:
                self.choices.addItem(name, name)

    def _pick_camera(self, i: int):
        index = self.choices.itemData(i)
        if index is not None:
            self.device.config.camera = int(index)
            self.config_changed.emit()

    def _pick_ndi(self, i: int):
        name = self.choices.itemData(i)
        if name:
            self.device.config.source = name
            self.ndi_field.setText(name)
            self.config_changed.emit()

    # -- live -----------------------------------------------------------------------

    def set_state(self, state):
        super().set_state(state)
        if hasattr(self, "preview") and state != "running":
            self.preview.blank("Connect to see the video")
            self.info.setText("")

    def refresh(self):
        d = self.device
        frame = d.latest()
        if frame is None:
            return
        self.preview.show_image(frame.image, frame)
        size = d.source.info.get("size", "") if d.source else ""
        text = f"{d.identity()} · {size} · {d.fps:.1f} fps"
        if d.recording:
            text += " · recording"
        self.info.setText(text)

    def status_icons(self):
        d = self.device
        if not d.running:
            return []
        return [(icons.device(d.TYPE, theme.ICON_ACTIVE), d.status())]
