"""Video inputs: a webcam, and OBS (or any NDI source) over NDI.

Both are stored as BIDS BEP047 behavioural video
(https://bids.neuroimaging.io/extensions/beps/bep_047.html), a proposal
not yet in the released specification::

    beh/sub-.._task-.._recording-<label>_video.mp4   the video
    beh/sub-.._task-.._recording-<label>_video.json  BEP047's sidecar
    beh/sub-.._task-.._recording-<label>_physio.tsv.gz  every frame's time
    beh/sub-.._task-.._recording-<label>_physio.json

BEP047 aligns a video with the other recordings through scans.tsv, which
gives its first frame's time. A webcam's or a network stream's frames do
not come at an exact rate, so each frame's time is kept as well, in a
physio file of the same ``recording`` label (``timestamp``, ``frame``):
plain BIDS, which the validator accepts today. The video files are listed
in ``.bidsignore`` until BEP047 is part of the specification.

Frames are written as they arrive, at the nominal rate the file declares;
the physio file says when each was really taken.

In the session pipeline, a video is a :class:`VideoSource`: one row per
frame (its time and number), while the images wait in the source for the
:class:`VideoWriter`, g.Pype carrying samples rather than images::

    VideoSource ── VideoWriter   (the MP4 and its frame times, while
                                  recording)
"""
from __future__ import annotations

import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import gpype as gp
import numpy as np

from .. import __version__, clock
from ..bids import NA, GzipTsvWriter, write_json
from ..pipeline import BidsWriter, ClockMap, PushSource
from .base import (IDLE, RUNNING, STARTING, STOPPING, Device, DeviceConfig,
                   RecordedFiles, RecordingTarget, Timeline, start_time)

#: BIDS datatype of video (BEP047): behavioural.
DATATYPE = "beh"
#: OpenCV's MPEG-4 Part 2 writer: in every opencv-python wheel, on every
#: system. FFmpeg calls the codec "mpeg4".
FOURCC, CODEC, PIXEL_FORMAT = "mp4v", "mpeg4", "yuv420p"
#: Frames queued for the writer at most (a slow disk drops, not stalls).
QUEUE_FRAMES = 240


@dataclass
class Frame:
    image: np.ndarray        # BGR, uint8
    time: float              # brainstream.clock, when it was taken


class FrameSource:
    """Delivers frames: ``open()``, then ``read()`` until ``close()``."""

    name = ""
    rate = 30.0
    info: dict

    def open(self) -> None:
        raise NotImplementedError

    def read(self) -> Optional[Frame]:
        """The next frame, or None when none came in time."""
        raise NotImplementedError

    def close(self) -> None:
        pass


# -- webcam ---------------------------------------------------------------------------

class WebcamSource(FrameSource):
    """A camera, through OpenCV (AVFoundation on macOS, Media Foundation
    on Windows, V4L2 on Linux). A frame is stamped when the camera hands
    it over, less ``latency_ms`` (exposure and transfer)."""

    def __init__(self, index: int, width: int, height: int, fps: float,
                 latency_ms: float):
        self.index, self.width, self.height = index, width, height
        self.rate = float(fps)
        self.latency_s = latency_ms / 1000.0
        self.capture = None
        self.info = {}

    def open(self):
        import cv2
        capture = cv2.VideoCapture(self.index)
        if not capture.isOpened():
            raise ConnectionError(
                f"Camera {self.index} could not be opened. Check that it is "
                f"connected and not in use (on macOS, that brAIn stream may "
                f"use the camera: System Settings → Privacy & Security → "
                f"Camera).")
        if self.width and self.height:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.rate:
            capture.set(cv2.CAP_PROP_FPS, self.rate)
        ok, image = capture.read()
        if not ok:
            capture.release()
            raise ConnectionError(f"Camera {self.index} sends no frames.")
        reported = capture.get(cv2.CAP_PROP_FPS)
        if reported and 1 <= reported <= 240:
            self.rate = float(reported)
        self.capture = capture
        h, w = image.shape[:2]
        self.info = {"name": self.name or f"Camera {self.index}",
                     "size": f"{w}×{h}"}

    def read(self):
        ok, image = self.capture.read()
        taken = clock.now() - self.latency_s
        return Frame(image, taken) if ok else None

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None


@dataclass
class Camera:
    """A camera on this computer: OpenCV's number for it, its name, and
    the system's identifier for it (stable while the number is not: it
    shifts as cameras come and go)."""
    index: int
    name: str
    id: str = ""


def opencv_indices(ids: list[str], platform: str = sys.platform
                   ) -> list[Optional[int]]:
    """OpenCV's camera number for each of the system's camera
    identifiers (Qt's ``QCameraDevice.id()``), or None where it cannot
    tell.

    - **macOS**: OpenCV's AVFoundation backend numbers the cameras in
      the order of their unique identifiers (the list sorted with
      NSString ``compare:``, an ordinal comparison), which is also what
      Qt gives as the id; the system's own order is another one.
    - **Linux**: V4L2 numbers them by their device file, /dev/video<n>.
    - **Windows**: OpenCV's Media Foundation backend and Qt both list
      Media Foundation's devices, in its order.
    """
    if platform == "darwin":
        ranked = sorted(ids)
        return [ranked.index(i) for i in ids]
    if platform.startswith("linux"):
        out = []
        for i in ids:
            m = re.search(r"/dev/video(\d+)$", i)
            out.append(int(m.group(1)) if m else None)
        return out
    return list(range(len(ids)))


def find_cameras(limit: int = 6) -> list[Camera]:
    """The cameras, in the system's order, each with OpenCV's number for
    it: named by Qt where it can list them, else found by trying
    OpenCV's numbers (and called "Camera <n>")."""
    devices = []
    try:
        from PySide6.QtMultimedia import QMediaDevices
        devices = [(d.description(), bytes(d.id()).decode(errors="replace"))
                   for d in QMediaDevices.videoInputs()]
    except Exception:  # noqa: BLE001
        pass
    if devices:
        indices = opencv_indices([i for _, i in devices])
        return [Camera(index, name, ident)
                for (name, ident), index in zip(devices, indices)
                if index is not None]
    import cv2
    found = []
    for index in range(limit):
        capture = cv2.VideoCapture(index)
        if capture.isOpened():
            found.append(Camera(index, f"Camera {index}"))
        capture.release()
    return found


def resolve_camera(index: int, ident: str, name: str = "") -> int:
    """OpenCV's number, now, for the camera chosen as ``ident`` (or
    ``index``, when it was chosen by number).

    Raises:
        ConnectionError: If the chosen camera is not connected.
    """
    if not ident:
        return int(index)
    cameras = find_cameras()
    for camera in cameras:
        if camera.id == ident:
            return camera.index
    if not cameras:
        # The cameras could not be listed: trust the number.
        return int(index)
    raise ConnectionError(
        f"The camera {name or ident} is not connected. Connect it, or "
        f"choose another in the settings.")


# -- NDI (OBS) ----------------------------------------------------------------------------

class NdiSource(FrameSource):
    """A video stream over NDI, e.g. OBS's output through its NDI plugin
    (DistroAV): low latency, on this computer or another on the network.
    Each frame carries its sender's time stamp; frames are placed on this
    computer's clock by the smallest delay seen (as for the Bluetooth
    devices), which also covers a sender on another computer."""

    def __init__(self, source_name: str, latency_ms: float):
        self.wanted = source_name.strip()
        self.latency_s = latency_ms / 1000.0
        self.rate = 30.0
        self.info = {}
        self._recv = self._find = None
        self._clock = ClockMap(window_s=30.0, latency_s=self.latency_s)

    def open(self):
        try:
            import NDIlib as ndi
        except ImportError as error:
            raise RuntimeError("NDI needs ndi-python (pip install "
                               "ndi-python).") from error
        if not ndi.initialize():
            raise RuntimeError("NDI could not start on this computer.")
        self._ndi = ndi
        find = ndi.find_create_v2()
        self._find = find
        source, deadline = None, time.monotonic() + 8.0
        while source is None and time.monotonic() < deadline:
            ndi.find_wait_for_sources(find, 1000)
            for s in ndi.find_get_current_sources(find):
                if (self.wanted and self.wanted in s.ndi_name) or \
                        (not self.wanted and "OBS" in s.ndi_name.upper()):
                    source = s
                    break
        if source is None:
            self.close()
            raise ConnectionError(
                f"No NDI source {self.wanted!r} found. In OBS, turn on the "
                f"NDI output (DistroAV plugin: Tools → NDI Output "
                f"Settings), on a computer on this network."
                if self.wanted else
                "No OBS NDI source found. In OBS, turn on the NDI output "
                "(DistroAV plugin: Tools → NDI Output Settings).")
        create = ndi.RecvCreateV3()
        create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
        create.bandwidth = ndi.RECV_BANDWIDTH_HIGHEST
        self._recv = ndi.recv_create_v3(create)
        ndi.recv_connect(self._recv, source)
        self.info = {"name": source.ndi_name}

    def read(self):
        ndi = self._ndi
        kind, video, _, _ = ndi.recv_capture_v2(self._recv, 500)
        if kind != ndi.FRAME_TYPE_VIDEO:
            return None
        received = clock.now()
        try:
            image = np.ascontiguousarray(video.data[:, :, :3])
            if video.frame_rate_D:
                self.rate = video.frame_rate_N / video.frame_rate_D
            stamp = video.timestamp
            if stamp and stamp != ndi.RECV_TIMESTAMP_UNDEFINED:
                taken = self._clock(stamp / 1e7, received)
            else:
                taken = received - self.latency_s
            self.info["size"] = f"{video.xres}×{video.yres}"
        finally:
            ndi.recv_free_video_v2(self._recv, video)
        return Frame(image, taken)

    def close(self):
        ndi = getattr(self, "_ndi", None)
        if ndi is None:
            return
        if self._recv is not None:
            ndi.recv_destroy(self._recv)
            self._recv = None
        if self._find is not None:
            ndi.find_destroy(self._find)
            self._find = None


def find_ndi_sources(timeout_s: float = 3.0) -> list[str]:
    import NDIlib as ndi
    if not ndi.initialize():
        return []
    find = ndi.find_create_v2()
    try:
        deadline = time.monotonic() + timeout_s
        names: list[str] = []
        while time.monotonic() < deadline:
            ndi.find_wait_for_sources(find, 500)
            names = [s.ndi_name for s in ndi.find_get_current_sources(find)]
        return names
    finally:
        ndi.find_destroy(find)


# -- simulated -------------------------------------------------------------------------------

class SimulatedSource(FrameSource):
    """Moving bars and the time, at a steady rate, with the odd late
    frame, as a camera delivers them."""

    def __init__(self, label: str, width: int = 640, height: int = 360,
                 fps: float = 30.0):
        self.label, self.width, self.height = label, width, height
        self.rate = float(fps)
        self.info = {"name": f"{label} (simulated)",
                     "size": f"{width}×{height}"}
        self._n = 0
        self._t0 = 0.0

    def open(self):
        self._t0 = clock.now()
        self._rng = np.random.default_rng()

    def read(self):
        import cv2
        due = self._t0 + self._n / self.rate
        wait = due - clock.now()
        if wait > 0:
            time.sleep(wait)
        taken = due + (self._rng.exponential(0.004))
        image = np.full((self.height, self.width, 3), 40, np.uint8)
        x = int((self._n * 6) % self.width)
        image[:, x:x + 24] = (60, 180, 240)
        cv2.putText(image, f"{self.label}  frame {self._n}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        self._n += 1
        return Frame(image, taken)


# -- the device -----------------------------------------------------------------------------------

@dataclass
class VideoConfig(DeviceConfig):
    simulate: bool = False
    #: BIDS recording label: tells this video from the others.
    label: str = "video"
    #: Where it sees from, for the sidecar (DevicePosition), e.g. "front".
    position: str = ""
    latency_ms: float = 0.0


class VideoSource(PushSource):
    """A video's frames in the session pipeline: one row per frame,
    ``[time, frame number]``. The images stay here, the latest
    ``QUEUE_FRAMES`` of them, for :class:`VideoWriter` to fetch."""

    def __init__(self, rate: float, **kwargs):
        super().__init__(["FRAME"], [""], rate, **kwargs)
        self._images: dict[int, np.ndarray] = {}
        self._order: deque[int] = deque()
        self._images_lock = threading.Lock()
        self._next = 0

    def add(self, frame: Frame) -> None:
        with self._images_lock:
            number = self._next
            self._next += 1
            self._images[number] = frame.image
            self._order.append(number)
            while len(self._order) > QUEUE_FRAMES:
                self._images.pop(self._order.popleft(), None)
        self.push([frame.time, number])

    def image(self, number: int) -> Optional[np.ndarray]:
        with self._images_lock:
            return self._images.get(number)


class VideoWriter(BidsWriter):
    """The frames of a :class:`VideoSource`, while recording, as an MP4
    (at the source's frame rate, read at the first frame: a stream says
    its rate only with its frames) and every frame's time as a physio
    file (``meta["frames"]``). A frame whose image is gone by the time it
    is written (a disk too slow) counts as dropped."""

    extension = ".mp4"

    def __init__(self, name: str, source: VideoSource, rate_of, **kwargs):
        super().__init__(name, **kwargs)
        self.source = source
        self.rate_of = rate_of
        self.video: Optional[Path] = None
        self.frames: Optional[Path] = None

    def _open(self, path, meta):
        self.video, self.frames = Path(path), Path(meta["frames"])
        self.epoch = float(meta.get("epoch", 0.0))
        self.rate = float(self.rate_of())
        self.count = self.dropped = 0
        self.first = self.last = None
        self.size: Optional[tuple[int, int]] = None
        self._writer = self._times = None

    def _write(self, block, first_index):
        import cv2
        for row in block:
            image = self.source.image(int(round(float(row[1]))))
            if image is None:
                self.dropped += 1
                continue
            t = self.epoch + float(row[0])
            if self._writer is None:
                self.rate = float(self.rate_of())
                h, w = image.shape[:2]
                self.size = (w, h)
                writer = cv2.VideoWriter(
                    str(self.video), cv2.VideoWriter_fourcc(*FOURCC),
                    self.rate, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"could not write {self.video}")
                self._writer = writer
                self._times = GzipTsvWriter(self.frames)
            if image.shape[1::-1] != self.size:
                image = cv2.resize(image, self.size)
            self._writer.write(image)
            self._times.write([t, self.count])
            if self.first is None:
                self.first = t
            self.last = t
            self.count += 1

    def _close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._times is not None:
            self._times.close()
            self._times = None
        return {"first": self.first, "last": self.last, "count": self.count,
                "dropped": self.dropped, "rate": self.rate,
                "size": list(self.size) if self.size else None}


@dataclass
class _Recording:
    """What one recording wrote, for its sidecars."""
    video: Path
    frames: Path
    rate: float
    count: int
    dropped: int
    first: float
    last: float
    size: Optional[tuple[int, int]]


@dataclass
class Graph:
    """A video's nodes in the session pipeline."""
    source: VideoSource
    writer: VideoWriter


class VideoDevice(Device):
    """What the webcam and the OBS stream share: frames from a source on
    a thread, the latest kept for the view, and while recording, written
    as BEP047 video with every frame's time."""

    MODALITY = "video"
    CONFIG = VideoConfig
    MODEL = ""

    def __init__(self, config, project_folder: Path, simulate: bool = False):
        super().__init__(config, project_folder, simulate)
        self.source: Optional[FrameSource] = None
        self.graph: Optional[Graph] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._latest: Optional[Frame] = None
        self._times: deque[float] = deque(maxlen=90)
        self._target: Optional[RecordingTarget] = None

    def make_source(self) -> FrameSource:
        raise NotImplementedError

    # -- the session pipeline --------------------------------------------------

    def build(self, pipeline: gp.Pipeline) -> None:
        frames = SimulatedSource(self.config.label) if self.simulated \
            else self.make_source()
        source = VideoSource(float(frames.rate or 30.0),
                             name=self.node_name("source"))
        writer = VideoWriter(self.node_name("bids"), source,
                             lambda: frames.rate or 30.0)
        pipeline.connect(source, writer)
        self.source = frames
        self.graph = Graph(source, writer)

    def opened(self) -> None:
        try:
            self.source.open()
        except BaseException:
            try:
                self.source.close()
            except Exception:  # noqa: BLE001
                pass
            raise
        self._latest = None
        self._times.clear()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"{self.TYPE}-frames")
        self._thread.start()

    def closing(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        if self.source is not None:
            self.source.close()

    def closed(self) -> None:
        self.graph = None
        self.source = None
        self._target = None

    def _run(self):
        misses = 0
        while self._running:
            try:
                frame = self.source.read()
            except Exception as error:  # noqa: BLE001
                if self._running:
                    self._running = False
                    self._fail(f"{self.NAME} stopped: {error}")
                return
            if frame is None:
                misses += 1
                if misses * 0.5 > 10 and self._running:
                    self._running = False
                    self._fail(f"{self.NAME} sent no frames for 10 s.")
                    return
                continue
            misses = 0
            self._latest = frame
            self._times.append(frame.time)
            graph = self.graph
            if graph is not None:
                graph.source.add(frame)

    # -- view ------------------------------------------------------------------

    def latest(self) -> Optional[Frame]:
        return self._latest

    @property
    def fps(self) -> float:
        times = list(self._times)
        if len(times) < 2 or times[-1] <= times[0]:
            return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    # -- recording -------------------------------------------------------------

    @property
    def recording(self) -> bool:
        graph = self.graph
        return bool(graph and graph.writer.recording)

    def paths(self, target: RecordingTarget) -> tuple[Path, Path]:
        label = self.config.label or self.TYPE
        video = target.dataset.path(target.entities, DATATYPE, "video",
                                    ".mp4", recording=label)
        frames = target.dataset.path(target.entities, DATATYPE, "physio",
                                     ".tsv.gz", recording=label)
        return video, frames

    def begin_recording(self, target: RecordingTarget) -> list[Path]:
        with self._lock:
            graph = self.graph
            if graph is None:
                raise RuntimeError(f"The {self.NAME} is not running.")
            video, frames = self.paths(target)
            self.invoke(graph.writer, "begin", {
                "path": str(video),
                "meta": {"frames": str(frames),
                         "epoch": graph.source.epoch}})
            self._target = target
            return [video, frames]

    def end_recording(self) -> RecordedFiles:
        with self._lock:
            graph, target = self.graph, self._target
            if graph is None or target is None:
                return RecordedFiles()
            self._target = None
            summary = self.invoke(graph.writer, "end")
            writer = graph.writer
            info = dict(self.source.info) if self.source else {}
        count = int(summary.get("count") or 0)
        if count == 0:
            return RecordedFiles()
        size = summary.get("size")
        recording = _Recording(writer.video, writer.frames,
                               float(summary["rate"]), count,
                               int(summary.get("dropped") or 0),
                               float(summary["first"]),
                               float(summary["last"]),
                               tuple(size) if size else None)

        def finalize(reference: Optional[Timeline]) -> None:
            self._write_sidecars(target, recording, info, reference)

        video, frames = recording.video, recording.frames
        moment = datetime.fromtimestamp(recording.first)
        sidecars = [_json(video), _json(frames)]
        return RecordedFiles(
            scans=[(video, moment), (frames, moment)],
            files=[video, frames, *sidecars], first_sample=recording.first,
            finalize=finalize, samples=recording.count)

    def _write_sidecars(self, target: RecordingTarget, rec: _Recording,
                        info: dict, reference: Optional[Timeline]) -> None:
        cfg = self.config
        duration = (rec.last - rec.first) + 1.0 / rec.rate \
            if rec.count > 1 else 0.0
        width, height = rec.size or (0, 0)
        meta = {
            "TaskName": target.entities.task,
            **target.common_metadata,
            "Manufacturer": self.MANUFACTURER or NA,
            "ManufacturersModelName": info.get("name") or self.MODEL or NA,
            "RecordingDuration": round(duration, 3),
            "VideoCodec": CODEC,
            "VideoFrameRate": round(rec.rate, 6),
            "VideoFrameCount": rec.count,
            "ImageWidth": width,
            "ImageHeight": height,
            "ImagePixelFormat": PIXEL_FORMAT,
            "FrameTimes": rec.frames.name,
            "DroppedFrames": rec.dropped,
            "SoftwareVersions": f"brAIn stream {__version__}",
            "SimulatedDevice": self.simulated,
        }
        if cfg.position:
            meta["DevicePosition"] = cfg.position
        write_json(_json(rec.video), meta)
        write_json(_json(rec.frames), {
            "Description": f"When each frame of {rec.video.name} was taken "
                           f"(BIDS BEP047 video, aligned through "
                           f"scans.tsv): one row per frame.",
            "PhysioType": "generic",
            "SamplingFrequency": round(rec.rate, 6),
            "StartTime": start_time(reference, rec.first),
            "Columns": ["timestamp", "frame"],
            "timestamp": {"Description": "When the frame was taken, seconds "
                                         "since 1970-01-01 UTC on the "
                                         "recording computer's clock",
                          "Units": "s"},
            "frame": {"Description": f"Frame number in {rec.video.name} "
                                     f"(from 0)"},
            "TaskName": target.entities.task,
            "Manufacturer": self.MANUFACTURER or NA,
            "RecordingDuration": round(duration, 3),
            **target.common_metadata,
        })

    # -- status ------------------------------------------------------------------

    def identity(self) -> str:
        source = self.source
        if source is not None and source.info.get("name"):
            return source.info["name"]
        return ""

    def status(self) -> str:
        name = self.identity() or self.NAME
        if self.state == RUNNING:
            text = "recording" if self.recording else "streaming"
            size = self.source.info.get("size", "") if self.source else ""
            return f"{name}: {text}, {size} at {self.fps:.0f} fps"
        return {IDLE: f"{self.NAME}: not connected",
                STARTING: f"{self.NAME}: connecting…",
                STOPPING: f"{self.NAME}: stopping…"}.get(self.state,
                                                        self.state)


@dataclass
class WebcamConfig(VideoConfig):
    label: str = "webcam"
    #: OpenCV's number for the camera, when it was chosen; worked out
    #: again from camera_id at each connect, as numbers shift.
    camera: int = 0
    #: The system's identifier for the chosen camera, and its name; empty
    #: for a camera chosen by number only.
    camera_id: str = ""
    camera_name: str = ""
    width: int = 1280
    height: int = 720
    fps: float = 30.0


class WebcamDevice(VideoDevice):
    """Video Stream: a camera on this computer. Its type stays "webcam", so
    projects saved before the rename still open."""
    TYPE = "webcam"
    NAME = "Video Stream"
    MANUFACTURER = ""
    DESCRIPTION = "Any camera on this computer (a webcam, a capture card), " \
                  "recorded as BIDS video"
    CONFIG = WebcamConfig

    def make_source(self):
        cfg = self.config
        index = resolve_camera(int(cfg.camera), str(cfg.camera_id or ""),
                               str(cfg.camera_name or ""))
        source = WebcamSource(index, int(cfg.width), int(cfg.height),
                              float(cfg.fps), float(cfg.latency_ms))
        source.name = str(cfg.camera_name or "")
        return source


@dataclass
class ObsConfig(VideoConfig):
    #: OBS usually captures the screen (the stimulus, the game).
    label: str = "screen"
    #: The NDI source's name, or part of it (e.g. "OBS"); empty takes the
    #: first source whose name says OBS.
    source: str = ""


class ObsDevice(VideoDevice):
    """Screen Capture: OBS's output over NDI. Its type stays "obs", so
    projects saved before the rename still open."""
    TYPE = "obs"
    NAME = "Screen Capture"
    MANUFACTURER = "OBS Studio"
    MODEL = "NDI output"
    DESCRIPTION = "The screen as OBS Studio captures it, sent over NDI " \
                  "(DistroAV plugin) with low latency from this or another " \
                  "computer, recorded as BIDS video"
    CONFIG = ObsConfig

    def make_source(self):
        cfg = self.config
        return NdiSource(cfg.source, float(cfg.latency_ms))


def _json(path: Path) -> Path:
    name = path.name
    for ext in (".tsv.gz", ".mp4"):
        if name.endswith(ext):
            return path.with_name(name[:-len(ext)] + ".json")
    return path.with_suffix(".json")
