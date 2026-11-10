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
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .. import __version__, clock
from ..bids import NA, GzipTsvWriter, write_json
from .base import (IDLE, RUNNING, STARTING, STOPPING, Device, DeviceConfig,
                   RecordedFiles, RecordingTarget, Timeline, start_time)
from .push import ClockMap

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
    time: float              # braindump.clock, when it was taken


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
                f"connected and not in use (on macOS, that brAIn dump may "
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
        self.info = {"name": f"Camera {self.index}", "size": f"{w}×{h}"}

    def read(self):
        ok, image = self.capture.read()
        taken = clock.now() - self.latency_s
        return Frame(image, taken) if ok else None

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None


def find_cameras(limit: int = 6) -> list[tuple[int, str]]:
    """(index, name) of the cameras: names from Qt where it knows them
    (in the system's order, which OpenCV's indices follow), else
    "Camera <n>"."""
    names = []
    try:
        from PySide6.QtMultimedia import QMediaDevices
        names = [d.description() for d in QMediaDevices.videoInputs()]
    except Exception:  # noqa: BLE001
        pass
    if names:
        return list(enumerate(names))
    import cv2
    found = []
    for index in range(limit):
        capture = cv2.VideoCapture(index)
        if capture.isOpened():
            found.append((index, f"Camera {index}"))
        capture.release()
    return found


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


class _Recording:
    """One recording's files, written on their own thread so a slow disk
    cannot hold up the frames."""

    def __init__(self, video: Path, frames: Path, rate_of):
        """``rate_of()`` gives the source's frame rate: read when the
        first frame is written, as a stream says its rate only with its
        frames."""
        self.video, self.frames = video, frames
        self.rate_of = rate_of
        self.rate = float(rate_of())
        self.queue: queue.Queue = queue.Queue(maxsize=QUEUE_FRAMES)
        self.count = 0
        self.dropped = 0
        self.first: Optional[float] = None
        self.last: Optional[float] = None
        self.size: Optional[tuple[int, int]] = None
        self.error: Optional[str] = None
        self._thread = threading.Thread(target=self._write, daemon=True,
                                        name="video-writer")
        self._thread.start()

    def add(self, frame: Frame) -> None:
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            self.dropped += 1

    def _write(self):
        import cv2
        writer = times = None
        try:
            while True:
                frame = self.queue.get()
                if frame is None:
                    break
                image = frame.image
                if writer is None:
                    self.rate = float(self.rate_of())
                    h, w = image.shape[:2]
                    self.size = (w, h)
                    self.video.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(self.video), cv2.VideoWriter_fourcc(*FOURCC),
                        self.rate, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(f"could not write {self.video}")
                    times = GzipTsvWriter(self.frames)
                if image.shape[1::-1] != self.size:
                    image = cv2.resize(image, self.size)
                writer.write(image)
                times.write([frame.time, self.count])
                if self.first is None:
                    self.first = frame.time
                self.last = frame.time
                self.count += 1
        except Exception as error:  # noqa: BLE001
            self.error = str(error)
        finally:
            if writer is not None:
                writer.release()
            if times is not None:
                times.close()

    def close(self) -> None:
        self.queue.put(None)
        self._thread.join(timeout=30)


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
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._latest: Optional[Frame] = None
        self._times: deque[float] = deque(maxlen=90)
        self._recording: Optional[_Recording] = None
        self._target: Optional[RecordingTarget] = None
        self._record_from = 0.0

    def make_source(self) -> FrameSource:
        raise NotImplementedError

    # -- acquisition ---------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self.state != IDLE:
                return
            self.error = None
            self._set_state(STARTING)
            source = SimulatedSource(self.config.label) if self.simulated \
                else self.make_source()
            try:
                source.open()
            except BaseException:
                try:
                    source.close()
                except Exception:  # noqa: BLE001
                    pass
                self._set_state(IDLE)
                raise
            self.source = source
            self._latest = None
            self._times.clear()
            self._running = True
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name=f"{self.TYPE}-frames")
            self._thread.start()
            self._set_state(RUNNING)

    def stop(self) -> None:
        with self._lock:
            if self.state == IDLE:
                return
            self._set_state(STOPPING)
            try:
                if self.recording:
                    self.end_recording()
            except Exception:  # noqa: BLE001
                pass
            self._running = False
            if self._thread:
                self._thread.join(timeout=3)
                self._thread = None
            if self.source is not None:
                self.source.close()
                self.source = None
            self._set_state(IDLE)

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
            recording = self._recording
            if recording is not None and frame.time >= self._record_from:
                recording.add(frame)

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
        return self._recording is not None

    def paths(self, target: RecordingTarget) -> tuple[Path, Path]:
        label = self.config.label or self.TYPE
        video = target.dataset.path(target.entities, DATATYPE, "video",
                                    ".mp4", recording=label)
        frames = target.dataset.path(target.entities, DATATYPE, "physio",
                                     ".tsv.gz", recording=label)
        return video, frames

    def begin_recording(self, target: RecordingTarget) -> list[Path]:
        with self._lock:
            if self.source is None:
                raise RuntimeError(f"The {self.NAME} is not running.")
            video, frames = self.paths(target)
            self._record_from = target.started
            self._target = target
            source = self.source
            self._recording = _Recording(video, frames, lambda: source.rate)
            return [video, frames]

    def end_recording(self) -> RecordedFiles:
        with self._lock:
            recording, target = self._recording, self._target
            if recording is None or target is None:
                return RecordedFiles()
            self._recording = self._target = None
            recording.close()
            info = dict(self.source.info) if self.source else {}

        video, frames = recording.video, recording.frames
        if recording.count == 0:
            return RecordedFiles()

        def finalize(reference: Optional[Timeline]) -> None:
            self._write_sidecars(target, recording, info, reference)

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
            "SoftwareVersions": f"brAIn dump {__version__}",
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
    camera: int = 0
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
        return WebcamSource(int(cfg.camera), int(cfg.width), int(cfg.height),
                            float(cfg.fps), float(cfg.latency_ms))


@dataclass
class ObsConfig(VideoConfig):
    #: OBS usually captures the screen (the stimulus, the game).
    label: str = "screen"
    #: The NDI source's name, or part of it (e.g. "OBS"); empty takes the
    #: first source whose name says OBS.
    source: str = ""


class ObsDevice(VideoDevice):
    TYPE = "obs"
    NAME = "OBS (NDI)"
    MANUFACTURER = "OBS Studio"
    MODEL = "NDI output"
    DESCRIPTION = "OBS's output over NDI, low latency, from this or another " \
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
