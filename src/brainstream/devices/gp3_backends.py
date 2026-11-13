"""Where the GP3's data comes from: three backends with one interface.

- :class:`LocalBackend`: the GP3 on this computer, through OpenGP3's own
  driver and :class:`opengp3.GazeTracker`.
- :class:`RemoteBackend`: a GP3 on another computer, through the Open
  Gaze API (OpenGP3's ``opengp3 server`` or Gazepoint Control), with
  OpenGP3's :class:`opengp3.OpenGazeClient`.
- :class:`SimulatedBackend`: no hardware.

Each delivers rows of ``[time.time() of capture, *CHANNELS]`` to
``on_row``; :data:`CHANNELS` are OpenGP3's LSL channels, named after the
Open Gaze API's fields. Each also gives the eye view
(:meth:`eye_view`) and runs calibration through a *driver*, which the
calibration window ticks and draws (:class:`CalibrationDriver`).
"""
from __future__ import annotations

import math
import random
import socket
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from opengp3.calibration import (API_DEFAULT_POINTS, NINE_POINTS,
                                 VALIDATION_POINTS, grid_points)
from opengp3.lsl import CHANNELS as LSL_CHANNELS

from .. import clock
from ..pipeline import ClockMap

#: Channel names, in row order after the capture time.
CHANNELS = [name for name, _, _ in LSL_CHANNELS]
UNITS = {name: unit for name, unit, _ in LSL_CHANNELS}
DESCRIPTIONS = {name: text for name, _, text in LSL_CHANNELS}
INDEX = {name: i for i, name in enumerate(CHANNELS)}
CAMERA_SIZE = (752, 480)
#: The GP3's frame rate.
RATE = 60.0
#: The trackers OpenGP3 drives, and the frame rates each can run at. A
#: webcam is OpenGP3's low-accuracy webcam eye tracking.
WEBCAM = "Webcam"
MODELS = ("GP3", "GP3 HD", WEBCAM)
HD_RATES = (150, 60)
#: A webcam's usual frame rate, until one has been connected.
WEBCAM_RATE = 30.0


def nominal_rate(model: str, hd_rate: float = HD_RATES[0],
                 webcam_rate: float = WEBCAM_RATE) -> float:
    """The frame rate a ``model`` runs at: the GP3 always 60 Hz, the GP3 HD
    at ``hd_rate``, a webcam at ``webcam_rate``."""
    if model == WEBCAM:
        return float(webcam_rate or WEBCAM_RATE)
    return float(hd_rate) if model == "GP3 HD" else RATE


def model_name(model: str) -> str:
    """brAIn stream's name for the tracker OpenGP3 calls ``model`` (``GP3``,
    ``GP3 HD``, ``WEBCAM``, ``Simulated GP3``) or the Open Gaze API's
    ``PRODUCT_ID`` names (``GP3``, ``GP3HD``, ``WEBCAM``)."""
    text = (model or "").upper()
    if "WEBCAM" in text:
        return WEBCAM
    if "SIMULATED" in text:
        return "Simulated GP3"
    return "GP3 HD" if "HD" in text else "GP3"


model_from_product = model_name


def calibration_points(n: int) -> list[tuple[float, float]]:
    """The calibration targets for ``n`` points, as fractions of the
    calibration area."""
    if n == 5:
        return list(API_DEFAULT_POINTS)
    if n == 9:
        return list(NINE_POINTS)
    return grid_points(n)


# -- what the user interface reads ---------------------------------------------

@dataclass
class EyeView:
    """The camera's view of the eyes, for display."""

    #: Grey camera image, if the backend has one.
    image: Optional[np.ndarray] = None
    image_size: tuple[int, int] = CAMERA_SIZE
    #: Eye ("L"/"R", the subject's left/right) -> pupil centre x, y as
    #: fractions of the image, pupil diameter as a fraction of its width.
    eyes: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    #: Distance of the eyes from the tracker, in metres.
    distance_m: Optional[float] = None
    fps: float = 0.0


@dataclass
class QualityPoint:
    #: Target, as fractions of the screen.
    x: float
    y: float
    #: Eye -> where its gaze landed (fractions of the screen), if valid.
    gaze: dict[str, tuple[float, float]] = field(default_factory=dict)
    accuracy_deg: Optional[float] = None


@dataclass
class CalibrationQuality:
    """How good a calibration (or a validation of one) is."""

    kind: str = "calibration"            # or "validation"
    points: list[QualityPoint] = field(default_factory=list)
    #: Mean distance of gaze from target, in screen points.
    average_error_px: Optional[float] = None
    #: Accuracy and precision (RMS sample to sample), in degrees.
    accuracy_deg: Optional[float] = None
    max_error_deg: Optional[float] = None
    precision_deg: Optional[float] = None
    valid_points: int = 0
    screen: tuple[int, int] = (0, 0)
    #: The calibration area, as fractions of the screen (x, y, w, h).
    area: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    created: float = field(default_factory=time.time)
    source: str = ""

    @property
    def n_points(self) -> int:
        return len(self.points)

    def summary(self) -> str:
        parts = [f"{self.valid_points}/{self.n_points} points"]
        if self.accuracy_deg is not None:
            parts.append(f"accuracy {self.accuracy_deg:.2f}°")
        if self.precision_deg is not None:
            parts.append(f"precision {self.precision_deg:.2f}°")
        if self.average_error_px is not None:
            parts.append(f"mean error {self.average_error_px:.0f} px")
        return ", ".join(parts)

    def rating(self) -> str:
        """"good", "fair" or "poor", for the colour it is shown in."""
        if self.n_points and self.valid_points < self.n_points * 0.75:
            return "poor"
        if self.accuracy_deg is not None:
            return ("good" if self.accuracy_deg <= 1.0 else
                    "fair" if self.accuracy_deg <= 2.0 else "poor")
        if self.average_error_px is not None and self.screen[0]:
            share = self.average_error_px / self.screen[0]
            return "good" if share <= 0.03 else "fair" if share <= 0.06 \
                else "poor"
        return "fair"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationQuality":
        data = dict(data or {})
        points = [QualityPoint(p["x"], p["y"],
                               {k: tuple(v) for k, v in
                                (p.get("gaze") or {}).items()},
                               p.get("accuracy_deg"))
                  for p in data.pop("points", [])]
        data["screen"] = tuple(data.get("screen") or (0, 0))
        data["area"] = tuple(data.get("area") or (0, 0, 1, 1))
        known = set(cls.__dataclass_fields__)
        return cls(points=points, **{k: v for k, v in data.items()
                                     if k in known})


class CalibrationFailed(RuntimeError):
    pass


class CalibrationDriver:
    """One calibration or validation run, ticked and drawn by the
    calibration window (or only ticked, when the targets are shown on the
    tracker's own computer)."""

    #: Whether this computer shows the targets.
    shows_targets = True
    kind = "calibration"

    def tick(self) -> bool:
        """Advance; True once all targets are done."""
        raise NotImplementedError

    def target(self) -> Optional[tuple[float, float, float]]:
        """The target to draw: x, y (fractions of the screen) and how far
        its collection has got (0 while it travels, then 0 to 1)."""
        raise NotImplementedError

    def area(self) -> tuple[float, float, float, float]:
        return (0.0, 0.0, 1.0, 1.0)

    def progress(self) -> tuple[int, int]:
        """Targets done, and in all."""
        return 0, 0

    def result(self) -> CalibrationQuality:
        """Fit (or evaluate) and return the quality; raises
        CalibrationFailed."""
        raise NotImplementedError

    def cancel(self) -> None:
        pass


class _TimedTargets(CalibrationDriver):
    """Targets shown one after another: ``delay`` s travelling, then
    ``timeout`` s collecting, as OpenGP3 and Gazepoint Control do."""

    def __init__(self, points, delay: float, timeout: float,
                 area=(0.0, 0.0, 1.0, 1.0)):
        self.points = list(points)
        self.delay, self.timeout = delay, timeout
        self._area = area
        self.t0: Optional[float] = None
        self.done = False

    def _to_screen(self, fx, fy):
        x, y, w, h = self._area
        return x + fx * w, y + fy * h

    def _index(self, now: float) -> int:
        return int((now - self.t0) // (self.delay + self.timeout))

    def tick(self) -> bool:
        now = time.time()
        if self.t0 is None:
            self.t0 = now
        if self._index(now) >= len(self.points):
            self.done = True
        return self.done

    def target(self):
        if self.t0 is None or self.done:
            return None
        now = time.time()
        i = self._index(now)
        if i >= len(self.points):
            return None
        el = (now - self.t0) - i * (self.delay + self.timeout)
        x, y = self._to_screen(*self.points[i])
        if el < self.delay:
            px, py = (self._to_screen(*self.points[i - 1]) if i
                      else (0.5, 0.5))
            a = el / self.delay
            a = a * a * (3 - 2 * a)
            return px + (x - px) * a, py + (y - py) * a, 0.0
        return x, y, (el - self.delay) / self.timeout

    def area(self):
        return self._area

    def progress(self):
        if self.t0 is None:
            return 0, len(self.points)
        return min(self._index(time.time()), len(self.points)), \
            len(self.points)


# -- the backends -----------------------------------------------------------------

@dataclass
class ScreenInfo:
    width: int = 1920
    height: int = 1080
    width_cm: Optional[float] = None
    distance_cm: float = 65.0

    def degrees(self, dx_px: float, dy_px: float) -> Optional[float]:
        """Visual angle of an offset from the screen's centre region."""
        if not self.width_cm:
            return None
        cm = math.hypot(dx_px, dy_px) * self.width_cm / self.width
        return math.degrees(math.atan2(cm, self.distance_cm))


class Backend:
    """The interface every GP3 backend has."""

    kind = ""
    #: Whether validation (accuracy in degrees on new targets) is possible.
    can_validate = False

    def __init__(self, config):
        self.config = config
        self.on_row: Callable[[np.ndarray], None] = lambda row: None
        self.on_error: Callable[[str], None] = lambda message: None
        self.screen = ScreenInfo()
        self.info: dict[str, str] = {}
        #: The eye tracker's model ("GP3", "GP3 HD" or "Webcam") and frame
        #: rate, once started.
        self.model = "GP3"
        self.rate = RATE
        #: OpenGP3's warning when the data is degraded or not real (a
        #: webcam, a simulated GP3), else "".
        self.warning = ""

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def eye_view(self) -> EyeView:
        return EyeView()

    def calibrate(self, points: int) -> CalibrationDriver:
        raise NotImplementedError

    def validate(self) -> CalibrationDriver:
        raise CalibrationFailed("This backend cannot validate.")

    @property
    def calibrated(self) -> Optional[bool]:
        """Whether a calibration is active; None when unknown."""
        return None


# ---- simulated -----------------------------------------------------------------

class SimulatedBackend(Backend):
    """Gaze that fixates random points, blinks now and then, and shows a
    synthetic camera image."""

    kind = "simulated"
    can_validate = True

    def __init__(self, config):
        super().__init__(config)
        self.screen = ScreenInfo(1920, 1080, 53.0, 65.0)
        self.info = {"serial": "GP3-SIMULATED", "software": "simulated"}
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._rng = np.random.default_rng()
        self._latest = EyeView()
        self._lock = threading.Lock()
        self._calibrated = False
        self._fix = (0.5, 0.5)
        self._fix_id = 0
        self._fix_start = 0.0
        self._next_fix = 0.0
        self._next_blink = time.time() + 3.0
        self._blink_until = 0.0
        self._blink_id = 0
        self._blink_end_pending = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="gp3-sim")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
            self._thread = None

    @property
    def calibrated(self):
        return self._calibrated

    def _run(self):
        period = 1.0 / RATE
        next_t = clock.now()
        while self._running:
            now = clock.now()
            if now < next_t:
                time.sleep(next_t - now)
                continue
            next_t += period
            self.on_row(self._sample(next_t - period))

    def _sample(self, t: float) -> np.ndarray:
        rng = self._rng
        if t >= self._next_fix:
            self._fix = (rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9))
            self._fix_id += 1
            self._fix_start = t
            self._next_fix = t + rng.uniform(0.25, 1.2)
        blink = t < self._blink_until
        if t >= self._next_blink:
            self._blink_until = t + rng.uniform(0.1, 0.25)
            self._next_blink = t + rng.uniform(2.5, 6.0)
            self._blink_end_pending = True
            blink = True
        v = np.zeros(len(CHANNELS))
        drift = 0.02 * math.sin(t / 7.0)
        eyes = {"L": (0.62 + drift, 0.48), "R": (0.38 + drift, 0.49)}
        pupil = 3.6 + 0.4 * math.sin(t / 3.0)
        view = EyeView(image_size=CAMERA_SIZE, distance_m=0.64 + drift,
                       fps=RATE)
        if not blink:
            noise = rng.normal(0, 0.006, 4)
            gx, gy = self._fix
            lx, ly = gx + noise[0], gy + noise[1]
            rx, ry = gx + noise[2], gy + noise[3]
            if not self._calibrated:
                lx, ly, rx, ry = (0.5 + (lx - 0.5) * 0.6 + 0.1,
                                  ly * 0.7, 0.5 + (rx - 0.5) * 0.6 + 0.08,
                                  ry * 0.7)
            for name, value in (("LPOGX", lx), ("LPOGY", ly), ("LPOGV", 1),
                                ("RPOGX", rx), ("RPOGY", ry), ("RPOGV", 1),
                                ("BPOGX", (lx + rx) / 2),
                                ("BPOGY", (ly + ry) / 2), ("BPOGV", 1),
                                ("FPOGX", gx), ("FPOGY", gy),
                                ("FPOGD", t - self._fix_start),
                                ("FPOGID", self._fix_id), ("FPOGV", 1),
                                ("LPMM", pupil), ("LPMMV", 1),
                                ("RPMM", pupil * 1.02), ("RPMMV", 1),
                                ("LPCX", eyes["L"][0]), ("LPCY", eyes["L"][1]),
                                ("RPCX", eyes["R"][0]), ("RPCY", eyes["R"][1]),
                                ("EYEZ", 0.64 + drift)):
                v[INDEX[name]] = value
            d = pupil / 1000 * 2750 / (0.64 + drift) / CAMERA_SIZE[0]
            view.eyes = {side: (x, y, d) for side, (x, y) in eyes.items()}
            if self._blink_end_pending:
                self._blink_id += 1
                v[INDEX["BKID"]] = self._blink_id
                self._blink_end_pending = False
        v[INDEX["CX"]], v[INDEX["CY"]] = 0.5, 0.5
        with self._lock:
            self._latest = view
        return np.concatenate(([t], v))

    def eye_view(self) -> EyeView:
        with self._lock:
            view = self._latest
        return EyeView(_synthetic_image(view), view.image_size,
                       dict(view.eyes), view.distance_m, view.fps)

    # -- calibration -----------------------------------------------------------

    def calibrate(self, points: int) -> CalibrationDriver:
        return _SimulatedRun(self, calibration_points(points), "calibration")

    def validate(self) -> CalibrationDriver:
        if not self._calibrated:
            raise CalibrationFailed("Calibrate first.")
        pts = list(VALIDATION_POINTS)
        random.shuffle(pts)
        return _SimulatedRun(self, pts, "validation")


class _SimulatedRun(_TimedTargets):
    def __init__(self, backend: SimulatedBackend, points, kind: str):
        cfg = backend.config
        super().__init__(points, cfg.calibration_delay,
                         cfg.calibration_timeout, (0.1, 0.1, 0.8, 0.8))
        self.backend, self.kind = backend, kind

    def result(self) -> CalibrationQuality:
        s = self.backend.screen
        rng = np.random.default_rng()
        q = CalibrationQuality(kind=self.kind, screen=(s.width, s.height),
                               area=self._area, source="simulated")
        errors, degs = [], []
        for fx, fy in self.points:
            x, y = self._to_screen(fx, fy)
            point = QualityPoint(x, y)
            for eye in ("L", "R"):
                if rng.random() < 0.05:
                    continue
                gx, gy = x + rng.normal(0, 0.008), y + rng.normal(0, 0.008)
                point.gaze[eye] = (gx, gy)
                dx, dy = (gx - x) * s.width, (gy - y) * s.height
                errors.append(math.hypot(dx, dy))
                degs.append(s.degrees(dx, dy))
            if point.gaze:
                point.accuracy_deg = float(np.mean(
                    [s.degrees((g[0] - x) * s.width, (g[1] - y) * s.height)
                     for g in point.gaze.values()]))
            q.points.append(point)
        q.valid_points = sum(1 for p in q.points if p.gaze)
        q.average_error_px = float(np.mean(errors)) if errors else None
        q.accuracy_deg = float(np.mean(degs)) if degs else None
        q.max_error_deg = float(np.max(degs)) if degs else None
        q.precision_deg = float(abs(rng.normal(0.35, 0.05)))
        if self.kind == "calibration":
            self.backend._calibrated = True
        return q


def _synthetic_image(view: EyeView) -> np.ndarray:
    """A camera-like image: a face lit in infrared, dark pupils, glints."""
    import cv2
    w, h = view.image_size
    img = np.full((h, w), 38, np.uint8)
    if view.eyes:
        xs = [e[0] for e in view.eyes.values()]
        cx, cy = int(np.mean(xs) * w), int(view.eyes["L"][1] * h) + 60
        cv2.ellipse(img, (cx, cy), (260, 300), 0, 0, 360, 120, -1)
        for x, y, d in view.eyes.values():
            px, py = int(x * w), int(y * h)
            cv2.ellipse(img, (px, py), (34, 18), 0, 0, 360, 170, -1)
            cv2.circle(img, (px, py), 14, 70, -1)
            cv2.circle(img, (px, py), max(2, int(d * w / 2)), 12, -1)
            cv2.circle(img, (px - 4, py - 3), 2, 255, -1)
            cv2.circle(img, (px + 4, py - 3), 2, 255, -1)
    noise = np.random.default_rng().integers(0, 8, img.shape, np.uint8)
    return cv2.add(img, noise)


# ---- this computer -----------------------------------------------------------------

class LocalBackend(Backend):
    """The GP3 on this computer, through OpenGP3."""

    kind = "local"
    can_validate = True

    def __init__(self, config, calibration_folder: Path):
        super().__init__(config)
        self.calibration_folder = Path(calibration_folder)
        self.tracker = None
        self.camera = None
        self.saved_calibration: Optional[Path] = None

    @property
    def webcam(self) -> bool:
        return self.config.tracker == "webcam"

    def _open_camera(self):
        """The chosen tracker's camera: the Gazepoint tracker on USB, never
        a webcam in its place; or the chosen webcam (by default the one
        OpenGP3 prefers, a built-in camera first)."""
        cfg = self.config
        if not self.webcam:
            from opengp3.camera import open_camera
            return open_camera(webcam_fallback=False)
        from opengp3.webcam import open_webcam
        index = None
        if cfg.webcam_id:
            from .video import resolve_camera
            index = resolve_camera(0, cfg.webcam_id, cfg.webcam_name)
        return open_webcam(index)

    def start(self):
        from opengp3 import Calibration, GazeTracker, Screen
        from opengp3.lsl import _serial
        cfg = self.config
        screen = Screen.detect(width_cm=cfg.screen_width_cm or None,
                               distance_cm=cfg.distance_cm,
                               display=cfg.display)
        calibration = None
        path = self._calibration_path()
        if path is not None and path.exists() and \
                calibration_is_webcam(path) == self.webcam:
            try:
                calibration = Calibration.load(str(path))
                if (calibration.screen.width, calibration.screen.height) != \
                        (screen.width, screen.height):
                    calibration = None  # made on another screen
            except Exception:  # noqa: BLE001
                calibration = None
        camera = self._open_camera()
        try:
            tracker = GazeTracker(
                screen, calibration=calibration, smoothing=cfg.smoothing,
                camera=camera, monocular=cfg.monocular or None,
                auto_gain=cfg.auto_gain, input_monitor=False,
                max_diagonal_in=cfg.max_diagonal_in or None,
                rate_hz=cfg.hd_rate)
            tracker.start()
        except BaseException:
            camera.close()
            raise
        self.camera, self.tracker = camera, tracker
        self.model, self.rate = model_name(tracker.model), tracker.rate
        self.warning = tracker.warning or ""
        self.screen = ScreenInfo(screen.width, screen.height, screen.width_cm,
                                 screen.distance_cm)
        import opengp3
        self.info = {"serial": _serial(tracker),
                     "software": f"OpenGP3 {opengp3.__version__}"}
        tracker.add_callback(self._on_sample)

    def _calibration_path(self) -> Optional[Path]:
        name = self.config.calibration_file
        if not name:
            return None
        path = Path(name)
        return path if path.is_absolute() else self.calibration_folder.parent \
            / path

    def stop(self):
        tracker, self.tracker = self.tracker, None
        camera, self.camera = self.camera, None
        if tracker is not None:
            tracker.remove_callback(self._on_sample)
            tracker.stop()
        if camera is not None:
            # Opened here, so the tracker leaves it open.
            camera.close()

    @property
    def calibrated(self):
        return self.tracker is not None and \
            self.tracker.calibration is not None

    def _on_sample(self, s):
        from opengp3.lsl import sample_values
        # OpenGP3 stamps frames with time.time() as they are captured.
        self.on_row(np.array([clock.from_wall(s.t),
                              *sample_values(s, self.tracker)]))

    def eye_view(self) -> EyeView:
        tracker = self.tracker
        if tracker is None:
            return EyeView()
        img, s = tracker.latest_frame()
        eyes = {}
        distance = None
        if s is not None:
            w = tracker.image_size[0]
            for side, m in s.metrics.items():
                if m.valid:
                    eyes[side] = (m.cx, m.cy, m.diameter_px / w)
                    distance = distance or (m.z_m or None)
        return EyeView(img, tuple(tracker.image_size), eyes, distance,
                       tracker.fps)

    def calibrate(self, points: int) -> CalibrationDriver:
        if self.tracker is None:
            raise CalibrationFailed("The GP3 is not running.")
        cfg = self.config
        run = self.tracker.start_calibration(
            calibration_points(points), delay=cfg.calibration_delay,
            timeout=cfg.calibration_timeout, retry=True)
        return _LocalRun(self, run, "calibration")

    def validate(self) -> CalibrationDriver:
        from opengp3.tracker import CalibrationRun
        tracker = self.tracker
        if tracker is None or tracker.calibration is None:
            raise CalibrationFailed("Calibrate first.")
        cfg = self.config
        run = CalibrationRun(tracker, list(VALIDATION_POINTS),
                             tracker.calibration.area,
                             cfg.calibration_delay, cfg.calibration_timeout,
                             retry=False)
        random.shuffle(run.order)
        return _LocalRun(self, run, "validation")

    def save_calibration(self, calibration) -> Path:
        self.calibration_folder.mkdir(parents=True, exist_ok=True)
        prefix = "webcam" if self.webcam else "gp3"
        path = self.calibration_folder / \
            f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        calibration.save(str(path))
        self.saved_calibration = path
        return path


def calibration_is_webcam(path: Path) -> bool:
    """Whether a calibration saved here was made with a webcam: a webcam's
    gaze features are not a Gazepoint tracker's, so neither's calibration
    fits the other."""
    return Path(path).name.startswith("webcam_")


class _LocalRun(CalibrationDriver):
    """Drives an OpenGP3 :class:`~opengp3.tracker.CalibrationRun` without
    its OpenCV window: the run keeps time and labels the samples, the
    calibration window draws."""

    def __init__(self, backend: LocalBackend, run, kind: str):
        self.backend, self.run, self.kind = backend, run, kind
        screen = backend.tracker.screen
        self._w, self._h = screen.width, screen.height
        a = run.area
        self._area = (a.x / self._w, a.y / self._h, a.w / self._w,
                      a.h / self._h)

    def tick(self) -> bool:
        return self.run.tick(None)

    def target(self):
        run = self.run
        current = run.current
        if current is None or run.done or run._t_point is None:
            return None
        _, (fx, fy) = current
        tx, ty = run.area.to_screen(fx, fy)
        el = time.time() - run._t_point
        if el < run.delay:
            a = el / run.delay if run.delay else 1.0
            a = a * a * (3 - 2 * a)
            px, py = run._prev_xy
            return ((px + (tx - px) * a) / self._w,
                    (py + (ty - py) * a) / self._h, 0.0)
        progress = (el - run.delay) / run.timeout if run.timeout else 1.0
        return tx / self._w, ty / self._h, min(1.0, progress)

    def area(self):
        return self._area

    def progress(self):
        return max(0, self.run._i), len(self.run.order)

    def cancel(self):
        self.run.close()

    def result(self) -> CalibrationQuality:
        from opengp3 import CalibrationError
        from opengp3.calibration import evaluate
        tracker, run = self.backend.tracker, self.run
        screen = tracker.screen
        sinfo = self.backend.screen
        q = CalibrationQuality(kind=self.kind, screen=(self._w, self._h),
                               area=self._area, source="OpenGP3")
        if self.kind == "calibration":
            try:
                cal = tracker.finish_calibration(run)
            except CalibrationError as error:
                run.close()
                raise CalibrationFailed(str(error)) from error
            path = self.backend.save_calibration(cal)
            self.backend.config.calibration_file = \
                str(path.relative_to(self.backend.calibration_folder.parent))
            degs = []
            for p in cal.points:
                point = QualityPoint(p.x / self._w, p.y / self._h)
                here = []
                for eye, g in p.gaze.items():
                    if g and p.valid.get(eye):
                        point.gaze[eye] = (g[0] / self._w, g[1] / self._h)
                        here.append(sinfo.degrees(g[0] - p.x, g[1] - p.y))
                here = [d for d in here if d is not None]
                if here:
                    point.accuracy_deg = float(np.mean(here))
                degs += here
                q.points.append(point)
            q.valid_points = cal.valid_points
            err = cal.average_error
            q.average_error_px = float(err) if err == err else None
            degs = [d for d in degs if d is not None]
            if degs:
                q.accuracy_deg = float(np.mean(degs))
                q.max_error_deg = float(np.max(degs))
            return q
        run._collect()
        run.close()
        labelled = run.labelled()
        gaze = [(i, x, y, s.t, s.raw_x, s.raw_y) for i, x, y, s in labelled
                if s.valid]
        result = evaluate(gaze, screen)
        targets = {}
        for i, x, y, _ in labelled:
            targets[i] = (x, y)
        for i, (x, y) in sorted(targets.items()):
            point = QualityPoint(x / self._w, y / self._h)
            match = next((p for p in result.points if p.target == i), None)
            if match is not None:
                point.gaze["B"] = (match.gaze_x / self._w,
                                   match.gaze_y / self._h)
                point.accuracy_deg = match.accuracy_deg
            q.points.append(point)
        q.valid_points = len(result.points)
        offsets = [p.offset for p in result.points]
        q.average_error_px = float(np.mean(offsets)) if offsets else None
        q.accuracy_deg = result.accuracy_deg
        q.precision_deg = result.precision_rms_s2s_deg
        accs = [p.accuracy_deg for p in result.points
                if p.accuracy_deg is not None]
        q.max_error_deg = float(max(accs)) if accs else None
        if not result.points:
            raise CalibrationFailed("No gaze at the validation targets.")
        return q


# ---- another computer ---------------------------------------------------------------

#: Open Gaze API fields asked for.
REMOTE_FIELDS = ("TIME", "COUNTER", "POG_FIX", "POG_LEFT", "POG_RIGHT",
                 "POG_BEST", "PUPIL_LEFT", "PUPIL_RIGHT", "EYE_LEFT",
                 "EYE_RIGHT", "PUPILMM", "BLINK", "CURSOR")


class RemoteBackend(Backend):
    """A GP3 served over the Open Gaze API by another computer, through
    OpenGP3's client. Capture times are the server's TIME moved onto this
    computer's clock by the smallest delay seen."""

    kind = "remote"
    can_validate = False
    TIMEOUT_S = 5.0

    def __init__(self, config):
        super().__init__(config)
        self.client = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._send_lock = threading.Lock()
        self._waiting: dict[str, tuple[threading.Event, list]] = {}
        self._clock = ClockMap(window_s=30.0)
        self._view = EyeView()
        self._view_lock = threading.Lock()
        self._camera = CAMERA_SIZE
        self._cal_listeners: list[Callable[[str, dict], None]] = []
        self._fps_n, self._fps_t, self.fps = 0, clock.now(), 0.0
        self.calibration_delay = 0.5
        self.calibration_timeout = 1.25
        self._calibrated: Optional[bool] = None

    def start(self):
        from opengp3 import OpenGazeClient
        cfg = self.config
        client = OpenGazeClient(cfg.host, int(cfg.port),
                                timeout=self.TIMEOUT_S)
        self.client = client
        try:
            def get(ident, *keys):
                try:
                    a = client.get(ident)
                    return [a.get(k, "") for k in keys]
                except (ValueError, OSError):
                    return ["" for _ in keys]
            impl, version = get("API_ID", "IMPL", "VALUE")
            (serial,) = get("SERIAL_ID", "VALUE")
            # OpenGP3 0.5 adds WARNING when the data is degraded or not
            # real (a webcam, a simulated GP3).
            product, rate, warning = get("PRODUCT_ID", "VALUE", "RATE",
                                         "WARNING")
            cw, ch = get("CAMERA_SIZE", "WIDTH", "HEIGHT")
            sw, sh = get("SCREEN_SIZE", "WIDTH", "HEIGHT")
            (delay,) = get("CALIBRATE_DELAY", "VALUE")
            (timeout,) = get("CALIBRATE_TIMEOUT", "VALUE")
            valid = get("CALIBRATE_RESULT_SUMMARY", "VALID_POINTS")[0]
            self._calibrated = bool(valid and float(valid) > 0) \
                if valid else None
            self.model = model_name(product)
            self.warning = warning
            try:
                self.rate = float(rate)
            except ValueError:
                self.rate = nominal_rate(self.model)
            self.info = {"serial": serial, "product": product,
                         "software": impl or f"Open Gaze API {version}",
                         "server": f"{cfg.host}:{cfg.port}"}
            try:
                self._camera = (int(float(cw)), int(float(ch)))
            except ValueError:
                pass
            try:
                self.screen = ScreenInfo(int(float(sw)), int(float(sh)),
                                         cfg.screen_width_cm or None,
                                         cfg.distance_cm)
            except ValueError:
                self.screen = ScreenInfo(1920, 1080,
                                         cfg.screen_width_cm or None,
                                         cfg.distance_cm)
            for name in REMOTE_FIELDS:
                try:
                    client.set(f"ENABLE_SEND_{name}", STATE=1)
                except ValueError:
                    pass  # a server without this field
            client.set("ENABLE_SEND_DATA", STATE=1)
        except BaseException:
            client.close()
            self.client = None
            raise
        client.sock.settimeout(0.5)
        self._running = True
        self._thread = threading.Thread(target=self._read, daemon=True,
                                        name="gp3-remote")
        self._thread.start()

    def stop(self):
        self._running = False
        client = self.client
        if client is not None:
            try:
                with self._send_lock:
                    client._send("SET", "ENABLE_SEND_DATA", STATE=0)
            except OSError:
                pass
        if self._thread:
            self._thread.join(timeout=1.5)
            self._thread = None
        if client is not None:
            client.close()
        self.client = None

    @property
    def calibrated(self):
        return self._calibrated

    # -- the connection -------------------------------------------------------

    def _read(self):
        client = self.client
        pending = list(client.pending)
        client.pending.clear()
        for tag, attrs in pending:
            self._handle(tag, attrs, clock.now())
        while self._running:
            try:
                tag, attrs = client._next()
            except socket.timeout:
                continue
            except (OSError, ConnectionError) as error:
                if self._running:
                    self._running = False
                    self.on_error(f"Lost the connection to the GP3 server: "
                                  f"{error}")
                return
            self._handle(tag, attrs, clock.now())

    def _handle(self, tag: str, a: dict, received: float):
        if tag == "REC":
            self._record(a, received)
        elif tag in ("ACK", "NACK"):
            waiter = self._waiting.pop(a.get("ID", "").upper(), None)
            if waiter is not None:
                waiter[1].append((tag, a))
                waiter[0].set()
        elif tag == "CAL":
            for listener in list(self._cal_listeners):
                listener(a.get("ID", ""), a)

    def command(self, verb: str, ident: str, **attrs) -> dict:
        """Send a GET/SET and wait for its answer; raises ValueError on a
        NACK, TimeoutError when none comes."""
        client = self.client
        if client is None:
            raise ConnectionError("Not connected to the GP3 server.")
        event, box = threading.Event(), []
        self._waiting[ident.upper()] = (event, box)
        with self._send_lock:
            client._send(verb, ident, **attrs)
        if not event.wait(self.TIMEOUT_S):
            self._waiting.pop(ident.upper(), None)
            raise TimeoutError(f"No answer to {verb} {ident}.")
        tag, answer = box[0]
        if tag == "NACK":
            raise ValueError(f"The server refused {verb} {ident}.")
        return answer

    def _record(self, a: dict, received: float):
        def f(key, default=0.0):
            try:
                return float(a.get(key, default))
            except ValueError:
                return default
        server_time = f("TIME", float("nan"))
        if server_time == server_time:
            # The smallest delay seen lately is the best estimate of the
            # offset between the clocks.
            t = self._clock(server_time, received)
        else:
            t = received
        v = np.zeros(len(CHANNELS))
        for name in CHANNELS:
            if name in a:
                v[INDEX[name]] = f(name)
        zs = [f(f"{s}EYEZ") for s in ("L", "R")
              if f(f"{s}PUPILV") and f(f"{s}EYEZ")]
        v[INDEX["EYEZ"]] = zs[0] if zs else 0.0
        eyes = {}
        for side in ("L", "R"):
            if f(f"{side}PV"):
                eyes[side] = (f(f"{side}PCX"), f(f"{side}PCY"),
                              f(f"{side}PD") / max(1, self._camera[0]))
        self._fps_n += 1
        if received - self._fps_t >= 1.0:
            self.fps = self._fps_n / (received - self._fps_t)
            self._fps_n, self._fps_t = 0, received
        with self._view_lock:
            self._view = EyeView(None, self._camera, eyes,
                                 v[INDEX["EYEZ"]] or None, self.fps)
        self.on_row(np.concatenate(([t], v)))

    def eye_view(self) -> EyeView:
        with self._view_lock:
            return self._view

    # -- calibration ---------------------------------------------------------------

    def calibrate(self, points: int) -> CalibrationDriver:
        if self.client is None:
            raise CalibrationFailed("Not connected to the GP3 server.")
        return _RemoteRun(self, points,
                          self.config.remote_calibration_display == "here")


class _RemoteRun(CalibrationDriver):
    """A calibration the server runs: it keeps time and sends a CAL record
    as each target starts and ends, then the results."""

    def __init__(self, backend: RemoteBackend, points: int, here: bool):
        self.backend = backend
        self.shows_targets = here
        self.points = calibration_points(points)
        self.result_attrs: Optional[dict] = None
        self.current: Optional[tuple[float, float, float]] = None
        self.started_points = 0
        self.error: Optional[str] = None
        self._lock = threading.Lock()
        b = backend
        b._cal_listeners.append(self._cal)
        try:
            for ident in ("CALIBRATE_DELAY", "CALIBRATE_TIMEOUT"):
                try:
                    value = float(b.command("GET", ident).get("VALUE", ""))
                    if ident == "CALIBRATE_DELAY":
                        b.calibration_delay = value
                    else:
                        b.calibration_timeout = value
                except (ValueError, TimeoutError):
                    pass
            if points == 5:
                b.command("SET", "CALIBRATE_RESET")
            else:
                b.command("SET", "CALIBRATE_CLEAR")
                for x, y in self.points:
                    b.command("SET", "CALIBRATE_ADDPOINT", X=f"{x:.4f}",
                              Y=f"{y:.4f}")
            b.command("SET", "CALIBRATE_SHOW", STATE=0 if here else 1)
            b.command("SET", "CALIBRATE_START", STATE=1)
        except BaseException as error:
            b._cal_listeners.remove(self._cal)
            raise CalibrationFailed(f"The server did not start the "
                                    f"calibration: {error}") from error

    def _cal(self, ident: str, a: dict):
        with self._lock:
            if ident == "CALIB_START_PT":
                try:
                    self.current = (float(a.get("CALX", 0.5)),
                                    float(a.get("CALY", 0.5)), time.time())
                except ValueError:
                    pass
                self.started_points += 1
            elif ident == "CALIB_RESULT":
                self.result_attrs = dict(a)
                self.current = None

    def tick(self) -> bool:
        return self.result_attrs is not None

    def target(self):
        with self._lock:
            current = self.current
        if current is None:
            return None
        x, y, t0 = current
        b = self.backend
        el = time.time() - t0
        progress = 0.0 if el < b.calibration_delay else \
            (el - b.calibration_delay) / max(0.01, b.calibration_timeout)
        return x, y, min(1.0, progress)

    def progress(self):
        return self.started_points, len(self.points)

    def cancel(self):
        b = self.backend
        try:
            b.command("SET", "CALIBRATE_START", STATE=0)
            b.command("SET", "CALIBRATE_SHOW", STATE=0)
        except Exception:  # noqa: BLE001
            pass
        self._detach()

    def _detach(self):
        try:
            self.backend._cal_listeners.remove(self._cal)
        except ValueError:
            pass

    def result(self) -> CalibrationQuality:
        self._detach()
        b = self.backend
        a = self.result_attrs or {}
        s = b.screen
        q = CalibrationQuality(kind="calibration", screen=(s.width, s.height),
                               source=b.info.get("software", "Open Gaze API"))
        k, errors, degs = 1, [], []
        while f"CALX{k}" in a:
            def f(key):
                try:
                    return float(a.get(key, 0))
                except ValueError:
                    return 0.0
            x, y = f(f"CALX{k}"), f(f"CALY{k}")
            point = QualityPoint(x, y)
            for eye in ("L", "R"):
                if f(f"{eye}V{k}"):
                    gx, gy = f(f"{eye}X{k}"), f(f"{eye}Y{k}")
                    point.gaze[eye] = (gx, gy)
                    dx, dy = (gx - x) * s.width, (gy - y) * s.height
                    errors.append(math.hypot(dx, dy))
                    degs.append(s.degrees(dx, dy))
            q.points.append(point)
            k += 1
        q.valid_points = sum(1 for p in q.points if p.gaze)
        try:
            summary = b.command("GET", "CALIBRATE_RESULT_SUMMARY")
            q.average_error_px = float(summary.get("AVE_ERROR", "nan"))
            q.valid_points = int(float(summary.get("VALID_POINTS",
                                                   q.valid_points)))
        except Exception:  # noqa: BLE001
            q.average_error_px = float(np.mean(errors)) if errors else None
        degs = [d for d in degs if d is not None]
        if degs:
            q.accuracy_deg = float(np.mean(degs))
            q.max_error_deg = float(np.max(degs))
        if not self.shows_targets and b.config.hide_remote_calibration:
            try:
                b.command("SET", "CALIBRATE_SHOW", STATE=0)
            except Exception:  # noqa: BLE001
                pass
        b._calibrated = q.valid_points > 0
        if not q.valid_points:
            raise CalibrationFailed("No point could be calibrated.")
        return q
