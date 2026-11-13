"""The Gazepoint GP3 eye tracker, through OpenGP3 and g.Pype.

Whatever the backend (this computer, another one over the Open Gaze API,
or simulated; see :mod:`.gp3_backends`), its samples enter the session
pipeline through :class:`GazeSource`::

    GazeSource ─┬─ SweepTap        (the gaze view)
                └─ EyetrackWriter  (BIDS physio files, while recording)
"""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import gpype as gp
import numpy as np
from gpype.common.constants import Constants

from .. import __version__
from ..bids import NA, GzipTsvWriter, write_json
from ..pipeline import BidsWriter, PushSource, SweepTap
from .base import (IDLE, RUNNING, STARTING, STOPPING, Device, DeviceConfig,
                   RecordedFiles, RecordingTarget, Timeline,
                   start_time)
from .gp3_backends import (CHANNELS, DESCRIPTIONS, INDEX, MODELS, UNITS,
                           WEBCAM, nominal_rate, Backend, CalibrationQuality,
                           LocalBackend, RemoteBackend, SimulatedBackend)

PORT_OUT = Constants.Defaults.PORT_OUT
#: The pipeline's channels: seconds since the source started, then the
#: GP3's channels.
PIPELINE_CHANNELS = ["TIME", *CHANNELS]
COL = {name: i for i, name in enumerate(PIPELINE_CHANNELS)}

#: Where BIDS eye-tracking files go: beh/, even beside EEG.
DATATYPE = "beh"

MODES = {"local": "On this computer (OpenGP3 driver)",
         "remote": "On another computer (Open Gaze API)"}
#: What tracks the eyes on this computer.
TRACKERS = {"gazepoint": "Gazepoint GP3 or GP3 HD",
            "webcam": "Webcam (low accuracy)"}


@dataclass
class GP3Config(DeviceConfig):
    #: "local" or "remote".
    mode: str = "local"
    simulate: bool = False
    #: On this computer: "gazepoint" (the GP3 or GP3 HD on USB) or
    #: "webcam" (OpenGP3's webcam eye tracking, 2-5° instead of 0.5°).
    tracker: str = "gazepoint"
    #: The webcam, by the system's identifier (as the Video Stream keeps
    #: it), and its name; empty lets OpenGP3 choose, a built-in one first.
    webcam_id: str = ""
    webcam_name: str = ""
    # remote
    host: str = "127.0.0.1"
    port: int = 4242
    #: Where calibration targets appear in remote mode: "tracker" (the
    #: server's screen) or "here".
    remote_calibration_display: str = "tracker"
    #: Close the server's calibration window once its results are in.
    hide_remote_calibration: bool = True
    # screen
    display: int = 0
    #: Physical screen width; 0 asks the operating system.
    screen_width_cm: float = 0.0
    distance_cm: float = 65.0
    #: Calibrate within a centred area this large on bigger screens (the
    #: GP3 is specified for up to 22"); 0 is the whole screen.
    max_diagonal_in: float = 22.0
    # tracking
    calibration_points: int = 9
    calibration_delay: float = 0.6
    calibration_timeout: float = 1.2
    #: "", "L" or "R": track one eye only.
    monocular: str = ""
    auto_gain: bool = False
    smoothing: bool = True
    #: The GP3 HD's frame rate, 150 or 60 Hz (the GP3 runs at 60 Hz).
    hd_rate: int = 150
    #: The model last connected ("GP3", "GP3 HD" or "Webcam"), and a
    #: webcam's frame rate: the session pipeline is laid out for its frame
    #: rate before the tracker is opened.
    model: str = "GP3"
    webcam_rate: float = 30.0
    # the view
    mirror_view: bool = True
    #: The last calibration made here (local mode), relative to the
    #: project folder.
    calibration_file: str = ""
    #: The last calibration's (and validation's) quality, for the view.
    last_calibration: dict = field(default_factory=dict)
    last_validation: dict = field(default_factory=dict)


class GazeSource(PushSource):
    """The GP3's samples, pushed by its backend: one row per camera
    frame."""

    def __init__(self, rate: float, **kwargs):
        super().__init__(CHANNELS, [UNITS[c] for c in CHANNELS], rate,
                         **kwargs)


# -- BIDS ----------------------------------------------------------------------------

#: Columns of the per-eye physio files.
EYE_COLUMNS = ["timestamp", "x_coordinate", "y_coordinate", "pupil_size",
               "gaze_valid", "pupil_valid", "pupil_x_camera",
               "pupil_y_camera"]
EVENT_COLUMNS = ["onset", "duration", "trial_type", "blink", "message"]


class EyetrackWriter(BidsWriter):
    """Writes the BIDS eye-tracking files while a recording is open: one
    physio file per eye, one with every GP3 channel, and the fixations
    and blinks as physioevents. A g.Pype writer of brAIn dump's
    (:class:`~braindump.pipeline.writers.BidsWriter`): the session begins
    and ends it, at a sample, with its ``begin`` and ``end`` actions;
    ``meta`` gives the source's epoch and the screen's size in pixels."""

    extension = ".tsv.gz"

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.epoch = 0.0
        self.screen = (1920, 1080)
        self.first_sample: Optional[float] = None
        self.last_sample: Optional[float] = None
        self.paths: dict[str, Path] = {}
        self._writers: dict[str, GzipTsvWriter] = {}
        self.events: list[tuple[float, float, str, int]] = []
        self.valid = {"L": 0, "R": 0}

    @staticmethod
    def paths_for(eye1: Path) -> dict[str, Path]:
        name = eye1.name
        return {
            "eye1": eye1,
            "eye2": eye1.with_name(name.replace("recording-eye1",
                                                "recording-eye2")),
            "gaze": eye1.with_name(name.replace("recording-eye1",
                                                "recording-gaze")),
            "events1": eye1.with_name(name.replace("_physio.",
                                                   "_physioevents.")),
            "events2": eye1.with_name(name.replace(
                "recording-eye1", "recording-eye2").replace(
                "_physio.", "_physioevents.")),
        }

    def _open(self, path, meta):
        self.epoch = float(meta.get("epoch", 0.0))
        self.screen = tuple(meta.get("screen") or self.screen)
        self.paths = self.paths_for(Path(path))
        self._writers = {k: GzipTsvWriter(self.paths[k])
                         for k in ("eye1", "eye2", "gaze")}
        self.first_sample = self.last_sample = None
        self.events = []
        self.valid = {"L": 0, "R": 0}
        self._fix_id = None
        self._fix_start = None
        self._fix_last = None
        self._lost_since = None

    def _write(self, block, first_index):
        w, h = self.screen
        for row in block:
            t = self.epoch + float(row[COL["TIME"]])
            if self.first_sample is None:
                self.first_sample = t
            self.last_sample = t
            for side, key in (("L", "eye1"), ("R", "eye2")):
                gv = bool(row[COL[f"{side}POGV"]])
                pv = bool(row[COL[f"{side}PMMV"]])
                self.valid[side] += gv
                self._writers[key].write([
                    t,
                    row[COL[f"{side}POGX"]] * w if gv else None,
                    row[COL[f"{side}POGY"]] * h if gv else None,
                    row[COL[f"{side}PMM"]] if pv else None,
                    int(gv), int(pv),
                    row[COL[f"{side}PCX"]] if pv else None,
                    row[COL[f"{side}PCY"]] if pv else None])
            self._writers["gaze"].write([t, *(float(v) for v in row[1:])])
            self._track_events(t, row)

    def _track_events(self, t: float, row) -> None:
        fix_valid = bool(row[COL["FPOGV"]])
        fix_id = int(row[COL["FPOGID"]])
        if self._fix_id is not None and (fix_id != self._fix_id
                                         or not fix_valid):
            self.events.append((self._fix_start,
                                self._fix_last - self._fix_start,
                                "fixation", 0))
            self._fix_id = None
        if fix_valid and self._fix_id is None:
            self._fix_id, self._fix_start = fix_id, t
        if fix_valid:
            self._fix_last = t
        # A blink: the eyes lost, until the sample the GP3 counts it on.
        if not row[COL["BPOGV"]] and not row[COL["LPMMV"]] \
                and not row[COL["RPMMV"]]:
            if self._lost_since is None:
                self._lost_since = t
        else:
            if row[COL["BKID"]] and self._lost_since is not None:
                self.events.append((self._lost_since, t - self._lost_since,
                                    "blink", 1))
            self._lost_since = None

    def _close(self):
        if self._fix_id is not None and self._fix_last is not None:
            self.events.append((self._fix_start,
                                self._fix_last - self._fix_start,
                                "fixation", 0))
        for writer in self._writers.values():
            writer.close()
        self._writers = {}
        self.events.sort()
        for key in ("events1", "events2"):
            out = GzipTsvWriter(self.paths[key])
            for onset, duration, kind, blink in self.events:
                out.write([onset, round(duration, 4), kind, blink, None])
            out.close()
        return {"first": self.first_sample, "last": self.last_sample,
                "valid": dict(self.valid)}


# -- the device ----------------------------------------------------------------------

@dataclass
class Graph:
    """The GP3's nodes in the session pipeline."""
    source: GazeSource
    tap: SweepTap
    writer: EyetrackWriter


class GP3Device(Device):
    TYPE = "gp3"
    NAME = "Gazepoint GP3"
    MANUFACTURER = "Gazepoint"
    MODEL = "GP3"
    DESCRIPTION = "Eye tracker (GP3 60 Hz, GP3 HD 150 Hz, or a webcam), " \
                  "on this computer or over the Open Gaze API (OpenGP3)"
    MODALITY = "eyetrack"
    CONFIG = GP3Config

    def __init__(self, config: GP3Config, project_folder: Path,
                 simulate: bool = False):
        super().__init__(config, project_folder, simulate)
        self.backend: Optional[Backend] = None
        self.graph: Optional[Graph] = None
        self._target: Optional[RecordingTarget] = None
        #: Calibrations made since the GP3 was connected.
        self.calibrations: list[CalibrationQuality] = []
        self.validations: list[CalibrationQuality] = []

    @property
    def calibration_folder(self) -> Path:
        return self.project_folder / "calibrations"

    def make_backend(self) -> Backend:
        if self.simulated:
            return SimulatedBackend(self.config)
        if self.config.mode == "remote":
            return RemoteBackend(self.config)
        return LocalBackend(self.config, self.calibration_folder)

    # -- the session pipeline ----------------------------------------------------

    def build(self, pipeline: gp.Pipeline) -> None:
        cfg = self.config
        model = cfg.model if cfg.model in MODELS and not self.simulated \
            else "GP3"
        if cfg.mode == "local" and not self.simulated:
            # The tracker chosen since may not be the one last connected.
            if cfg.tracker == "webcam":
                model = WEBCAM
            elif model == WEBCAM:
                model = "GP3"
        source = GazeSource(nominal_rate(model, cfg.hd_rate, cfg.webcam_rate),
                            name=self.node_name("source"))
        tap = SweepTap(seconds=10, name=self.node_name("view"))
        writer = EyetrackWriter(self.node_name("bids"))
        pipeline.connect(source, tap)
        pipeline.connect(source, writer)
        self.graph = Graph(source, tap, writer)

    def opened(self) -> None:
        backend = self.make_backend()
        backend.on_row = self.graph.source.push
        backend.on_error = self._fail
        backend.start()
        self.backend = backend
        self.calibrations, self.validations = [], []
        if not self.simulated:
            # The next connection lays its pipeline out for this model.
            self.config.model = backend.model
            if backend.model == WEBCAM:
                self.config.webcam_rate = float(backend.rate)

    def closing(self) -> None:
        backend, self.backend = self.backend, None
        if backend is not None:
            backend.stop()

    def closed(self) -> None:
        self.graph = None
        self._target = None

    # -- calibration ---------------------------------------------------------

    def calibration_done(self, quality: CalibrationQuality) -> None:
        """Keep a finished calibration's (or validation's) quality."""
        if quality.kind == "validation":
            self.validations.append(quality)
            self.config.last_validation = quality.to_dict()
        else:
            self.calibrations.append(quality)
            self.config.last_calibration = quality.to_dict()
            self.config.last_validation = {}

    @property
    def last_calibration(self) -> Optional[CalibrationQuality]:
        data = self.config.last_calibration
        return CalibrationQuality.from_dict(data) if data else None

    @property
    def last_validation(self) -> Optional[CalibrationQuality]:
        data = self.config.last_validation
        return CalibrationQuality.from_dict(data) if data else None

    # -- recording -----------------------------------------------------------

    @property
    def recording(self) -> bool:
        graph = self.graph
        return bool(graph and graph.writer.recording)

    def begin_recording(self, target: RecordingTarget) -> list[Path]:
        with self._lock:
            graph, backend = self.graph, self.backend
            if graph is None or backend is None:
                raise RuntimeError("The GP3 is not running.")
            # Eye tracking is behavioural data: beh/, whatever else is
            # recorded with it.
            path = target.dataset.path(target.entities, DATATYPE,
                                       "physio", ".tsv.gz", recording="eye1")
            self.invoke(graph.writer, "begin", {
                "path": str(path),
                "meta": {"epoch": graph.source.epoch,
                         "screen": [backend.screen.width,
                                    backend.screen.height]}})
            self._target = target
            return list(EyetrackWriter.paths_for(path).values())

    def end_recording(self) -> RecordedFiles:
        with self._lock:
            graph, target, backend = self.graph, self._target, self.backend
            if graph is None or target is None or not graph.writer.recording:
                return RecordedFiles()
            # The frames already on their way from the tracker.
            time.sleep(0.05)
            summary = self.invoke(graph.writer, "end")
            self._target = None
            writer = graph.writer
            paths = dict(writer.paths)
            first, last = summary.get("first"), summary.get("last")
            samples = int(summary.get("samples") or 0)
            valid = dict(summary.get("valid") or writer.valid)
            screen = backend.screen if backend else None
            info = dict(backend.info) if backend else {}
            if backend is not None:
                info["model"], info["rate"] = backend.model, backend.rate
                info["warning"] = backend.warning
            kind = backend.kind if backend else ""

        def finalize(reference: Optional[Timeline]) -> None:
            self._write_sidecars(target, paths, first, last, samples, valid,
                                 screen, info, kind, reference)

        moment = datetime.fromtimestamp(first or target.started)
        data = [paths[k] for k in ("eye1", "eye2", "gaze")]
        events_meta = {}
        if screen is not None:
            size, _ = screen_size_m(screen)
            events_meta["StimulusPresentation"] = {
                "ScreenDistance": round(screen.distance_cm / 100, 3),
                "ScreenOrigin": ["top", "left"],
                "ScreenResolution": [screen.width, screen.height],
                "ScreenSize": size,
            }
        sidecars = [p.with_name(p.name.replace(".tsv.gz", ".json"))
                    for p in (*data, paths["events1"], paths["events2"])]
        return RecordedFiles(
            scans=[(p, moment) for p in data],
            files=[*data, paths["events1"], paths["events2"], *sidecars],
            first_sample=first, finalize=finalize, samples=samples,
            events_metadata=events_meta)

    def _write_sidecars(self, target, paths, first, last, samples, valid,
                        screen, info, kind, reference) -> None:
        cfg = self.config
        start = start_time(reference, first)
        duration = (last - first) if (first is not None and last) else 0.0
        cal = self.calibrations[-1] if self.calibrations else \
            self.last_calibration
        val = self.validations[-1] if self.validations else None
        software = [f"brAIn dump {__version__}", f"g.Pype {gp.__version__}"]
        if info.get("software"):
            software.append(info["software"])
        model = info.get("model") or self.MODEL
        webcam = model == WEBCAM
        setup = {"local": f"{model} on the recording computer "
                          + ("(OpenGP3 webcam eye tracking, MediaPipe face "
                             "landmarks)" if webcam else "(OpenGP3 driver)"),
                 "remote": f"{model} served over the Open Gaze API by "
                           f"{info.get('server', '')}",
                 "simulated": "simulated GP3"}.get(kind, kind)
        if info.get("warning"):
            # OpenGP3's own words: a webcam's (or a simulated tracker's)
            # data must not pass for a Gazepoint tracker's.
            setup += f". WARNING: {info['warning']}"
        common = {
            "SamplingFrequency": info.get("rate") or nominal_rate(model),
            "StartTime": round(start, 6),
            "Manufacturer": NA if webcam else self.MANUFACTURER,
            "ManufacturersModelName": model,
            "DeviceSerialNumber": info.get("serial") or NA,
            "SoftwareVersions": ", ".join(software),
            "TaskName": target.entities.task,
            "RecordingDuration": round(duration, 3),
            "TrackerSetup": setup,
            "SimulatedDevice": self.simulated,
            **target.common_metadata,
        }
        eye_meta = {
            **common,
            "PhysioType": "eyetrack",
            "Columns": EYE_COLUMNS,
            "timestamp": {"Description": "Capture time of the camera frame, "
                                         "seconds since 1970-01-01 UTC on "
                                         "the recording computer's clock",
                          "Units": "s"},
            "x_coordinate": {"Description": "Gaze position on the screen, "
                                            "from the left edge",
                             "Units": "pixel"},
            "y_coordinate": {"Description": "Gaze position on the screen, "
                                            "from the top edge",
                             "Units": "pixel"},
            "pupil_size": {"Description": "Pupil diameter", "Units": "mm"},
            "gaze_valid": {"Description": "1 when the gaze was tracked",
                           "Levels": {"0": "lost", "1": "valid"}},
            "pupil_valid": {"Description": "1 when the pupil was found",
                            "Levels": {"0": "lost", "1": "valid"}},
            "pupil_x_camera": {"Description": "Pupil centre in the camera "
                                              "image, fraction of its width",
                               "Units": "arbitrary"},
            "pupil_y_camera": {"Description": "Pupil centre in the camera "
                                              "image, fraction of its height",
                               "Units": "arbitrary"},
            "SampleCoordinateUnits": "pixel",
            "SampleCoordinateSystem": "gaze-on-screen",
            "EnvironmentCoordinates": "top-left",
            "EyeTrackingMethod": "P-CR",
            "PupilFitMethod": "ellipse",
            "ScreenResolution": [screen.width, screen.height] if screen
            else NA,
            "ScreenDistance": round(screen.distance_cm / 100, 3) if screen
            else NA,
            "ScreenSize": screen_size_m(screen)[0] if screen else NA,
            "ScreenOrigin": ["top", "left"],
            "EyeTrackerDistance": round(screen.distance_cm / 100, 3)
            if screen else NA,
            "CalibrationCount": len(self.calibrations),
            "CalibrationType": f"HV{cal.n_points}" if cal else NA,
        }
        if cal is not None:
            # BIDS gives calibration errors in pixel, mm or cm: pixels here,
            # degrees in CalibrationQuality.
            eye_meta["CalibrationUnit"] = "pixel"
            errors = [math.hypot((g[0] - p.x) * cal.screen[0],
                                 (g[1] - p.y) * cal.screen[1])
                      for p in cal.points for g in p.gaze.values()]
            if cal.average_error_px is not None:
                eye_meta["AverageCalibrationError"] = round(
                    cal.average_error_px, 2)
            if errors:
                eye_meta["MaximalCalibrationError"] = round(max(errors), 2)
            eye_meta["CalibrationPosition"] = [[round(p.x * cal.screen[0]),
                                                round(p.y * cal.screen[1])]
                                               for p in cal.points]
            eye_meta["CalibrationQuality"] = {
                "Summary": cal.summary(),
                "AccuracyDegrees": cal.accuracy_deg,
                "MaximalErrorDegrees": cal.max_error_deg,
                "ValidPoints": cal.valid_points,
                "Time": datetime.fromtimestamp(cal.created).isoformat(),
            }
        if val is not None:
            eye_meta["ValidationQuality"] = {
                "Summary": val.summary(),
                "AccuracyDegrees": val.accuracy_deg,
                "PrecisionRMSS2SDegrees": val.precision_deg,
                "Time": datetime.fromtimestamp(val.created).isoformat(),
            }
        if screen is not None and screen_size_m(screen)[1]:
            eye_meta["ScreenSizeEstimated"] = (
                "The screen's physical width was not known: ScreenSize "
                "assumes 96 pixels per inch. Set it in the GP3's settings.")
        for key, eye in (("eye1", "left"), ("eye2", "right")):
            meta = dict(eye_meta, RecordedEye=eye)
            side = "L" if eye == "left" else "R"
            if samples:
                meta["ValidSampleFraction"] = round(valid[side] / samples, 4)
            write_json(_json(paths[key]), meta)
        gaze = {**common, "PhysioType": "generic",
                "Columns": ["timestamp", *CHANNELS],
                "timestamp": eye_meta["timestamp"]}
        for name in CHANNELS:
            unit = UNITS[name]
            gaze[name] = {"Description": DESCRIPTIONS[name] +
                          (" (fraction of the screen)"
                           if unit == "normalized" and "camera" not in
                           DESCRIPTIONS[name] else ""),
                          "Units": {"normalized": "arbitrary",
                                    "bool": "arbitrary",
                                    "count": "arbitrary"}.get(unit, unit)}
        write_json(_json(paths["gaze"]), gaze)
        events = {
            "Description": "Fixations (binocular, from the tracker's "
                           "fixation filter) and blinks",
            "TaskName": target.entities.task,
            "OnsetSource": "timestamp",
            "Columns": EVENT_COLUMNS,
            "onset": {"Description": "Start, on the timestamp clock",
                      "Units": "s"},
            "duration": {"Description": "Duration", "Units": "s"},
            "trial_type": {"Description": "Event type",
                           "Levels": {"fixation": "fixation (binocular)",
                                      "blink": "blink"}},
            "blink": {"Description": "1 for blinks",
                      "Levels": {"0": "no", "1": "yes"}},
            "message": {"Description": "Message from the tracker"},
        }
        for key in ("events1", "events2"):
            write_json(_json(paths[key]), events)

    # -- status ------------------------------------------------------------------

    def identity(self) -> str:
        backend = self.backend
        if self.simulated:
            return "GP3 (simulated)"
        model = backend.model if backend else "GP3"
        if self.config.mode == "remote":
            return f"{model} @ {self.config.host}:{self.config.port}"
        serial = backend.info.get("serial") if backend else ""
        return f"{model} {serial}".strip()

    @property
    def calibrated(self) -> Optional[bool]:
        backend = self.backend
        return backend.calibrated if backend else None

    def status(self) -> str:
        name = self.identity()
        if self.state == RUNNING:
            text = "recording" if self.recording else "tracking"
            fps = self.backend.eye_view().fps if self.backend else 0
            return f"{name}: {text}, {fps:.0f} Hz"
        return {IDLE: f"{name}: not connected",
                STARTING: f"{name}: connecting…",
                STOPPING: f"{name}: stopping…"}.get(self.state, self.state)


#: Metres per pixel when a screen's size is unknown (96 pixels per inch).
_M_PER_PX = 0.0254 / 96


def screen_size_m(screen) -> tuple[list[float], bool]:
    """The screen's width and height in metres, and whether it is a
    guess (when its physical width is unknown)."""
    if screen.width_cm:
        w = screen.width_cm / 100
        return [round(w, 4), round(w * screen.height / screen.width, 4)], \
            False
    return [round(screen.width * _M_PER_PX, 4),
            round(screen.height * _M_PER_PX, 4)], True


def _json(path: Path) -> Path:
    return path.with_name(path.name.replace(".tsv.gz", ".json"))
