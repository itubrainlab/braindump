"""What every device brAIn dump records has in common."""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Callable, ClassVar, Optional

from ..bids import Dataset, Entities

IDLE, STARTING, RUNNING, STOPPING = "idle", "starting", "running", "stopping"


@dataclass
class DeviceConfig:
    """A device's configuration, stored in the project as a plain dict."""

    @classmethod
    def from_dict(cls, data: Optional[dict]):
        """Missing keys take their defaults, unknown ones are ignored, so
        projects saved by other versions still open."""
        data = data or {}
        known = {f.name: f for f in fields(cls)}
        values = {}
        for name, value in data.items():
            if name in known:
                values[name] = value
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RecordingTarget:
    """Where and as what a recording is written."""

    dataset: Dataset
    entities: Entities
    #: BIDS datatype folder: "eeg" when EEG is recorded, else "beh".
    datatype: str
    #: When Record was pressed (time.time()).
    started: float
    #: Mains frequency, in Hz (BIDS PowerLineFrequency).
    power_line_frequency: int = 50
    #: Fields every data sidecar gets (InstitutionName and the like).
    common_metadata: dict = field(default_factory=dict)
    #: EEG-only fields about the participant (HeadCircumference).
    eeg_metadata: dict = field(default_factory=dict)


@dataclass
class Event:
    """Something that happened during a recording, for events.tsv."""

    #: time.time() of its onset.
    time: float
    trial_type: str
    value: str = ""
    duration: float = 0.0
    #: Where it came from, e.g. "trigger" or "lsl:<stream name>".
    source: str = ""


@dataclass
class RecordedFiles:
    """What a device wrote for one recording."""

    #: Data files, with the time of their first sample, for scans.tsv.
    scans: list[tuple[Path, datetime]] = field(default_factory=list)
    #: Every file written, sidecars included.
    files: list[Path] = field(default_factory=list)
    #: time.time() of the first sample, if any was written.
    first_sample: Optional[float] = None
    #: Writes the sidecars, given the first sample time of the recording's
    #: reference device (the EEG when there is one).
    finalize: Callable[[Optional[float]], None] = lambda reference: None
    samples: int = 0
    #: Events for the run's events.tsv.
    events: list[Event] = field(default_factory=list)
    #: Fields this device adds to the run's events.json (the eye
    #: tracker's StimulusPresentation, say). Asking for it means the run
    #: gets an events.tsv even without events.
    events_metadata: dict = field(default_factory=dict)


class Device:
    """A recording device: connects, acquires, and records into BIDS.

    :meth:`start`, :meth:`stop` and the recording calls block, so a user
    interface calls them off its main thread. The ``on_*`` callbacks may
    be called from any thread.
    """

    TYPE: ClassVar[str] = ""
    NAME: ClassVar[str] = ""
    MANUFACTURER: ClassVar[str] = ""
    DESCRIPTION: ClassVar[str] = ""
    #: "eeg" or "eyetrack".
    MODALITY: ClassVar[str] = ""
    CONFIG: ClassVar[type] = DeviceConfig

    def __init__(self, config: DeviceConfig, project_folder: Path,
                 simulate: bool = False):
        self.config = config
        self.project_folder = Path(project_folder)
        self.force_simulate = simulate
        self.state = IDLE
        self.error: Optional[str] = None
        self._lock = threading.RLock()
        self.on_state: Callable[[str], None] = lambda state: None
        self.on_error: Callable[[str], None] = lambda message: None

    @property
    def simulated(self) -> bool:
        return self.force_simulate or bool(getattr(self.config, "simulate",
                                                   False))

    @property
    def running(self) -> bool:
        return self.state == RUNNING

    @property
    def recording(self) -> bool:
        return False

    def _set_state(self, state: str) -> None:
        self.state = state
        self.on_state(state)

    def _fail(self, message: str) -> None:
        """The device stopped on its own, on an error."""
        self.error = message

        def work():
            try:
                self.stop()
            except Exception:  # noqa: BLE001
                pass
            self.on_error(message)
        threading.Thread(target=work, daemon=True).start()

    # -- to implement --------------------------------------------------------

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def begin_recording(self, target: RecordingTarget) -> list[Path]:
        """Start writing; returns the data files it writes."""
        raise NotImplementedError

    def end_recording(self) -> RecordedFiles:
        raise NotImplementedError

    def status(self) -> str:
        """One line about the device's state, for the status bar."""
        return self.state

    def identity(self) -> str:
        """The device's serial or address, for the user."""
        return ""
