"""What every device brAIn dump records has in common.

Every device is an input of the session pipeline (:mod:`braindump.pipeline`):
it builds its g.Pype nodes into it (:meth:`Device.build`), and connecting
or disconnecting it restarts that pipeline.
"""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Optional

from ..bids import Dataset, Entities

if TYPE_CHECKING:  # pragma: no cover
    import gpype as gp
    from ..pipeline.session import SessionPipeline

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
    #: When Record was pressed (braindump.clock).
    started: float
    #: Mains frequency, in Hz (BIDS PowerLineFrequency).
    power_line_frequency: int = 50
    #: Fields every data sidecar gets (InstitutionName and the like).
    common_metadata: dict = field(default_factory=dict)
    #: EEG-only fields about the participant (HeadCircumference).
    eeg_metadata: dict = field(default_factory=dict)


@dataclass
class Timeline:
    """The recording's time base: when its reference device's first
    sample was taken, and how fast that device's samples really came.

    BIDS gives StartTime and event onsets in the reference data's own
    time: sample number over its nominal rate. With EEG, that is the
    EEG's 250 Hz, which its amplifier's clock keeps only to some tens of
    ppm, so a clock time maps to EEG time through the rate measured over
    the recording, not the nominal one. Without EEG, it is plain seconds
    from the first sample of the earliest device.
    """

    #: Clock time (braindump.clock) of the reference's first sample.
    t0: float
    #: Samples per second the reference really delivered, by the clock.
    rate: float = 1.0
    #: Samples per second its files say (BIDS SamplingFrequency).
    nominal_rate: float = 1.0

    def seconds(self, t: float) -> float:
        """Clock time ``t`` in the reference's own time, from its first
        sample."""
        return (t - self.t0) * self.rate / self.nominal_rate

    def sample(self, t: float) -> int:
        """The reference's sample at clock time ``t`` (from 0)."""
        return int(round((t - self.t0) * self.rate))

    def clock(self, sample: float) -> float:
        """Clock time of the reference's sample ``sample``."""
        return self.t0 + sample / self.rate


def start_time(reference: Optional[Timeline], first: Optional[float]
               ) -> float:
    """BIDS StartTime of data whose first sample came at ``first``."""
    if reference is None or first is None:
        return 0.0
    return round(reference.seconds(first), 6)


@dataclass
class Event:
    """Something that happened during a recording, for events.tsv."""

    #: Clock time (braindump.clock) of its onset.
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
    #: Clock time (braindump.clock) of the first sample, if any.
    first_sample: Optional[float] = None
    #: Writes the sidecars, given the recording's time base (the EEG's,
    #: when there is EEG).
    finalize: Callable[[Optional["Timeline"]], None] = \
        lambda reference: None
    #: This device's own time base, when it can be the reference (EEG).
    timeline: Optional["Timeline"] = None
    samples: int = 0
    #: Events for the run's events.tsv.
    events: list[Event] = field(default_factory=list)
    #: Fields this device adds to the run's events.json (the eye
    #: tracker's StimulusPresentation, say). Asking for it means the run
    #: gets an events.tsv even without events.
    events_metadata: dict = field(default_factory=dict)


class Device:
    """A recording device: an input of the session pipeline, recorded into
    BIDS.

    A device builds its g.Pype nodes into the session pipeline
    (:meth:`build`): its source, the taps its page draws from, and its
    BIDS writers, whose ``begin`` and ``end`` actions record. Its own
    thread, if it has one (a Bluetooth link, a camera), starts once the
    pipeline runs (:meth:`opened`) and stops before it does
    (:meth:`closing`).

    :meth:`start` (connect), :meth:`stop` (disconnect) and the recording
    calls block, so a user interface calls them off its main thread. The
    ``on_*`` callbacks may be called from any thread.
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
        self._session: Optional["SessionPipeline"] = None
        #: The device's nodes in the running pipeline (its own dataclass);
        #: None while it is not connected.
        self.graph: Any = None
        self.on_state: Callable[[str], None] = lambda state: None
        self.on_error: Callable[[str], None] = lambda message: None

    # -- the session pipeline ----------------------------------------------------

    @property
    def session(self) -> "SessionPipeline":
        """The session pipeline this device connects into: the project's
        (see :meth:`attach`), or one of its own."""
        if self._session is None:
            from ..pipeline.session import SessionPipeline
            self._session = SessionPipeline()
        return self._session

    def attach(self, session: "SessionPipeline") -> None:
        """Connect into ``session`` from now on."""
        self._session = session

    def node_name(self, part: str) -> str:
        """The name of one of this device's g.Pype nodes; the session
        finds the device a failing node belongs to by it."""
        return f"{self.TYPE}_{part}"

    def invoke(self, node, action: str, arg: Optional[dict] = None):
        """Run one of this device's nodes' g.Pype actions."""
        return self.session.invoke(node, action, arg)
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
        """The device stopped on its own, on an error: the session
        pipeline stops (from any thread)."""
        self.error = message
        if self._session is not None and self in self._session.devices:
            self._session.fail(self, message)
        else:
            self.on_error(message)

    def start(self) -> None:
        """Connect: join the session pipeline (which restarts)."""
        self.session.connect([self])

    def stop(self) -> None:
        """Disconnect: leave the session pipeline."""
        if self._session is not None:
            self._session.disconnect([self])

    # -- to implement --------------------------------------------------------

    def build(self, pipeline: "gp.Pipeline") -> None:
        """Add this device's nodes to ``pipeline`` (not yet started), and
        keep them in :attr:`graph`."""
        raise NotImplementedError

    def opened(self) -> None:
        """The pipeline runs: start the device's own thread, if any."""

    def closing(self) -> None:
        """The pipeline is about to stop: stop the device's own thread."""

    def closed(self) -> None:
        """The pipeline has stopped."""
        self.graph = None

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
