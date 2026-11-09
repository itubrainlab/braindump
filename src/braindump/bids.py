"""Writing a BIDS dataset (https://bids-specification.readthedocs.io).

The layout brAIn dump writes, for one recording of every device::

    <root>/
      dataset_description.json  README  participants.tsv  participants.json
      sub-<label>/
        sub-<label>_sessions.tsv
        ses-<label>/
          sub-<label>_ses-<label>_scans.tsv
          eeg/
            sub-.._ses-.._task-.._run-.._eeg.bdf         Unicorn (BDF)
            sub-.._eeg.json  _channels.tsv  _events.tsv
            sub-.._recording-eye1_physio.tsv.gz  .json   GP3, left eye
            sub-.._recording-eye2_physio.tsv.gz  .json   GP3, right eye
            sub-.._recording-eye1_physioevents.tsv.gz ..
            sub-.._recording-gaze_physio.tsv.gz  .json   GP3, every channel

Eye tracking follows BIDS 1.10 (``physio`` with ``PhysioType``
``eyetrack``), placed with the EEG it was recorded with, or under
``beh/`` when no EEG was recorded. ``scans.tsv`` gives each file's
acquisition time to the microsecond, on this computer's clock, which is
what lines the devices up.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

from . import APP_NAME, __version__

BIDS_VERSION = "1.10.0"
NA = "n/a"

_LABEL = re.compile(r"[^A-Za-z0-9]")


def clean_label(text: str) -> str:
    """A BIDS label: letters and digits only."""
    return _LABEL.sub("", str(text or ""))


def is_label(text: str) -> bool:
    return bool(text) and clean_label(text) == text


def acq_time(moment: datetime) -> str:
    """``YYYY-MM-DDThh:mm:ss.ffffff``, as scans.tsv and sessions.tsv take."""
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%f")


def tsv_value(value) -> str:
    if value is None or value == "":
        return NA
    if isinstance(value, float):
        if value != value:  # NaN
            return NA
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return "0" if text in ("-0", "") else text
    return str(value)


@dataclass(frozen=True)
class Entities:
    """The entities naming one recording's files."""

    sub: str
    task: str
    ses: str = ""
    run: Optional[int] = None
    acq: str = ""

    def __post_init__(self):
        for name in ("sub", "task"):
            if not is_label(getattr(self, name)):
                raise ValueError(f"{name} must be letters and digits only, "
                                 f"got {getattr(self, name)!r}.")
        for name in ("ses", "acq"):
            value = getattr(self, name)
            if value and not is_label(value):
                raise ValueError(f"{name} must be letters and digits only, "
                                 f"got {value!r}.")

    def stem(self, recording: str = "") -> str:
        parts = [f"sub-{self.sub}"]
        if self.ses:
            parts.append(f"ses-{self.ses}")
        parts.append(f"task-{self.task}")
        if self.acq:
            parts.append(f"acq-{self.acq}")
        if self.run is not None:
            parts.append(f"run-{self.run}")
        if recording:
            parts.append(f"recording-{recording}")
        return "_".join(parts)


class Dataset:
    """A BIDS dataset rooted at ``root``."""

    def __init__(self, root: str | Path, name: str = "",
                 authors: Sequence[str] = (), description: str = "",
                 license: str = "", devices: Sequence[str] = ()):
        self.root = Path(root)
        self.name = name or self.root.name
        self.authors = list(authors)
        self.description = description
        self.license = license
        self.devices = list(devices)

    # -- folders -----------------------------------------------------------

    def subject_dir(self, e: Entities) -> Path:
        return self.root / f"sub-{e.sub}"

    def session_dir(self, e: Entities) -> Path:
        base = self.subject_dir(e)
        return base / f"ses-{e.ses}" if e.ses else base

    def datatype_dir(self, e: Entities, datatype: str) -> Path:
        return self.session_dir(e) / datatype

    def path(self, e: Entities, datatype: str, suffix: str, extension: str,
             recording: str = "") -> Path:
        return (self.datatype_dir(e, datatype)
                / f"{e.stem(recording)}_{suffix}{extension}")

    # -- dataset level -----------------------------------------------------

    def ensure(self) -> None:
        """Create the dataset's top-level files if they are missing."""
        self.root.mkdir(parents=True, exist_ok=True)
        description = self.root / "dataset_description.json"
        if not description.exists():
            write_json(description, {
                "Name": self.name,
                "BIDSVersion": BIDS_VERSION,
                "DatasetType": "raw",
                "Authors": self.authors or ["n/a"],
                **({"License": self.license} if self.license else {}),
                "GeneratedBy": [{
                    "Name": APP_NAME,
                    "Version": __version__,
                    "Description": "Multi-device recording for the ITU "
                                   "brAIn lab",
                    "CodeURL": "https://github.com/itubrainlab/braindump",
                }],
            })
        if not any((self.root / name).exists() for name in
                   ("README", "README.md", "README.txt", "README.rst")):
            (self.root / "README.md").write_text(self.readme(),
                                                 encoding="utf-8")
        sidecar = self.root / "participants.json"
        if not sidecar.exists():
            write_json(sidecar, {
                "participant_id": {"Description": "Participant label"},
                "age": {"Description": "Age of the participant",
                        "Units": "year"},
                "sex": {"Description": "Sex of the participant",
                        "Levels": {"M": "male", "F": "female",
                                   "O": "other"}},
                "handedness": {"Description": "Handedness of the participant",
                               "Levels": {"R": "right", "L": "left",
                                          "A": "ambidextrous"}},
            })

    def readme(self) -> str:
        devices = "".join(f"- {d}\n" for d in self.devices) or "- n/a\n"
        return f"""# {self.name}

{self.description or "(Describe the study here.)"}

## Recording

Recorded with {APP_NAME} {__version__}
(https://github.com/itubrainlab/braindump) at the ITU brAIn lab, from:

{devices}
## Layout

- `eeg/*_eeg.bdf`: EEG (g.tec Unicorn Hybrid Black), with `_eeg.json`,
  `_channels.tsv`. Recordings are raw: no software filters.
- `*_recording-eye1_physio.tsv.gz` / `eye2`: eye tracking (Gazepoint GP3),
  left and right eye, as BIDS physio with `PhysioType` `eyetrack`; fixations
  and blinks in `_physioevents.tsv.gz`. `recording-gaze` holds every
  channel the tracker reports.
- `*_events.tsv`: the EEG's triggers and the samples of the recorded Lab
  Streaming Layer streams (`trial_type` is the stream's name).
- Eye tracking without EEG goes under `beh/`.

## Timing

Every device is time-stamped on the recording computer's clock.
`scans.tsv` gives each file's first sample to the microsecond; event onsets
and the eye tracker's `StartTime` count from the EEG's first sample (or,
without EEG, from the earliest first sample).
"""

    def add_participant(self, sub: str, info: Optional[dict] = None) -> None:
        """Add ``sub-<sub>`` to participants.tsv, or fill in its blanks."""
        info = {k: v for k, v in (info or {}).items() if v not in (None, "")}
        update_tsv(self.root / "participants.tsv", "participant_id",
                   f"sub-{sub}", info,
                   default_columns=["participant_id", "age", "sex",
                                    "handedness"])

    def add_session(self, e: Entities, moment: datetime) -> None:
        if not e.ses:
            return
        path = self.subject_dir(e) / f"sub-{e.sub}_sessions.tsv"
        update_tsv(path, "session_id", f"ses-{e.ses}",
                   {"acq_time": acq_time(moment)}, overwrite=False)

    def add_scans(self, e: Entities,
                  scans: Iterable[tuple[Path, datetime]]) -> None:
        """List recorded files, with their acquisition times, in the
        session's scans.tsv."""
        folder = self.session_dir(e)
        stem = f"sub-{e.sub}" + (f"_ses-{e.ses}" if e.ses else "")
        path = folder / f"{stem}_scans.tsv"
        for file, moment in scans:
            name = Path(file).relative_to(folder).as_posix()
            update_tsv(path, "filename", name, {"acq_time": acq_time(moment)})

    # -- runs ----------------------------------------------------------------

    def runs(self, sub: str, ses: str = "", task: str = "") -> set[int]:
        """Runs already recorded for this subject, session and task."""
        if not (is_label(sub) and (not ses or is_label(ses))):
            return set()
        e = Entities(sub=sub, task=task or "x", ses=ses)
        folder = self.session_dir(e)
        if not folder.is_dir():
            return set()
        found = set()
        pattern = re.compile(rf"_task-{re.escape(task)}_" if task else "_")
        for file in folder.rglob("sub-*"):
            if task and not pattern.search(file.name):
                continue
            m = re.search(r"_run-(\d+)", file.name)
            if m:
                found.add(int(m.group(1)))
        return found

    def next_run(self, sub: str, ses: str = "", task: str = "") -> int:
        runs = self.runs(sub, ses, task)
        return max(runs) + 1 if runs else 1


# -- files ---------------------------------------------------------------------

def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def write_tsv(path: Path, header: Sequence[str],
              rows: Iterable[Sequence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(tsv_value(v) for v in row) + "\n")


def read_tsv(path: Path) -> tuple[list[str], list[dict]]:
    if not path.exists():
        return [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def update_tsv(path: Path, key: str, value: str, values: dict,
               default_columns: Sequence[str] = (),
               overwrite: bool = True) -> None:
    """Add or update the row whose ``key`` column is ``value``."""
    columns, rows = read_tsv(path)
    if not columns:
        columns = list(default_columns) or [key]
    if key not in columns:
        columns.insert(0, key)
    for name in values:
        if name not in columns:
            columns.append(name)
    row = next((r for r in rows if r.get(key) == value), None)
    if row is None:
        row = {key: value}
        rows.append(row)
    for name, v in values.items():
        if overwrite or row.get(name) in (None, "", NA):
            row[name] = tsv_value(v)
    write_tsv(path, columns,
              [[r.get(c) or NA for c in columns] for r in rows])


class GzipTsvWriter:
    """Rows into a headerless, gzipped TSV (as BIDS physio files are),
    written as they come."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        # No file name and no time in the gzip header: the BIDS validator
        # asks for neither, and the files then depend on their content only.
        self._raw = open(path, "wb")
        self._gzip = gzip.GzipFile(filename="", mode="wb", fileobj=self._raw,
                                   mtime=0)
        self._file = io.TextIOWrapper(self._gzip, encoding="utf-8",
                                      newline="")
        self.rows = 0

    def write(self, row: Sequence) -> None:
        self._file.write("\t".join(tsv_value(v) for v in row) + "\n")
        self.rows += 1

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._raw.close()
            self._file = None
