"""Projects: which devices a study records, how each is configured, and
where its BIDS dataset goes.

A project is one JSON file, ``<name>.braindump``, in a folder of its own.
The folder also holds the GP3's saved calibrations, and by default the
BIDS dataset (``bids/``). Every change is saved at once, so the devices'
configurations survive between runs.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional

from . import __version__
from .paths import config_dir

EXTENSION = ".braindump"
#: Recent projects remembered for the start dialog.
RECENT_LIMIT = 12


@dataclass
class Project:
    """A recording project. ``devices`` maps a device type (see
    :mod:`braindump.devices`) to its configuration, as a plain dict, in
    the order the devices were added."""

    path: Path
    name: str = "Untitled"
    #: The BIDS dataset's root; relative paths are relative to the
    #: project's folder.
    output_folder: str = "bids"
    #: Task label offered when a recording starts.
    task: str = "rest"
    authors: list[str] = field(default_factory=list)
    description: str = ""
    #: Mains frequency where the lab is, in Hz (BIDS PowerLineFrequency).
    power_line_frequency: int = 50
    #: Licence of the dataset (BIDS License), e.g. CC0 or CC-BY-4.0.
    license: str = "CC0"
    institution_name: str = "IT University of Copenhagen"
    institution_address: str = "Rued Langgaards Vej 7, 2300 Copenhagen S, " \
        "Denmark"
    institution_department: str = "brAIn lab"
    devices: dict[str, dict] = field(default_factory=dict)
    #: Task label -> {"TaskDescription": .., "Instructions": ..}, as
    #: last entered when recording it.
    tasks: dict[str, dict] = field(default_factory=dict)

    # -- files -----------------------------------------------------------

    @property
    def folder(self) -> Path:
        return self.path.parent

    @property
    def output_path(self) -> Path:
        out = Path(os.path.expanduser(self.output_folder))
        return out if out.is_absolute() else (self.folder / out)

    @property
    def calibration_folder(self) -> Path:
        return self.folder / "calibrations"

    @classmethod
    def create(cls, folder: str | Path, name: str,
               output_folder: str = "bids", **kwargs) -> "Project":
        """A new project in ``folder`` (created if needed), saved."""
        folder = Path(folder).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^\w\- ]+", "", name).strip() or "project"
        path = folder / f"{stem}{EXTENSION}"
        if path.exists():
            raise FileExistsError(f"{path} already exists.")
        project = cls(path=path, name=name, output_folder=output_folder,
                      **kwargs)
        project.save()
        return project

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        path = Path(path).expanduser()
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)} - {"path"}
        values = {k: v for k, v in data.items() if k in known}
        devices = values.get("devices") or {}
        values["devices"] = {str(k): dict(v or {}) for k, v in devices.items()}
        return cls(path=path, **values)

    def save(self) -> None:
        data = asdict(self)
        data.pop("path")
        data["format"] = 1
        data["saved_by"] = f"brAIn dump {__version__}"
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, self.path)

    # -- devices ---------------------------------------------------------

    def add_device(self, device_type: str, config: Optional[dict] = None
                   ) -> None:
        if device_type in self.devices:
            raise ValueError(f"The project already has a {device_type}.")
        self.devices[device_type] = dict(config or {})
        self.save()

    def remove_device(self, device_type: str) -> None:
        self.devices.pop(device_type, None)
        self.save()

    def set_device_config(self, device_type: str, config: dict) -> None:
        self.devices[device_type] = dict(config)
        self.save()


# -- recent projects ----------------------------------------------------------

def _recent_file() -> Path:
    return config_dir() / "recent.json"


def recent_projects() -> list[Path]:
    """Recently opened projects that still exist, newest first."""
    try:
        paths = json.loads(_recent_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [Path(p) for p in paths if Path(p).is_file()]


def remember(path: str | Path) -> None:
    path = str(Path(path).resolve())
    paths = [str(p) for p in recent_projects() if str(p) != path]
    paths.insert(0, path)
    try:
        _recent_file().write_text(json.dumps(paths[:RECENT_LIMIT], indent=2),
                                  encoding="utf-8")
    except OSError:
        pass


def forget(path: str | Path) -> None:
    path = str(Path(path).resolve())
    paths = [str(p) for p in recent_projects() if str(p) != path]
    try:
        _recent_file().write_text(json.dumps(paths, indent=2),
                                  encoding="utf-8")
    except OSError:
        pass
