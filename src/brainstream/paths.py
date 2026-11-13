"""Where brAIn stream keeps its own files (not the projects')."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _system_dir(name: str, unix_name: str) -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / name
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home())) / name
    return Path(os.environ.get("XDG_CONFIG_HOME",
                               Path.home() / ".config")) / unix_name


def config_dir() -> Path:
    """The per-user folder for brAIn stream's settings, created on demand.

    ``BRAINSTREAM_HOME`` overrides it (the tests use this), as
    ``BRAINDUMP_HOME`` did. The settings brAIn stream kept under its old
    name, brAIn dump, are moved to the new folder the first time.
    """
    override = os.environ.get("BRAINSTREAM_HOME") or \
        os.environ.get("BRAINDUMP_HOME")
    if override:
        path = Path(override)
    else:
        path = _system_dir("brAIn stream", "brainstream")
        old = _system_dir("brAIn dump", "braindump")
        if not path.exists() and old.is_dir():
            try:
                old.rename(path)
            except OSError:
                pass
    path.mkdir(parents=True, exist_ok=True)
    return path
