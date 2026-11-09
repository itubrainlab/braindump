"""Where brAIn dump keeps its own files (not the projects')."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def config_dir() -> Path:
    """The per-user folder for brAIn dump's settings, created on demand.

    ``BRAINDUMP_HOME`` overrides it (the tests use this).
    """
    override = os.environ.get("BRAINDUMP_HOME")
    if override:
        path = Path(override)
    elif sys.platform == "darwin":
        path = Path.home() / "Library" / "Application Support" / "brAIn dump"
    elif sys.platform == "win32":
        path = Path(os.environ.get("APPDATA", Path.home())) / "brAIn dump"
    else:
        path = Path(os.environ.get("XDG_CONFIG_HOME",
                                   Path.home() / ".config")) / "braindump"
    path.mkdir(parents=True, exist_ok=True)
    return path
