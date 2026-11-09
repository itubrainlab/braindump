import os

import pytest


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    """Keep brAIn dump's own settings out of the user's."""
    monkeypatch.setenv("BRAINDUMP_HOME", str(tmp_path / "home"))
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
