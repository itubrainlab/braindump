import os

import pytest


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    """Keep brAIn dump's own settings out of the user's."""
    monkeypatch.setenv("BRAINDUMP_HOME", str(tmp_path / "home"))
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def connect_all(devices):
    """Connect ``devices`` as the app does: into one session pipeline,
    started once with all of them. Returns the pipeline (stop() it)."""
    from braindump.pipeline import SessionPipeline
    pipeline = SessionPipeline()
    for device in devices:
        device.attach(pipeline)
    pipeline.connect(devices)
    return pipeline
