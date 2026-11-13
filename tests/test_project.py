import json

import pytest

from brainstream import project as proj
from brainstream.devices import DEVICE_TYPES, make_device
from brainstream.devices.gp3 import GP3Config
from brainstream.devices.unicorn import UnicornConfig


def test_create_load_roundtrip(tmp_path):
    p = proj.Project.create(tmp_path / "study", "My Study", task="nback")
    p.add_device("unicorn", UnicornConfig(serial="UN-1").to_dict())
    p.add_device("gp3", GP3Config(mode="remote", host="10.0.0.2").to_dict())
    loaded = proj.Project.load(p.path)
    assert loaded.name == "My Study"
    assert loaded.task == "nback"
    assert list(loaded.devices) == ["unicorn", "gp3"]
    assert loaded.devices["gp3"]["host"] == "10.0.0.2"
    assert loaded.output_path == tmp_path / "study" / "bids"
    with pytest.raises(ValueError):
        loaded.add_device("gp3")
    loaded.remove_device("unicorn")
    assert list(proj.Project.load(p.path).devices) == ["gp3"]


def test_config_tolerates_unknown_and_missing_keys(tmp_path):
    cfg = GP3Config.from_dict({"mode": "remote", "from_the_future": 1})
    assert cfg.mode == "remote" and cfg.port == 4242
    device = make_device("unicorn", {"electrodes": ["A", "B"]}, tmp_path)
    assert device.config.electrode_names()[:3] == ["A", "B", "EEG 3"]


def test_absolute_output_folder(tmp_path):
    p = proj.Project.create(tmp_path / "a", "A",
                            output_folder=str(tmp_path / "out"))
    assert p.output_path == tmp_path / "out"


def test_recent_projects(tmp_path):
    a = proj.Project.create(tmp_path / "a", "A")
    b = proj.Project.create(tmp_path / "b", "B")
    proj.remember(a.path)
    proj.remember(b.path)
    proj.remember(a.path)
    assert proj.recent_projects() == [a.path.resolve(), b.path.resolve()]
    b.path.unlink()
    assert proj.recent_projects() == [a.path.resolve()]


def test_registry():
    assert set(DEVICE_TYPES) == {"unicorn", "hiamp", "gp3", "lsl", "polar", "shimmer", "webcam", "obs"}
    # Shown as Screen Capture, still saved as "obs" (issue #3).
    assert DEVICE_TYPES["obs"].NAME == "Screen Capture"
    assert "OBS" in DEVICE_TYPES["obs"].DESCRIPTION
    # Shown as Video Stream, still saved as "webcam" (issue #4).
    assert DEVICE_TYPES["webcam"].NAME == "Video Stream"
    saved = json.loads(json.dumps(UnicornConfig().to_dict()))
    assert UnicornConfig.from_dict(saved) == UnicornConfig()


def test_settings_from_before_the_rename_are_kept(tmp_path, monkeypatch):
    """brAIn stream was brAIn dump: its settings folder is moved to the new
    name the first time, and BRAINDUMP_HOME still works."""
    from brainstream import paths
    monkeypatch.delenv("BRAINSTREAM_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    old = paths._system_dir("brAIn dump", "braindump")
    old.mkdir(parents=True)
    (old / "recent.json").write_text("[]")
    new = paths.config_dir()
    assert new == paths._system_dir("brAIn stream", "brainstream")
    assert (new / "recent.json").read_text() == "[]" and not old.exists()
    monkeypatch.setenv("BRAINDUMP_HOME", str(tmp_path / "custom"))
    assert paths.config_dir() == tmp_path / "custom"


def test_projects_from_before_the_rename_open(tmp_path):
    project = proj.Project.create(tmp_path / "p", "Old")
    old = project.path.with_suffix(".braindump")
    project.path.rename(old)
    loaded = proj.Project.load(old)
    assert loaded.name == "Old" and loaded.path == old
    loaded.save()
    assert old.exists()
