import json

import pytest

from braindump import project as proj
from braindump.devices import DEVICE_TYPES, make_device
from braindump.devices.gp3 import GP3Config
from braindump.devices.unicorn import UnicornConfig


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
    saved = json.loads(json.dumps(UnicornConfig().to_dict()))
    assert UnicornConfig.from_dict(saved) == UnicornConfig()
