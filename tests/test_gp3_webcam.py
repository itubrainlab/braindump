"""OpenGP3's webcam eye tracking on this computer: chosen, never a
fallback; its warning kept; its calibrations apart from the GP3's."""
import time

import pytest
from opengp3 import camera as gp3_camera
from opengp3 import webcam as gp3_webcam
from opengp3.simulate import SimulatedCamera
from opengp3.webcam import WEBCAM_WARNING

from braindump.devices.gp3 import GP3Config, GP3Device
from braindump.devices.gp3_backends import (LocalBackend,
                                           calibration_is_webcam)


class FakeWebcam(SimulatedCamera):
    """OpenGP3's simulated camera, passing for its webcam."""

    model = "WEBCAM"
    warning = WEBCAM_WARNING
    simulated = False
    webcam = True
    rate = 30.0

    def __init__(self, index=None):
        super().__init__()
        self.index = index
        self.closed = False

    def close(self):
        self.closed = True
        super().close()


def test_the_gp3_is_never_replaced_by_a_webcam(tmp_path, monkeypatch):
    asked = {}

    def open_camera(**kwargs):
        asked.update(kwargs)
        raise gp3_camera.GP3NotFoundError("no GP3")
    monkeypatch.setattr(gp3_camera, "open_camera", open_camera)
    backend = LocalBackend(GP3Config(), tmp_path)
    with pytest.raises(gp3_camera.GP3NotFoundError):
        backend.start()
    assert asked == {"webcam_fallback": False}


def test_a_chosen_webcam_tracks_and_says_it_is_one(tmp_path, monkeypatch):
    opened = []

    def open_webcam(index=None):
        opened.append(FakeWebcam(index))
        return opened[-1]
    monkeypatch.setattr(gp3_webcam, "open_webcam", open_webcam)
    backend = LocalBackend(GP3Config(tracker="webcam"), tmp_path / "cal")
    rows = []
    backend.on_row = rows.append
    backend.start()
    try:
        assert (backend.model, backend.rate) == ("Webcam", 30.0)
        assert backend.warning == WEBCAM_WARNING
        deadline = time.time() + 5
        while not rows and time.time() < deadline:
            time.sleep(0.05)
        assert rows
    finally:
        backend.stop()
    # Automatic choice, and the camera closed by whoever opened it.
    assert opened[0].index is None and opened[0].closed
    path = backend.save_calibration(_Saved())
    assert calibration_is_webcam(path)
    assert not calibration_is_webcam(tmp_path / "cal" / "gp3_1.json")


def test_the_pipeline_is_laid_out_for_the_chosen_tracker(tmp_path):
    import gpype as gp
    cfg = GP3Config(tracker="webcam", webcam_rate=25.0)
    device = GP3Device(cfg, tmp_path)
    device.build(gp.Pipeline())
    assert device.graph.source.config["sampling_rate"] == 25.0
    # Back to the GP3: not at the webcam's rate.
    cfg.tracker, cfg.model = "gazepoint", "Webcam"
    device.build(gp.Pipeline())
    assert device.graph.source.config["sampling_rate"] == 60.0


class _Saved:
    def save(self, path):
        open(path, "w").write("{}")
