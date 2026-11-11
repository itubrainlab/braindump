"""The Video Stream's camera choice: each camera's OpenCV number, the
choice kept by the camera's identifier, and the page showing it."""
import pytest
from PySide6.QtWidgets import QApplication

from braindump.devices import make_device, video
from braindump.devices.video import Camera, opencv_indices
from braindump.project import Project

#: Qt's camera list on a MacBook, in the system's order: the identifiers
#: are AVFoundation's unique IDs.
MAC = [("MacBook Pro Camera", "6C707041-05AC-0011-0007-000000000001"),
       ("HP 4K Streaming Webcam", "0x212200003f00760"),
       ("OBS Virtual Camera", "7626645E-4425-469E-9D8B-97E0FA59AC75"),
       ("Neko Camera", "4E166378-3B94-4DA3-B382-87C500000001")]


def test_opencv_numbers_cameras_by_unique_id_on_macos():
    # OpenCV sorts by uniqueID: HP, Neko, MacBook, OBS. Qt's order is not
    # that one, which is how the wrong camera was opened.
    assert opencv_indices([i for _, i in MAC], "darwin") == [2, 0, 3, 1]


def test_opencv_numbers_on_linux_and_windows():
    assert opencv_indices(["/dev/video2", "/dev/video0", "usb-xyz"],
                          "linux") == [2, 0, None]
    assert opencv_indices(["a", "b", "c"], "win32") == [0, 1, 2]


def _cameras(names_ids, platform="darwin"):
    indices = opencv_indices([i for _, i in names_ids], platform)
    return [Camera(index, name, ident)
            for (name, ident), index in zip(names_ids, indices)]


def test_the_chosen_camera_is_found_again_by_its_id(monkeypatch):
    monkeypatch.setattr(video, "find_cameras", lambda: _cameras(MAC))
    hp = MAC[1][1]
    assert video.resolve_camera(5, hp, "HP 4K Streaming Webcam") == 0
    # Another camera plugged in moves the HP's number; its id finds it.
    more = MAC + [("A capture card", "0x000000000000001")]
    monkeypatch.setattr(video, "find_cameras", lambda: _cameras(more))
    assert video.resolve_camera(0, hp) == 1
    # Chosen by number only (a project from before): the number.
    assert video.resolve_camera(3, "") == 3
    # Not connected: refused, not another camera opened instead.
    monkeypatch.setattr(video, "find_cameras", lambda: _cameras(MAC[:1]))
    with pytest.raises(ConnectionError, match="HP 4K Streaming Webcam"):
        video.resolve_camera(0, hp, "HP 4K Streaming Webcam")


def test_the_page_shows_and_stores_the_chosen_camera(tmp_path, monkeypatch):
    from braindump.ui import video_page
    from braindump.ui.video_page import VideoPage
    monkeypatch.setattr(video_page, "find_cameras", lambda: _cameras(MAC))
    app = QApplication.instance() or QApplication([])
    project = Project.create(tmp_path / "p", "P")
    project.add_device("webcam", {"camera": 0,
                                  "camera_id": MAC[3][1],
                                  "camera_name": "Neko Camera"})
    device = make_device("webcam", project.devices["webcam"],
                         project.folder)
    page = VideoPage(device)
    page._found(_cameras(MAC))
    app.processEvents()
    combo = page.choices
    # The list in the system's order, the chosen camera selected by id.
    assert [combo.itemText(i) for i in range(combo.count())] == \
        [name for name, _ in MAC]
    assert combo.currentText() == "Neko Camera"
    # Choosing the HP stores its id, name and OpenCV number together.
    combo.setCurrentIndex(1)
    combo.activated.emit(1)
    cfg = device.config
    assert (cfg.camera_id, cfg.camera_name, cfg.camera) == \
        (MAC[1][1], "HP 4K Streaming Webcam", 0)
    # A camera that is gone stays named, marked as not connected.
    page._found(_cameras(MAC[:1]))
    assert combo.currentText() == "HP 4K Streaming Webcam (not connected)"
