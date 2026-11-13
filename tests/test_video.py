"""Video inputs, stored as BIDS BEP047 behavioural video with every
frame's time."""
import gzip
import json
import threading
import time

import cv2
import numpy as np
import pytest

from conftest import connect_all
from brainstream.bids import read_tsv
from brainstream.devices import make_device
from brainstream.pipeline import DeviceError
from brainstream.project import Project
from brainstream.recording import Session, Subject


def _record(project, devices, seconds=2.5, task="watch", warmup=1.0):
    pipeline = connect_all(devices)
    try:
        time.sleep(warmup)
        session = Session(project, devices)
        session.start(Subject(sub="01", task=task))
        time.sleep(seconds)
        return session.stop()
    finally:
        pipeline.stop()


@pytest.mark.slow
def test_simulated_webcam_with_eeg(tmp_path):
    project = Project.create(tmp_path / "p", "P")
    project.add_device("unicorn", {"simulate": True})
    project.add_device("webcam", {"simulate": True, "position": "front"})
    devices = [make_device(t, c, project.folder)
               for t, c in project.devices.items()]
    result = _record(project, devices)
    assert not result.problems
    beh = project.output_path / "sub-01" / "beh"
    video = beh / "sub-01_task-watch_recording-webcam_video.mp4"
    capture = cv2.VideoCapture(str(video))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    meta = json.loads((beh / "sub-01_task-watch_recording-webcam_video.json")
                      .read_text())
    assert meta["VideoFrameCount"] == frames and 60 <= frames <= 90
    assert meta["VideoCodec"] == "mpeg4" and meta["ImageWidth"] == 640
    assert meta["DevicePosition"] == "front"
    with gzip.open(beh / "sub-01_task-watch_recording-webcam_physio.tsv.gz",
                   "rt") as f:
        rows = np.array([[float(v) for v in line.split("\t")]
                         for line in f.read().splitlines()])
    assert len(rows) == frames and rows[:, 1].tolist() == list(range(frames))
    assert abs(np.median(np.diff(rows[:, 0])) - 1 / 30) < 0.003
    physio = json.loads((beh / "sub-01_task-watch_recording-webcam_physio"
                                ".json").read_text())
    assert physio["Columns"] == ["timestamp", "frame"]
    assert abs(physio["StartTime"]) < 0.2
    # scans.tsv aligns the video (BEP047), events go beside it.
    _, scans = read_tsv(project.output_path / "sub-01" / "sub-01_scans.tsv")
    assert "beh/sub-01_task-watch_recording-webcam_video.mp4" in \
        [r["filename"] for r in scans]
    assert (beh / "sub-01_task-watch_events.tsv").exists()
    ignore = (project.output_path / ".bidsignore").read_text()
    assert "*_video.mp4" in ignore and "*_video.json" in ignore


@pytest.mark.slow
def test_obs_over_ndi(tmp_path):
    """A real NDI stream, sent in this process as OBS would send it."""
    ndi = pytest.importorskip("NDIlib")
    if not ndi.initialize():
        pytest.skip("NDI does not start here")
    create = ndi.SendCreate()
    create.ndi_name = "OBS brAIn stream test"
    sender = ndi.send_create(create)
    image = np.zeros((180, 320, 4), np.uint8)
    frame = ndi.VideoFrameV2()
    frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX
    frame.frame_rate_N, frame.frame_rate_D = 25, 1
    running = True

    def pump():
        n = 0
        while running:
            image[:] = n % 255
            frame.data = image
            ndi.send_send_video_v2(sender, frame)
            n += 1
            time.sleep(1 / 25)
    threading.Thread(target=pump, daemon=True).start()
    project = Project.create(tmp_path / "p", "P")
    project.add_device("obs", {"source": "OBS brAIn stream test"})
    obs = make_device("obs", project.devices["obs"], project.folder)
    try:
        try:
            obs.start()
        except DeviceError as error:
            if not isinstance(error.__cause__, ConnectionError):
                raise
            pytest.skip(f"NDI discovery does not work here: {error}")
        obs.stop()
        # Recording starts as the stream connects, before its first frame
        # has said its rate: the file must still get the stream's 25 fps.
        # It runs until frames have come, however long NDI takes to
        # connect (seconds, on some CI runners).
        obs.start()
        try:
            session = Session(project, [obs])
            session.start(Subject(sub="01", task="game"))
            deadline = time.time() + 20
            while time.time() < deadline and \
                    getattr(obs.graph.writer, "count", 0) < 15:
                time.sleep(0.1)
            result = session.stop()
        finally:
            obs.stop()
    finally:
        running = False
        ndi.send_destroy(sender)
    assert not result.problems
    beh = project.output_path / "sub-01" / "beh"
    meta = json.loads((beh / "sub-01_task-game_recording-screen_video.json")
                      .read_text())
    assert (meta["ImageWidth"], meta["ImageHeight"]) == (320, 180)
    assert meta["VideoFrameCount"] >= 15
    assert abs(meta["VideoFrameRate"] - 25) < 0.01
    with gzip.open(beh / "sub-01_task-game_recording-screen_physio.tsv.gz",
                   "rt") as f:
        times = [float(line.split("\t")[0]) for line in f]
    assert len(times) == meta["VideoFrameCount"]
    assert all(b > a for a, b in zip(times, times[1:]))
