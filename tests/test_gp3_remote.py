"""The remote GP3 backend against a small Open Gaze API server."""
import socket
import threading
import time

from opengp3.server import element, parse_command

from braindump.devices.gp3 import GP3Config
from braindump.devices.gp3_backends import INDEX, RemoteBackend


class FakeServer:
    """Answers every command, streams REC records and runs a 5-point
    calibration the way Gazepoint Control does."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self.sending = False
        self.commands = []
        self.t0 = time.time()
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        conn, _ = self.sock.accept()
        self.conn = conn
        threading.Thread(target=self._stream, daemon=True).start()
        buf = ""
        while True:
            try:
                data = conn.recv(4096)
            except OSError:
                return
            if not data:
                return
            buf += data.decode()
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                cmd = parse_command(line)
                if cmd:
                    self._handle(*cmd)

    def _send(self, line):
        try:
            self.conn.sendall(line.encode())
        except OSError:
            pass

    def _handle(self, verb, ident, a):
        self.commands.append((verb, ident, a))
        extra = {"SCREEN_SIZE": [("WIDTH", 1280), ("HEIGHT", 1024)],
                 "CAMERA_SIZE": [("WIDTH", 640), ("HEIGHT", 480)],
                 "API_ID": [("IMPL", "Fake 1.0"), ("VALUE", "2.0")],
                 "CALIBRATE_DELAY": [("VALUE", "0.05")],
                 "CALIBRATE_TIMEOUT": [("VALUE", "0.05")],
                 "CALIBRATE_RESULT_SUMMARY": [("AVE_ERROR", "21.5"),
                                              ("VALID_POINTS", 5)],
                 }.get(ident, [])
        if ident == "ENABLE_SEND_DATA":
            self.sending = a.get("STATE") == "1"
        self._send(element("ACK", [("ID", ident), *extra]))
        if ident == "CALIBRATE_START" and a.get("STATE") == "1":
            threading.Thread(target=self._calibrate, daemon=True).start()

    def _calibrate(self):
        pts = [(0.5, 0.5), (0.85, 0.15), (0.85, 0.85), (0.15, 0.85),
               (0.15, 0.15)]
        attrs = [("ID", "CALIB_RESULT")]
        for k, (x, y) in enumerate(pts, 1):
            self._send(element("CAL", [("ID", "CALIB_START_PT"), ("PT", k),
                                       ("CALX", x), ("CALY", y)]))
            time.sleep(0.05)
            self._send(element("CAL", [("ID", "CALIB_RESULT_PT"), ("PT", k),
                                       ("CALX", x), ("CALY", y)]))
            attrs += [(f"CALX{k}", x), (f"CALY{k}", y),
                      (f"LX{k}", x + 0.01), (f"LY{k}", y), (f"LV{k}", 1),
                      (f"RX{k}", x), (f"RY{k}", y - 0.01), (f"RV{k}", 1)]
        self._send(element("CAL", attrs))

    def _stream(self):
        n = 0
        while True:
            time.sleep(1 / 60)
            if not self.sending:
                continue
            n += 1
            t = time.time() - self.t0
            self._send(element("REC", [
                ("CNT", n), ("TIME", f"{t:.5f}"), ("BPOGX", 0.25),
                ("BPOGY", 0.75), ("BPOGV", 1), ("LPCX", 0.6), ("LPCY", 0.5),
                ("LPD", 20), ("LPV", 1), ("LEYEZ", 0.6), ("LPUPILV", 1),
                ("LPMM", 3.5), ("LPMMV", 1)]))


def test_remote_backend_streams_and_calibrates():
    server = FakeServer()
    cfg = GP3Config(mode="remote", port=server.port,
                    remote_calibration_display="here")
    backend = RemoteBackend(cfg)
    rows = []
    backend.on_row = rows.append
    backend.start()
    try:
        time.sleep(0.5)
        assert len(rows) > 15
        row = rows[-1]
        assert abs(row[0] - time.time()) < 0.5
        assert row[1 + INDEX["BPOGX"]] == 0.25
        assert row[1 + INDEX["EYEZ"]] == 0.6
        assert backend.screen.width == 1280
        assert backend.info["software"] == "Fake 1.0"
        view = backend.eye_view()
        assert set(view.eyes) == {"L"} and view.eyes["L"][2] == 20 / 640

        run = backend.calibrate(5)
        assert run.shows_targets
        seen = set()
        deadline = time.time() + 3
        while not run.tick() and time.time() < deadline:
            target = run.target()
            if target:
                seen.add(target[:2])
            time.sleep(0.005)
        quality = run.result()
        assert quality.valid_points == 5 and quality.n_points == 5
        assert quality.average_error_px == 21.5
        assert quality.points[0].gaze["L"] == (0.51, 0.5)
        assert (0.85, 0.15) in seen
        sent = [(v, i) for v, i, _ in server.commands]
        assert ("SET", "CALIBRATE_RESET") in sent
        assert ("SET", "CALIBRATE_START") in sent
    finally:
        backend.stop()
