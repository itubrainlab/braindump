"""Running with Bluetooth access on macOS.

macOS ends a process that uses Bluetooth unless the app it belongs to
declares why (``NSBluetoothAlwaysUsageDescription``). Started from a
terminal, brAIn stream belongs to the terminal app, so it relaunches itself
inside a small app bundle that declares it; output still comes to the
terminal. The GP3 (USB) and the Open Gaze API (network) need none of this.
"""
from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from pathlib import Path

BUNDLED_ENV = "BRAINSTREAM_BUNDLED"
BUNDLE = (Path.home() / "Library" / "Caches" / "brAIn stream"
          / "brAIn stream.app")

_INFO = {
    "CFBundleName": "brAIn stream",
    "CFBundleDisplayName": "brAIn stream",
    "CFBundleIdentifier": "dk.itu.brainlab.brainstream.terminal",
    "CFBundleExecutable": "brainstream",
    "CFBundlePackageType": "APPL",
    "LSUIElement": False,
    "NSHighResolutionCapable": True,
    "NSCameraUsageDescription":
        "brAIn stream records video from the webcam.",
    "NSBluetoothAlwaysUsageDescription":
        "brAIn stream reads the Unicorn EEG headset over Bluetooth.",
    "NSLocalNetworkUsageDescription":
        "brAIn stream reads eye trackers over the Open Gaze API and receives "
        "triggers over UDP.",
}


def needs_relaunch() -> bool:
    return sys.platform == "darwin" and not os.environ.get(BUNDLED_ENV)


def _write_bundle(python: str) -> None:
    executable = BUNDLE / "Contents" / "MacOS" / "brainstream"
    script = ("#!/bin/sh\n"
              f"export {BUNDLED_ENV}=1\n"
              f'exec "{python}" -m brainstream "$@"\n')
    info = plistlib.dumps(_INFO)
    plist = BUNDLE / "Contents" / "Info.plist"
    if (executable.exists() and executable.read_text() == script
            and plist.exists() and plist.read_bytes() == info):
        return
    executable.parent.mkdir(parents=True, exist_ok=True)
    plist.write_bytes(info)
    executable.write_text(script)
    executable.chmod(0o755)
    subprocess.run(["codesign", "--force", "--sign", "-", str(BUNDLE)],
                   capture_output=True, check=False)


def relaunch_with_bluetooth(argv: list[str]) -> int:
    _write_bundle(sys.executable)
    command = ["open", "-W", "-n", str(BUNDLE)]
    for stream, name in ((sys.stdout, "--stdout"), (sys.stderr, "--stderr")):
        try:
            command += [name, os.ttyname(stream.fileno())]
        except (OSError, ValueError, AttributeError):
            pass
    if argv:
        command += ["--args", *argv]
    return subprocess.call(command)
