"""Render the app icon (SVG) into a macOS .icns file.

    python make_icns.py <icon.svg> <out.icns>
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtGui import QGuiApplication, QIcon  # noqa: E402

svg, out = Path(sys.argv[1]), Path(sys.argv[2])
app = QGuiApplication([])
icon = QIcon(str(svg))
with tempfile.TemporaryDirectory() as tmp:
    iconset = Path(tmp) / "icon.iconset"
    iconset.mkdir()
    for size in (16, 32, 128, 256, 512):
        icon.pixmap(size, size).save(str(iconset / f"icon_{size}x{size}.png"))
        icon.pixmap(2 * size, 2 * size).save(
            str(iconset / f"icon_{size}x{size}@2x.png"))
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out)],
                   check=True)
print(f"wrote {out}")
