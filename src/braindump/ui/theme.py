"""The look: g.Pype's dark theme, as Rhino Recorder wears it, with the
brAIn lab's red."""
from __future__ import annotations

from pathlib import Path

from rhinorecorder.ui import theme as rhino

APP_ICON = Path(__file__).resolve().parent.parent / "resources" / \
    "braindump.svg"

WINDOW = rhino.WINDOW
BASE = rhino.BASE
ALTERNATE = rhino.ALTERNATE
DARK = rhino.DARK
TEXT = rhino.TEXT
MUTED = rhino.MUTED
HIGHLIGHT = rhino.HIGHLIGHT
ICON = rhino.ICON
ICON_ACTIVE = rhino.ICON_ACTIVE
ICON_DISABLED = rhino.ICON_DISABLED
RECORDING = rhino.RECORDING
WARNING = rhino.WARNING
GOOD = rhino.GOOD
BAD = rhino.BAD
#: The brAIn lab's red.
BRAND = "#BA0022"
LEFT_EYE = "#4fc3f7"
RIGHT_EYE = "#ffb74d"

EXTRA = rhino.EXTRA + f"""
#Brand {{ color: {TEXT}; font-size: 16px; font-weight: 700; }}
#BrandAI {{ color: {BRAND}; font-size: 16px; font-weight: 800; }}
#BrandSub {{ color: {MUTED}; font-size: 16px; }}
#ProjectName {{ color: {TEXT}; font-weight: 600; }}
#DeviceColumn {{ background: {DARK}; border-right: 1px solid rgba(0,0,0,200); }}
#DeviceColumn QLabel#ColumnTitle {{
    color: {MUTED}; font-size: 11px; font-weight: 600;
    letter-spacing: 1px; padding: 10px 12px 4px 12px;
}}
#DeviceCard {{ background: transparent; border-left: 3px solid transparent; }}
#DeviceCard:hover {{ background: rgba(255,255,255,18); }}
#DeviceCard[selected="true"] {{ background: rgba(42,130,218,60);
    border-left: 3px solid {HIGHLIGHT}; }}
#DeviceCard QLabel#CardTitle {{ color: {TEXT}; font-weight: 600; }}
#DeviceCard QLabel#CardSub {{ color: {MUTED}; font-size: 11px; }}
#DeviceCard[available="true"] QLabel#CardTitle {{ color: {MUTED}; }}
#RecordTime {{ color: {RECORDING}; font-weight: 700; font-size: 14px; }}
#QualityBadge {{ border-radius: 4px; padding: 2px 8px; font-weight: 600; }}
#Hint {{ color: {MUTED}; }}
#Big {{ font-size: 18px; font-weight: 600; }}
QPushButton#Danger {{ background: {RECORDING}; border: 1px solid {RECORDING}; }}
"""


def apply(app) -> None:
    """Dress a QApplication as a g.Pype application, in brAIn lab red."""
    from gpype.frontend import identity
    from gpype.frontend import theme as gpype_theme

    gpype_theme.apply(app, gpype_theme.DARK)
    app.setStyleSheet(app.styleSheet() + EXTRA)
    identity.apply(app, "brAIn dump", icon=APP_ICON)
