"""The look: g.Pype's dark theme, with the brAIn lab's red."""
from __future__ import annotations

from pathlib import Path

APP_ICON = Path(__file__).resolve().parent.parent / "resources" / \
    "brainstream.svg"

# The colours of g.Pype's dark palette (gpype.frontend.theme), named.
WINDOW = "#353535"
BASE = "#2a2a2a"
ALTERNATE = "#424242"
DARK = "#232323"
TEXT = "#ffffff"
MUTED = "#7f7f7f"
HIGHLIGHT = "#2a82da"

ICON = TEXT
ICON_ACTIVE = HIGHLIGHT
ICON_DISABLED = MUTED
RECORDING = "#e53935"
WARNING = "#ff9800"
CURSOR = HIGHLIGHT
GOOD = "#2ecc40"
BAD = "#ff4b2b"
UNKNOWN = MUTED
#: The brAIn lab's red.
BRAND = "#BA0022"
LEFT_EYE = "#4fc3f7"
RIGHT_EYE = "#ffb74d"

EXTRA = f"""
#Toolbar {{
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(25,25,25,255), stop:1 rgba(45,45,45,255));
    border-bottom: 1px solid rgba(0,0,0,200);
}}
#Toolbar QToolButton {{ border: none; padding: 4px; }}
#Toolbar QToolButton:hover {{ background: rgba(255,255,255,30); }}
#Toolbar QToolButton:checked {{ background: {HIGHLIGHT}; }}
#Toolbar QLabel {{ color: {MUTED}; }}
#Toolbar QLabel#Brand {{ color: {TEXT}; font-size: 15px; font-weight: 600; }}
#BrandSub {{ color: {MUTED}; font-size: 15px; }}
#Divider {{ background: rgba(255,255,255,40); }}
QPushButton#Primary {{
    background: {HIGHLIGHT}; border: 1px solid {HIGHLIGHT};
}}
QPushButton#Primary:hover {{ background: #0078cc; border-color: #0078cc; }}
QPushButton#Primary:disabled {{ background: {ALTERNATE};
    border-color: {ALTERNATE}; }}
QLabel#Note {{ color: {MUTED}; }}
#SourceToggle QToolButton {{
    padding: 2px 10px; color: {MUTED}; background: {BASE};
    border: 1px solid {ALTERNATE};
}}
#SourceToggle QToolButton:checked {{
    color: {TEXT}; background: {HIGHLIGHT}; border-color: {HIGHLIGHT};
}}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; }}

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
    identity.apply(app, "brAIn stream", icon=APP_ICON)
