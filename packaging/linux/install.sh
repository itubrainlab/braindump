#!/bin/sh
# Install brAIn dump for the current user: copies this folder to
# ~/.local/share/braindump/app, adds a menu entry with its icon and a
# braindump command in ~/.local/bin, then sets it up (downloads g.Pype and
# the libraries from PyPI). No administrator rights needed.
set -eu

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
APP="$DATA/braindump/app"
BIN="$HOME/.local/bin"

case "$(uname -m)" in
    x86_64|amd64) ;;
    *) echo "brAIn dump needs an x86_64 PC (g.Pype has no build for $(uname -m))." >&2
       exit 1 ;;
esac
glibc="$(ldd --version 2>/dev/null | head -1 | grep -o '[0-9]\+\.[0-9]\+$' || true)"
if [ -n "$glibc" ] && [ "$(printf '%s\n2.34\n' "$glibc" | sort -V | head -1)" != "2.34" ]; then
    echo "brAIn dump needs glibc 2.34 or newer (this system has $glibc)," >&2
    echo "e.g. Ubuntu 22.04, Debian 12, Fedora 35 or later." >&2
    exit 1
fi

if [ "$HERE" != "$APP" ]; then
    echo "Copying brAIn dump to $APP"
    rm -rf "$APP"
    mkdir -p "$APP"
    cp -a "$HERE/." "$APP/"
fi

mkdir -p "$BIN" "$DATA/applications" \
    "$DATA/icons/hicolor/scalable/apps" "$DATA/icons/hicolor/256x256/apps"
ln -sf "$APP/braindump" "$BIN/braindump"
cp "$APP/braindump.svg" "$DATA/icons/hicolor/scalable/apps/braindump.svg"
cp "$APP/braindump.png" "$DATA/icons/hicolor/256x256/apps/braindump.png"
cat > "$DATA/applications/braindump.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=brAIn dump
GenericName=Multi-device Recorder
Comment=Record EEG, eye tracking, ECG, GSR and LSL streams into BIDS
Exec="$APP/braindump" %U
Icon=braindump
Terminal=false
Categories=Science;Education;MedicalSoftware;
StartupWMClass=braindump
DESKTOP
update-desktop-database "$DATA/applications" >/dev/null 2>&1 || true
gtk-update-icon-cache -q "$DATA/icons/hicolor" >/dev/null 2>&1 || true

echo "Setting up (downloads g.Pype and the libraries; a few minutes)..."
build="$(cat "$APP/build-id")"
"$APP/python/bin/python3" -I "$APP/bootstrap.py" "$APP" \
    "$DATA/braindump/runtime/$build"

echo
echo "brAIn dump is installed: open it from your applications menu, or"
echo "run braindump (add ~/.local/bin to PATH if the shell cannot find it)."
