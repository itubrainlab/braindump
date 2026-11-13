#!/bin/bash
# Build "brAIn stream.app" and a DMG for macOS (Apple Silicon).
#
#     packaging/macos/build_app.sh
#
# Needs: macOS (codesign, hdiutil, iconutil), uv, git, and the project's
# .venv (pip install -e ".[test,shimmer]"), which provides the tested
# library versions and PySide6 for the icon.
#
# The app carries its own Python (a relocatable python-build-standalone
# build, as uv installs them) and the wheels of brAIn stream
# and OpenGP3; see ../common.sh. Its first launch installs g.Pype, g.tec's
# gtec-unicorn driver and the other pinned libraries from PyPI. gtec-unicorn's
# macOS wheels need macOS 14 on Apple Silicon, hence the app's minimum.
#
# Environment:
#   PYTHON_HOME   the Python to bundle (default: uv's CPython 3.12)
#   OUT           output folder (default: dist/)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${OUT:-$ROOT/dist}"
DEV_PY="$ROOT/.venv/bin/python"

[ -x "$DEV_PY" ] || { echo "Run pip install -e '.[test,shimmer]' into .venv first." >&2; exit 1; }
PYTHON_HOME="${PYTHON_HOME:-$(dirname "$(dirname "$(readlink -f "$(uv python find 3.12)")")")}"
[ -x "$PYTHON_HOME/bin/python3" ] || { echo "No Python at $PYTHON_HOME" >&2; exit 1; }
ARCH="$(file -b "$(readlink -f "$PYTHON_HOME/bin/python3")" | grep -o 'arm64\|x86_64' | head -1)"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
source "$HERE/../common.sh"
BUILD_ID="$VERSION-$COMMIT-$ARCH"
APP="$OUT/brAIn stream.app"
RES="$APP/Contents/Resources"
echo "Building brAIn stream $BUILD_ID"

build_wheels
MACOSX_DEPLOYMENT_TARGET=14.0 pin_requirements \
    "$([ "$ARCH" = arm64 ] && echo aarch64 || echo x86_64)-apple-darwin"

# -- the bundle ---------------------------------------------------------------
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$RES/wheels"
cp -R "$PYTHON_HOME" "$RES/python"
rm -rf "$RES/python/include" "$RES/python/share" \
    "$RES/python/lib/python3.12/idlelib" "$RES/python/lib/python3.12/turtledemo"
"$RES/python/bin/python3" -I -m compileall -q -j 0 "$RES/python/lib" >/dev/null || true
cp "$WORK"/wheels/*.whl "$RES/wheels/"
cp "$WORK/requirements.txt" "$RES/requirements.txt"
cp "$HERE/../bootstrap.py" "$RES/bootstrap.py"
cp "$ROOT/LICENSE" "$RES/LICENSE"
echo "$BUILD_ID" > "$RES/build-id"
"$DEV_PY" "$HERE/make_icns.py" "$ROOT/src/brainstream/resources/brainstream.svg" \
    "$RES/brainstream.icns" >/dev/null
install -m 755 "$HERE/launcher.sh" "$APP/Contents/MacOS/brainstream"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>brAIn stream</string>
    <key>CFBundleDisplayName</key><string>brAIn stream</string>
    <key>CFBundleIdentifier</key><string>dk.itu.brainlab.brainstream</string>
    <key>CFBundleExecutable</key><string>brainstream</string>
    <key>CFBundleIconFile</key><string>brainstream</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>$VERSION</string>
    <key>CFBundleVersion</key><string>$BUILD_ID</string>
    <key>LSMinimumSystemVersion</key><string>14.0</string>
    <key>LSApplicationCategoryType</key><string>public.app-category.education</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSCameraUsageDescription</key>
    <string>brAIn stream records video from the webcam.</string>
    <key>NSBluetoothAlwaysUsageDescription</key>
    <string>brAIn stream reads the Unicorn EEG headset, the Polar H10 and the Shimmer over Bluetooth.</string>
    <key>NSLocalNetworkUsageDescription</key>
    <string>brAIn stream reads eye trackers over the Open Gaze API and receives LSL streams and triggers.</string>
</dict>
</plist>
PLIST

# Ad-hoc signature: enough to run here; distributing to other Macs without
# Gatekeeper warnings needs a Developer ID signature and notarization.
codesign --force --deep --sign - "$APP" >/dev/null

# -- disk image -----------------------------------------------------------------
STAGE="$WORK/dmg"
mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
DMG="$OUT/brAInstream-$VERSION-macos-$ARCH.dmg"
rm -f "$DMG"
hdiutil create -quiet -volname "brAIn stream" -srcfolder "$STAGE" -ov \
    -format UDZO "$DMG"

echo "Built:"
du -sh "$APP" "$DMG"
