#!/bin/bash
# Build the Windows x86_64 package: dist/brAInstream-<version>-windows-x86_64.zip
#
#     packaging/windows/build_windows.sh
#
# Runs on macOS or Linux; it only downloads and arranges files. Needs uv,
# git and the project's .venv (pip install -e ".[test,shimmer]").
#
# The package carries its own Python (python-build-standalone's Windows
# build, which uv downloads) and the wheels of brAIn stream
# and OpenGP3; see ../common.sh.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${OUT:-$ROOT/dist}"
DEV_PY="$ROOT/.venv/bin/python"

[ -x "$DEV_PY" ] || { echo "Run pip install -e '.[test,shimmer]' into .venv first." >&2; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
source "$HERE/../common.sh"
BUILD_ID="$VERSION-$COMMIT-windows-x86_64"
NAME="brAInstream-$VERSION-windows-x86_64"
DIR="$WORK/$NAME"
echo "Building brAIn stream $BUILD_ID"

uv python install --quiet --install-dir "$WORK/py" cpython-3.12-windows-x86_64-none
PYTHON_HOME="$(find "$WORK/py" -maxdepth 1 -type d -name 'cpython-3.12.*-windows-x86_64-none' | head -1)"
[ -n "$PYTHON_HOME" ] || { echo "uv did not install a Windows CPython 3.12" >&2; exit 1; }

build_wheels
pin_requirements x86_64-pc-windows-msvc

mkdir -p "$DIR/wheels"
cp -R "$PYTHON_HOME" "$DIR/python"
# uv adds a "python" symlink to the install; Windows has no use for it.
find "$DIR/python" -maxdepth 1 -type l -delete
rm -rf "$DIR/python/include" "$DIR/python/libs" \
    "$DIR/python/Lib/idlelib" "$DIR/python/Lib/turtledemo"
cp "$WORK"/wheels/*.whl "$DIR/wheels/"
cp "$WORK/requirements.txt" "$DIR/requirements.txt"
cp "$HERE/../bootstrap.py" "$HERE/launcher.py" "$DIR/"
cp "$HERE/install.cmd" "$HERE/install.ps1" "$HERE/uninstall.cmd" \
    "$HERE/uninstall.ps1" "$HERE/brAIn stream.cmd" "$HERE/README.txt" "$DIR/"
cp "$ROOT/LICENSE" "$DIR/"
echo "$BUILD_ID" > "$DIR/build-id"
render_icon "$DIR/brainstream.ico" 256 ICO

mkdir -p "$OUT"
ZIP="$OUT/$NAME.zip"
rm -f "$ZIP"
"$DEV_PY" - "$WORK" "$NAME" "$ZIP" <<'PY'
import os, sys, zipfile
work, name, out = sys.argv[1:]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for folder, _, files in os.walk(os.path.join(work, name)):
        for file in sorted(files):
            path = os.path.join(folder, file)
            z.write(path, os.path.relpath(path, work))
PY
echo "Built:"
du -sh "$ZIP"
