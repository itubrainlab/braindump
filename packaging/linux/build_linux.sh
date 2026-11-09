#!/bin/bash
# Build the Linux x86_64 package: dist/brAIndump-<version>-linux-x86_64.tar.gz
#
#     packaging/linux/build_linux.sh
#
# Runs on macOS or Linux; it only downloads and arranges files. Needs uv,
# git and the project's .venv (pip install -e ".[test,shimmer]").
#
# The package carries its own Python (python-build-standalone, which uv
# downloads) and the wheels of brAIn dump, Rhino Recorder and OpenGP3;
# see ../common.sh. g.Pype publishes no ARM Linux build, and PySide6's
# Linux wheels need glibc 2.34, hence x86_64 and manylinux_2_34.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="${OUT:-$ROOT/dist}"
DEV_PY="$ROOT/.venv/bin/python"

[ -x "$DEV_PY" ] || { echo "Run pip install -e '.[test,shimmer]' into .venv first." >&2; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
source "$HERE/../common.sh"
BUILD_ID="$VERSION-$COMMIT-linux-x86_64"
NAME="brAIndump-$VERSION-linux-x86_64"
DIR="$WORK/$NAME"
echo "Building brAIn dump $BUILD_ID"

uv python install --quiet --install-dir "$WORK/py" cpython-3.12-linux-x86_64-gnu
PYTHON_HOME="$(find "$WORK/py" -maxdepth 1 -type d -name 'cpython-3.12.*-linux-x86_64-gnu' | head -1)"
[ -n "$PYTHON_HOME" ] || { echo "uv did not install a Linux CPython 3.12" >&2; exit 1; }

build_wheels
pin_requirements x86_64-manylinux_2_34

mkdir -p "$DIR/wheels"
cp -R "$PYTHON_HOME" "$DIR/python"
rm -rf "$DIR/python/include" "$DIR/python/share" \
    "$DIR/python/lib/python3.12/idlelib" "$DIR/python/lib/python3.12/turtledemo"
cp "$WORK"/wheels/*.whl "$DIR/wheels/"
cp "$WORK/requirements.txt" "$DIR/requirements.txt"
cp "$HERE/../bootstrap.py" "$DIR/bootstrap.py"
cp "$ROOT/LICENSE" "$HERE/README.txt" "$DIR/"
echo "$BUILD_ID" > "$DIR/build-id"
install -m 755 "$HERE/braindump" "$HERE/install.sh" "$HERE/uninstall.sh" "$DIR/"
cp "$ROOT/src/braindump/resources/braindump.svg" "$DIR/braindump.svg"
render_icon "$DIR/braindump.png" 256

mkdir -p "$OUT"
TARBALL="$OUT/$NAME.tar.gz"
COPYFILE_DISABLE=1 tar -C "$WORK" -czf "$TARBALL" "$NAME"
echo "Built:"
du -sh "$TARBALL"
