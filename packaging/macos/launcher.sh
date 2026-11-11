#!/bin/bash
# brAIn dump.app's executable.
#
# The app carries its own Python and the wheels of brAIn dump and
# OpenGP3, but not g.Pype or g.tec's gtec-unicorn driver:
# their licences do not allow them in a bundle. On first launch, and after
# an update, bootstrap.py installs them and the other libraries from PyPI
# into ~/Library/Application Support/brAIn dump/runtime.
set -u

# This app declares Bluetooth use, so brAIn dump need not relaunch itself
# to get it (see braindump/macos.py).
export BRAINDUMP_BUNDLED=1

CONTENTS="$(cd "$(dirname "$0")/.." && pwd)"
RES="$CONTENTS/Resources"
PYTHON="$RES/python/bin/python3"
BUILD="$(cat "$RES/build-id")"
LIB="$HOME/Library/Application Support/brAIn dump/runtime/$BUILD"

if [ ! -f "$LIB/.complete" ]; then
    "$PYTHON" -I -B "$RES/bootstrap.py" "$RES" "$LIB" || exit 1
fi

# -I: ignore the environment and the user's site-packages; -B: write no
# bytecode into the signed bundle. The libraries come from $LIB alone.
exec "$PYTHON" -I -B -c '
import sys
sys.path.insert(0, sys.argv.pop(1))
sys.argv[0] = "braindump"
from braindump.app import main
sys.exit(main())
' "$LIB" "$@"
