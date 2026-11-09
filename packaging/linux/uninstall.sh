#!/bin/sh
# Remove brAIn dump for the current user. Projects, recordings and the
# list of recent projects (~/.config/braindump) are kept.
set -eu
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
rm -f "$HOME/.local/bin/braindump" \
    "$DATA/applications/braindump.desktop" \
    "$DATA/icons/hicolor/scalable/apps/braindump.svg" \
    "$DATA/icons/hicolor/256x256/apps/braindump.png"
rm -rf "$DATA/braindump"
update-desktop-database "$DATA/applications" >/dev/null 2>&1 || true
echo "brAIn dump is removed. Projects and recordings are kept."
