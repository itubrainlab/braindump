#!/usr/bin/env bash
# Qt's system libraries on the Ubuntu runners. Skips apt when they are
# already there; otherwise uses Ubuntu's own archive, since the runners'
# Azure mirror is sometimes unreachable, and retries.
set -u
PKGS="libegl1 libgl1 libxkbcommon0 libfontconfig1 libdbus-1-3"
if dpkg -s $PKGS >/dev/null 2>&1; then
    echo "Qt's libraries are already installed."
    exit 0
fi
sources=/etc/apt/sources.list.d/ubuntu.sources
if [ -f "$sources" ]; then
    sudo sed -i 's#mirror+file:/etc/apt/apt-mirrors.txt#http://archive.ubuntu.com/ubuntu/#' "$sources"
fi
for attempt in 1 2 3; do
    sudo apt-get update -q && sudo apt-get install -yq $PKGS && exit 0
    echo "apt failed (attempt $attempt); retrying."
    sleep 20
done
exit 1
