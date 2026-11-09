# Shared by the three build scripts: the version, the build id, the
# bundled wheels and the pinned requirements. Source it after setting
# ROOT, WORK and DEV_PY.
#
# The packages bundle brAIn dump's wheel and those of its two lab
# libraries, Rhino Recorder and OpenGP3 (both MIT), which it depends on by
# git URL: installing them from git would need git on every computer.
# Everything else, g.Pype and g.tec's gtec-unicorn driver among it, is
# installed from PyPI on first launch (see bootstrap.py): g.tec's licences
# forbid redistributing g.Pype and the driver in a bundle.

VERSION="$("$DEV_PY" -c 'import braindump; print(braindump.__version__)')"
COMMIT="$(git -C "$ROOT" rev-parse --short HEAD)"
git -C "$ROOT" diff --quiet HEAD -- src pyproject.toml packaging || COMMIT="$COMMIT-dirty"

# The git dependencies, as "name url@ref" lines.
GIT_DEPS="$("$DEV_PY" - "$ROOT/pyproject.toml" <<'PY'
import re, sys, tomllib
for dep in tomllib.load(open(sys.argv[1], "rb"))["project"]["dependencies"]:
    m = re.match(r"\s*([\w.-]+)\s*@\s*git\+(\S+)", dep)
    if m:
        print(m.group(1), m.group(2))
PY
)"

build_wheels() {
    mkdir -p "$WORK/wheels" "$WORK/src"
    uv build --quiet --wheel --out-dir "$WORK/wheels" "$ROOT"
    while read -r name url; do
        [ -n "$name" ] || continue
        repo="${url%@*}"
        ref="${url##*@}"
        git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$ref" "$repo" "$WORK/src/$name"
        uv build --quiet --wheel --out-dir "$WORK/wheels" "$WORK/src/$name"
    done <<< "$GIT_DEPS"
}

# requirements.txt for a platform: brAIn dump's and its lab libraries'
# dependencies from PyPI, at the versions the tests ran with.
pin_requirements() {
    local platform="$1"
    "$DEV_PY" - "$ROOT/pyproject.toml" > "$WORK/requirements.in" <<'PY'
import re, sys, tomllib
from importlib.metadata import requires
project = tomllib.load(open(sys.argv[1], "rb"))["project"]
deps = list(project["dependencies"])
deps += project.get("optional-dependencies", {}).get("shimmer", [])
lab = []
for dep in deps:
    m = re.match(r"\s*([\w.-]+)\s*@\s*git\+", dep)
    if m:
        lab.append(m.group(1))
    else:
        print(dep)
for name in lab:
    for dep in requires(name) or []:
        if "extra ==" not in dep:
            print(dep)
PY
    uv pip freeze --python "$DEV_PY" --exclude-editable | grep -v ' @ ' \
        > "$WORK/constraints.txt"
    uv pip compile --quiet --python-version 3.12 --python-platform "$platform" \
        --constraint "$WORK/constraints.txt" --no-header --no-annotate \
        "$WORK/requirements.in" -o "$WORK/requirements.txt"
}

render_icon() {  # render_icon <out> <size> [format]
    QT_QPA_PLATFORM=offscreen "$DEV_PY" -c '
import sys
from PySide6.QtGui import QGuiApplication, QIcon
app = QGuiApplication([])
fmt = sys.argv[4] if len(sys.argv) > 4 else None
QIcon(sys.argv[1]).pixmap(int(sys.argv[3]), int(sys.argv[3])).save(sys.argv[2], fmt)
' "$ROOT/src/braindump/resources/braindump.svg" "$1" "$2" ${3:-}
}
