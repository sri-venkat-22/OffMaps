#!/bin/sh
# OffMaps: one command to build, test and demo the navigation engine.
#
#   ./run.sh              setup (venv + requirements + C++ core), tests, demo
#   ./run.sh setup        only the setup
#   ./run.sh test         the pytest suite (real-data tests skip without IO-VNBD)
#   ./run.sh demo         the live loop through GNSS outages (py/demo.py)
#   ./run.sh ablation     the named-stage ablation on real drives (needs IO-VNBD, ~1 h)
#   ./run.sh edge ARGS    the edge engine CLI on your own IMU/GNSS CSVs, e.g.
#                         ./run.sh edge --dir mydrive --out track.csv --outage 300:360
#
# Real data: IO-VNBD, resynchronised as in REALDATA.md, at ~/OffMaps-data/IO-VNBD-sync
# (or set OFFMAPS_IOVNBD). Without it everything still runs; accuracy tests skip and
# the demo runs a synthetic plumbing check instead of the real drive.
# The Android app is built separately: see android/README.md.
set -e
CALLER="$(pwd)"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
VENV="$ROOT/.venv"

say() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

setup() {
    say "Python environment ($VENV)"
    "$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ needed"' \
        || { echo "Python 3.10+ is required (set PYTHON=/path/to/python3)"; exit 1; }
    [ -d "$VENV" ] || "$PY" -m venv "$VENV"
    "$VENV/bin/python" -m pip install --quiet --upgrade pip
    "$VENV/bin/python" -m pip install --quiet -r py/requirements.txt
    say "C++ core (core/libidr + tools/replay)"
    command -v clang++ >/dev/null || command -v g++ >/dev/null \
        || { echo "a C++17 compiler is required (clang++ or g++)"; exit 1; }
    sh core/build.sh
}

ensure() { [ -x "$VENV/bin/python" ] && [ -f core/libidr.dylib -o -f core/libidr.so ] || setup; }
pyrun() { (cd py && PYTHONPATH=. "$VENV/bin/python" "$@"); }

case "${1:-all}" in
    setup)    setup ;;
    test)     ensure; say "Tests"; pyrun -m pytest -q -p no:warnings ;;
    demo)     ensure; say "Demo"; pyrun demo.py ;;
    ablation) ensure; say "Ablation (real drives)"
              pyrun ablation.py --set val --out ../out/ablation/val.json
              pyrun ablation.py --set lodo --out ../out/ablation/lodo.json
              pyrun ablation.py --report ../out/ablation/lodo.json ../out/ablation/val.json ;;
    edge)     ensure; shift; say "Edge engine"
              # relative paths in ARGS are the caller's, so run from there
              (cd "$CALLER" && PYTHONPATH="$ROOT/py" "$VENV/bin/python" -m edge_engine "$@") ;;
    all)      setup; say "Tests"; pyrun -m pytest -q -p no:warnings; say "Demo"; pyrun demo.py ;;
    *)        sed -n '2,15p' "$0"; exit 1 ;;
esac
