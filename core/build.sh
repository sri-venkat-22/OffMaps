#!/bin/sh
# Build libidr as a shared lib (Python ctypes / Android JNI load it) and the
# standalone replay tool. Same core object for both -- the dual-deliverable claim.
set -e
cd "$(dirname "$0")/.."
EXT=$([ "$(uname)" = "Darwin" ] && echo dylib || echo so)
SRC="core/eskf.cpp core/eskf3d.cpp core/speed_cal.cpp core/align.cpp core/gnss_quality.cpp core/map_match.cpp core/vib.cpp"
clang++ -std=c++17 -O2 -fPIC -shared $SRC -o core/libidr.$EXT
clang++ -std=c++17 -O2 $SRC tools/replay.cpp -o tools/replay
echo "built core/libidr.$EXT and tools/replay"
