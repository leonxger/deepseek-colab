#!/bin/bash
# Builds patched jesdga95/basalt v0.5.7 (-DBASALT_MODELS=deepseek41) for Colab G4 (sm_120)
# plus standalone abliterate (zero llama.cpp dependency), ~75 seconds.
set -uo pipefail
TAG=v0.5.7
COMMIT=400b411c58031d064812ea21a259b2381acdaa3e
R=/content/res; V=/content/vm; S=/content/src/basalt; D=/content/basalt
st() { echo "$(date +%T) $*" >> $R/build.txt; }
mkdir -p $R /content/src $D
st "basalt build start"
command -v cmake > /dev/null || pip install -q cmake > $R/build_cmake.log 2>&1
if [ ! -d $S/.git ]; then
  git clone -q --depth 1 --branch $TAG https://github.com/jesdga95/basalt.git $S || { st "basalt clone FAILED"; exit 1; }
fi
[ "$(git -C $S rev-parse HEAD)" = "$COMMIT" ] || { st "basalt commit mismatch $(git -C $S rev-parse HEAD) FAILED"; exit 1; }
python3 $V/patch_basalt.py $S >> $R/build.txt 2>&1 || { st "basalt patch FAILED"; exit 1; }
for cand in /usr/local/cuda-13.2/bin /usr/local/cuda-13.1/bin /usr/local/cuda-13.0/bin /usr/local/cuda/bin; do
  [ -x "$cand/nvcc" ] && { export PATH="$cand:$PATH"; break; }
done
cmake -S $S -B $S/build -DCMAKE_BUILD_TYPE=Release -DBASALT_MODELS=deepseek41 -DBUILD_TESTING=OFF \
  > $R/build_cmake.log 2>&1 || { st "basalt cmake FAILED: $(tail -3 $R/build_cmake.log | tr '\n' ' ')"; exit 1; }
JOBS=$(nproc); [ "$JOBS" -gt 16 ] && JOBS=16
nice -n 5 cmake --build $S/build -j "$JOBS" --target basalt \
  > $R/build_make.log 2>&1 || { st "basalt make FAILED: $(grep -m3 -i error $R/build_make.log | tr '\n' ' ')"; exit 1; }
pip install -q -r $S/requirements.txt > $R/build_pip.log 2>&1 || { st "basalt pip FAILED"; exit 1; }
if [ -f $V/abliterate.c ]; then
  gcc -O3 -fopenmp -DABLITERATE_STANDALONE $V/abliterate.c -lm -o $D/abliterate \
    >> $R/build.txt 2>&1 || { st "standalone abliterate build FAILED"; exit 1; }
fi
st "basalt build done"
