#!/bin/bash
# Builds llama.cpp b11963 for the G4 (sm_120) with the patches in patches/series into /content/llama/bin, about 2 minutes.
set -uo pipefail
COMMIT=39febd2adb947b70da35e95f9351b3bf58ad170a
R=/content/res; P=/content/vm/patches; S=/content/src/llama.cpp; D=/content/llama/bin
st() { echo "$(date +%T) $*" >> $R/build.txt; }
st "build start"
command -v cmake > /dev/null || pip install -q cmake > $R/build_cmake.log 2>&1
mkdir -p /content/src
[ -d $S/.git ] || git clone -q --depth 1 --branch b11963 https://github.com/ggml-org/llama.cpp.git $S || { st "clone FAILED"; exit 1; }
[ "$(git -C $S rev-parse HEAD)" = "$COMMIT" ] || { st "commit mismatch $(git -C $S rev-parse HEAD) FAILED"; exit 1; }
for p in $(grep -v '^#' $P/series); do python3 $P/$p $S >> $R/build.txt 2>&1 || { st "patch $p FAILED"; exit 1; }; done
export PATH=/usr/local/cuda/bin:$PATH
cmake -S $S -B $S/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=ON \
  -DCMAKE_BUILD_TYPE=Release > $R/build_cmake.log 2>&1 || { st "cmake FAILED: $(tail -3 $R/build_cmake.log | tr '\n' ' ')"; exit 1; }
nice -n 5 cmake --build $S/build -j 40 --target llama-server > $R/build_make.log 2>&1 || { st "make FAILED: $(grep -m3 -i error $R/build_make.log | tr '\n' ' ')"; exit 1; }
mkdir -p $D && cp -a $S/build/bin/* $D/ && find $S/build -name "*.so*" -exec cp -a {} $D/ \;
for l in libcudart.so libcublas.so libcublasLt.so; do cp -a /usr/local/cuda/lib64/$l* $D/ 2>/dev/null; done
st "build done"
