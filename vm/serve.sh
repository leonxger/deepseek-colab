#!/bin/bash
# Starts llama-server detached with the tuned flags; MMPROJ_FETCH_ONLY=1 only downloads the image encoder.
MMP=/content/mmproj/mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf
if [ "${MMPROJ:-0}" = 1 ]; then  # image encoder on local disk (the RAM disk has no room), pinned HF revision, size and sha256 checked
  MM_SIZE=972208096; MM_SHA=0fdc4b7693f287e9665940cf42bd39c546ffe4f0def89c569d7d0e45d60d2df1
  mm_ok() { [ "$(stat -c %s $MMP 2>/dev/null)" = $MM_SIZE ] && sha256sum $MMP | grep -q "^$MM_SHA"; }
  mkdir -p /content/mmproj /content/res
  if ! mm_ok; then
    rm -f $MMP
    echo "[{\"url\": \"https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/resolve/a419d74cd28d5b9f8c5e2b9077d11a609c17bc32/$(basename $MMP)\", \"dest\": \"$MMP\", \"dest_size\": $MM_SIZE, \"src_off\": 0, \"dest_off\": 0, \"size\": $MM_SIZE}]" > /content/res/j_mmproj.json
    python3 /content/vm/pfetch.py /content/res/j_mmproj.json --workers 16 --chunk-mb 64 --tag mmproj && mm_ok || { echo "image encoder fetch or check FAILED"; exit 1; }
  fi
  [ "${MMPROJ_FETCH_ONLY:-0}" = 1 ] && exit 0
fi
BIN=/content/llama/bin/llama-server
export LLAMA_MOE_DEVCACHE=1 GGML_CUDA_HCFUSE=1 GGML_SCHED_ASYNC_INPUTS=1 GGML_CUDA_GRAPH_FIRST_CHUNK=32 LLAMA_KV_SEQRM_INDEX=1 GGML_CUDA_MOE_COPY_ALIGN=1
if [ "$MMPROJ" = 1 ] || [ "${DMA:-0}" != 1 ]; then  # misses by the copy engine stall on image prompts and were measured only in the fast layout
  export LLAMA_MOE_DEVCACHE_DMA=0
else
  export LLAMA_MOE_DEVCACHE_DMA=1 CUDA_DEVICE_MAX_CONNECTIONS=32
fi
ARGS=(-m /content/model/DeepSeek-V4.1-Flash-1017-GSQ-RCO-$RUNG-00001-of-00006.gguf -a ds41,ds41-max --host 127.0.0.1 --port $PORT
  -ngl 99 -fit off -ot "$CPU_RE" -fa on -c $CTX -np 2 -t 24 -b $UBATCH -ub $UBATCH -lv 4 -lm mmap --jinja --reasoning on --reasoning-effort 50
  --metrics --api-key-file /content/secrets/key --cache-ram $CACHE_RAM --moe-cache-mib $MOE_CACHE_MIB)
[ "$DRAFT" = 1 ] && ARGS+=(-md /mnt/ram/m/dspark.gguf --spec-type draft-dspark --spec-draft-n-max 2 -ngld 99 --spec-draft-n-ubatch 512
  --no-spec-draft-backend-sampling)  # backend draft sampling garbles the second sequence of a shared draft batch
[ "$MMPROJ" = 1 ] && ARGS+=(--mmproj $MMP)
echo "$BIN ${ARGS[*]}" > /content/res/serve_cmd.txt
cd "$(dirname "$BIN")"
rm -f /content/res/pin_result /content/res/pin_request
LD_PRELOAD=/content/vm/libpin.so LD_LIBRARY_PATH="$(dirname "$BIN"):/usr/lib64-nvidia" setsid nohup "$BIN" "${ARGS[@]}" > /content/res/server.log 2>&1 < /dev/null &
echo $! > /content/res/server.pid
