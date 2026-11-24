#!/bin/bash
# Cold start on a Colab G4 using Basalt v0.5.7 (DeepSeek-V4.1-Flash UD-IQ1_M):
#   1. Parallel: build Basalt (~80s) + fetch sparse backbone into /mnt/ram (~50s) + fetch abliteration edit
#   2. In-place abliteration via standalone abliterate (~1.5s, 62/62 tensors)
#   3. Start basalt serve + POST /load (with BASALT_PUNCH_FILE punch-on-read reclaiming ~140 GiB tmpfs in ~12s!)
#   4. Fetch 110.6 GB Engram into /mnt/ram/engram.bin (~34s @ 3.2 GB/s)
#   5. Warmup + open gate.py on port 18080 (~2.4 min VM boot!)
set -uo pipefail
: "${CTX:?} ${MMPROJ:?}"
export QUANT=${QUANT:-iq1_m} PORT=18081 DRAFT=${DRAFT:-1} UNCENSOR=${UNCENSOR:-1} WORKERS=${WORKERS:-64}
export VRAM_RESERVE=${VRAM_RESERVE:-700} RAM_BUDGET=${RAM_BUDGET:-55} DRAFT_EXPERTS=${DRAFT_EXPERTS:-32}
[ -s /content/secrets/hf_hdr ] && export HF_HDR=/content/secrets/hf_hdr
R=/content/res; V=/content/vm; D=/content/basalt
mkdir -p $R /mnt/ram /content/patch $D
rm -f $R/READY

T0=$(date +%s)
st() { echo "$(( $(date +%s) - T0 ))s $(date +%T) $*" >> $R/timeline.txt; }
seen() { grep -qE "$1" $R/timeline.txt; }
hdr() { echo "Authorization: Bearer $(cat /content/secrets/key)"; }

ABLIT_URL=https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-abliterated/resolve/4433164758a8045b1b915b7644cce97b9a0e274c/abliteration-edit.npz
ABLIT_SHA=83273cdbe06e65ad2852dd09e8d976a0e8ce3209788938add325989c20b63b02

st "boot start engine=basalt quant=$QUANT ctx=$CTX images=$MMPROJ draft=$DRAFT uncensored=$UNCENSOR"
RAM_GB=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1000000 - 6 ))
mountpoint -q /mnt/ram || mount -t tmpfs -o size=${RAM_GB}G tmpfs /mnt/ram

# UD-IQ1_M backbone (139.1 GiB) fits in /mnt/ram with punch-on-read; fall back to /content disk on smaller-RAM VMs.
if [ "$QUANT" = iq1_m ] && [ "$RAM_GB" -ge 162 ]; then
  export MODEL_PATH=/mnt/ram/model.basalt
  export BASALT_PUNCH_FILE=/mnt/ram/model.basalt
else
  export MODEL_PATH=/content/model.basalt
  export BASALT_PUNCH_FILE=""
fi
export ENGRAM_PATH=/mnt/ram/engram.bin

python3 $V/basalt_jobs.py --quant "$QUANT" \
  --backbone-dest "$MODEL_PATH" --backbone-jobs "$R/j_basalt_bb.json" \
  --engram-dest "$ENGRAM_PATH" --engram-jobs "$R/j_basalt_eng.json" \
  --engram-size-out "$R/engram_size.txt" > $R/fetch.log 2>&1 || { st "basalt_jobs FAILED"; exit 1; }
ENGRAM_SIZE=$(cat $R/engram_size.txt)

{ bash $V/build_basalt.sh > $R/basalt_build.log 2>&1 && st "basalt built" || st "basalt build FAILED ($(tail -1 $R/build.txt 2>/dev/null))"; } &
if [ "$UNCENSOR" = 1 ]; then
  { curl -sfL --retry 5 ${HF_HDR:+-H @$HF_HDR} -o /content/patch/abliteration-edit.npz "$ABLIT_URL" \
    && sha256sum /content/patch/abliteration-edit.npz | grep -q "^$ABLIT_SHA" \
    && st "abliteration edit fetched" || st "abliteration edit FAILED"; } &
fi
{ python3 $V/pfetch.py $R/j_basalt_bb.json --workers $WORKERS --chunk-mb 64 --tag basalt-bb --log $R/fetch.log \
  && st "backbone fetched (basalt $QUANT)" || st "backbone FAILED"; } &
wait
seen FAILED && { st "fetch/build FAILED, stopping"; exit 1; }

if [ "$UNCENSOR" = 1 ]; then
  python3 -I $V/abliterate_basalt.py "$MODEL_PATH" /content/patch/abliteration-edit.npz "$D/abliterate" \
    > $R/patch.log 2>&1 && st "abliteration applied" || { st "abliteration FAILED ($(tail -1 $R/patch.log 2>/dev/null))"; exit 1; }
fi

# Create 0-block sparse engram.bin so PleReader::open succeeds before we fill it after punch-on-read
python3 -I -c "import os; f=os.open('$ENGRAM_PATH', os.O_WRONLY|os.O_CREAT, 0o644); os.ftruncate(f, int('$ENGRAM_SIZE')); os.close(f)"

bash $V/serve_basalt.sh
st "server started"
for k in $(seq 1 120); do
  curl -sf -o /dev/null -H "$(hdr)" http://127.0.0.1:$PORT/health && { st "basalt http ready"; break; }
  [ -d /proc/$(cat $R/server.pid) ] || { st "server died ($(tail -5 $R/server.log 2>/dev/null | tr '\n' ' '))"; exit 1; }
  sleep 0.5
done

# Trigger eager weight load (without running any token so PleReader RowCache stays empty before Engram download)
curl -sf --max-time 900 -X POST -H "$(hdr)" -H "Content-Type: application/json" -d '{}' \
  http://127.0.0.1:$PORT/load > $R/load.json 2>&1 && st "weights loaded" || { st "weights load FAILED ($(tail -5 $R/server.log 2>/dev/null | tr '\n' ' '))"; exit 1; }

VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${VRAM:-0}" -lt 60000 ]; then
  st "GUARD: VRAM ${VRAM} MiB after /load FAILED"; exit 1
fi
if [ -n "${BASALT_PUNCH_FILE:-}" ]; then
  python3 $V/basalt_jobs.py --backbone-dest "$MODEL_PATH" --punch-sections dspark/gguf dense > $R/punch.txt 2>&1 || true
fi
st "pin: VRAM ${VRAM} MiB, model $(du -sh "$MODEL_PATH" | awk '{print $1}')"

# Fill 110.6 GB Engram table into /mnt/ram/engram.bin now that tmpfs backbone was punched down to ~2 GiB
( for k in $(seq 1 600); do [ -f $R/READY ] && exit 0; a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    [ "$a" -lt 1200 ] && { for p in $(pgrep -f "[p]fetch.py"); do kill $p; done; st "GUARD: ${a} MiB RAM available, engram fetch stopped FAILED"; exit 0; }; sleep 2; done ) &
python3 $V/pfetch.py $R/j_basalt_eng.json --workers $WORKERS --chunk-mb 64 --tag basalt-eng --log $R/fetch.log \
  && st "engram done" || { st "engram FAILED"; exit 1; }
free -b > $R/free_after_engram.txt
touch $R/READY

curl -sf --max-time 180 -H "$(hdr)" http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"ds41","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"temperature":0,"reasoning_effort":"low"}' \
  > $R/warmup.json && st "first token (warmup done)" || { st "warmup FAILED"; exit 1; }

setsid nohup python3 $V/gate.py 18080 $PORT > $R/gate.log 2>&1 < /dev/null &
for k in $(seq 1 50); do grep -q "gate open" $R/gate.log 2>/dev/null && break; sleep 0.1; done
grep -q "gate open" $R/gate.log && st "gate open 18080" || st "gate FAILED"
