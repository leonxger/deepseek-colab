#!/bin/bash
# Starts basalt serve detached on $PORT (18081) with split Engram + optional punch-on-read.
set -uo pipefail
S=/content/src/basalt
R=/content/res
BIN="$S/bin/basalt"
[ -x "$BIN" ] || BIN="$S/build/bin/basalt"
export PORT=${PORT:-18081}
export CTX=${CTX:-262144}
export MMPROJ=${MMPROJ:-1}
export DRAFT=${DRAFT:-1}
export VRAM_RESERVE=${VRAM_RESERVE:-700}
export RAM_BUDGET=${RAM_BUDGET:-55}
export DRAFT_EXPERTS=${DRAFT_EXPERTS:-32}
export SPEC_MAX=${SPEC_MAX:-6}
export ANSWER_ROOM=${ANSWER_ROOM:-0}
export MODEL_PATH=${MODEL_PATH:-/mnt/ram/model.basalt}
export ENGRAM_PATH=${ENGRAM_PATH:-/mnt/ram/engram.bin}
export CONV_CACHE_MIB=${CONV_CACHE_MIB:-32768}
export CONV_CACHE_DIR=${CONV_CACHE_DIR:-/content/conv_cache}
mkdir -p "$CONV_CACHE_DIR"

python3 - "$R/basalt_config.json" <<'EOF'
import json, os, sys
ar = os.environ["ANSWER_ROOM"]
engine = {
    "vram-reserve-mib": int(os.environ["VRAM_RESERVE"]),
    "resident-budget-gib": int(os.environ["RAM_BUDGET"]),
}
if int(os.environ["CONV_CACHE_MIB"]) > 0:
    engine["conversation-cache-disk-mib"] = int(os.environ["CONV_CACHE_MIB"])
    engine["conversation-cache-dir"] = os.environ["CONV_CACHE_DIR"]
cfg = {
    "version": 2,
    "models": [{
        "model": os.environ["MODEL_PATH"],
        "alias": "ds41",
        "gpu": "0",
        "max-context": int(os.environ["CTX"]),
        "kv": "fp8",
        "parallel": 2,
        "vision": "dynamic" if os.environ["MMPROJ"] == "1" else "off",
        "spec": int(os.environ["SPEC_MAX"]) if os.environ["DRAFT"] == "1" else 0,
        "draft-experts": int(os.environ["DRAFT_EXPERTS"]) if os.environ["DRAFT"] == "1" else 0,
        "answer-room": int(ar) if ar.isdigit() else ar,
        "idle-unload": 0,
    }],
    "engine": engine,
    "server": {
        "host": "127.0.0.1",
        "port": int(os.environ["PORT"]),
        "allowed-host": ["*"],
    },
}
open(sys.argv[1], "w").write(json.dumps(cfg, indent=2) + "\n")
EOF

export BASALT_ENGRAM_FILE="$ENGRAM_PATH"
if [ -n "${BASALT_PUNCH_FILE:-}" ]; then
  export BASALT_PUNCH_FILE
fi
echo "$BIN serve --config $R/basalt_config.json --api-key-file /content/secrets/key" > $R/serve_cmd.txt
cd "$S"
setsid nohup "$BIN" serve --config "$R/basalt_config.json" --api-key-file /content/secrets/key \
  > "$R/server.log" 2>&1 < /dev/null &
echo $! > "$R/server.pid"
