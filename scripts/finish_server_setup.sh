#!/usr/bin/env bash
# Continue the authorized setup without requiring the SSH client to remain open.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
for attempt in $(seq 1 240); do
  if grep -q '^OK$' runs/setup/tests.log 2>/dev/null && [[ -f models/Qwen2.5-3B-Instruct/DOWNLOAD_COMPLETE ]]; then
    export CUDA_VISIBLE_DEVICES=0
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    .venv/bin/python scripts/gpu_smoke.py
    exit 0
  fi
  printf 'Waiting for environment tests and verified 3B weights (%s/240)\n' "$attempt"
  sleep 30
done
echo 'Setup wait exceeded two hours; inspect bootstrap and download logs.' >&2
exit 1
