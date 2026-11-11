#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
python scripts/download_model.py Qwen/Qwen2.5-32B-Instruct models/Qwen2.5-32B-Instruct --metadata-only
ready=0
for attempt in $(seq 1 8); do
  if python scripts/download_ranges.py Qwen/Qwen2.5-32B-Instruct models/Qwen2.5-32B-Instruct --workers 8; then
    ready=1
    break
  fi
  printf 'Download pass %s failed; preserving checked files and retrying.\n' "$attempt"
  sleep 15
done
test "$ready" -eq 1
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python scripts/teacher_pilot.py
