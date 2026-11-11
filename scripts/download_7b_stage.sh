#!/usr/bin/env bash
# Public planned scale control; CPU/network only, with existing hash/disk guards.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -p runs/model-7b
exec 9>runs/model-7b/download.lock
flock -n 9 || exit 1
trap 'echo failed_or_interrupted > runs/model-7b/stage.txt' ERR
echo downloading > runs/model-7b/stage.txt
ready=0
for attempt in 1 2 3;do
  if .venv/bin/python scripts/download_model.py Qwen/Qwen2.5-7B-Instruct models/Qwen2.5-7B-Instruct --metadata-only;then ready=1;break;fi
  sleep 10
done
test "$ready" = 1
.venv/bin/python scripts/download_ranges.py Qwen/Qwen2.5-7B-Instruct models/Qwen2.5-7B-Instruct --workers 8
test -f models/Qwen2.5-7B-Instruct/DOWNLOAD_COMPLETE
echo verified_complete > runs/model-7b/stage.txt
