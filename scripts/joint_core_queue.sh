#!/usr/bin/env bash
set -euo pipefail
cd /root/tableqa-research
root=runs/joint-core-20261107
export PYTHONPATH="/root/tableqa-research/$root/code"
runner="$root/code/scripts/joint_core_stage.py"
stage="$root/prepared"
test -f "$stage/protocol.json"
test ! -e "$root/QUEUE_STARTED"
date -u > "$root/QUEUE_STARTED"
trap 'echo "queue failed at line $LINENO" > "$root/QUEUE_FAILED"' ERR
(
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u "$runner" B0 "$stage" > "$root/B0.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u "$runner" B2 "$stage" > "$root/B2.log" 2>&1
) &
first=$!
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u "$runner" B5 "$stage" > "$root/B5.log" 2>&1 &
second=$!
status_first=0;wait "$first" || status_first=$?
status_second=0;wait "$second" || status_second=$?
if [[ "$status_first" != 0 || "$status_second" != 0 ]]; then
  echo "worker exits: $status_first $status_second" > "$root/QUEUE_FAILED"
  exit 1
fi
.venv/bin/python -u "$runner" report "$stage" > "$root/report.log" 2>&1
date -u > "$root/QUEUE_COMPLETE"
