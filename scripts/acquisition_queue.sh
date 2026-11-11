#!/usr/bin/env bash
set -euo pipefail
cd /root/tableqa-research
root=runs/real-acquisition-20261107
export PYTHONPATH="/root/tableqa-research/$root/code"
test ! -e "$root/QUEUE_STARTED"
date -u > "$root/QUEUE_STARTED"
trap 'echo "acquisition queue failed at line $LINENO" > "$root/QUEUE_FAILED"' ERR
ready=0
for attempt in $(seq 1 1200); do
  if [[ -f runs/joint-core-20261107/QUEUE_FAILED || -f runs/teacher-ablation-20261107/QUEUE_FAILED ]]; then exit 1;fi
  if [[ -f runs/teacher-ablation-20261107/QUEUE_COMPLETE ]]; then ready=1;break;fi
  sleep 30
done
test "$ready" = 1
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -u "$root/code/scripts/acquire_real_trajectories.py" run "$root/prepared" > "$root/acquisition.log" 2>&1
tar --exclude='*/__pycache__' -czf "$root/review-artifacts.tar.gz" -C "$root" prepared code queue.log acquisition.log QUEUE_STARTED
date -u > "$root/QUEUE_COMPLETE"
