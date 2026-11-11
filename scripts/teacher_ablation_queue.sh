#!/usr/bin/env bash
set -euo pipefail
cd /root/tableqa-research
root=runs/teacher-ablation-20261107
export PYTHONPATH="/root/tableqa-research/$root/code"
test ! -e "$root/QUEUE_STARTED"
date -u > "$root/QUEUE_STARTED"
trap 'echo "teacher queue failed at line $LINENO" > "$root/QUEUE_FAILED"' ERR
ready=0
for attempt in $(seq 1 480); do
  if [[ -f runs/joint-core-20261107/QUEUE_FAILED ]]; then
    echo STUDENT_QUEUE_FAILED > "$root/QUEUE_FAILED"
    exit 1
  fi
  if [[ -f runs/joint-core-20261107/QUEUE_COMPLETE ]]; then ready=1;break;fi
  sleep 30
done
if [[ "$ready" != 1 ]]; then echo STUDENT_WAIT_TIMEOUT > "$root/QUEUE_FAILED";exit 1;fi
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -u "$root/code/scripts/teacher_module_ablation.py" run "$root/prepared" > "$root/teacher.log" 2>&1
date -u > "$root/QUEUE_COMPLETE"
