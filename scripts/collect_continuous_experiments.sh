#!/usr/bin/env bash
# Bounded artifact collection. The models are managed by separate fixed queues.
set -euo pipefail
cd /root/tableqa-research
student=runs/joint-core-20261107
teacher=runs/teacher-ablation-20261107
export PYTHONPATH="/root/tableqa-research/$teacher/code"
trap 'echo "artifact collection failed at line $LINENO" > "$student/ARTIFACT_FAILED"' ERR
ready=0
for attempt in $(seq 1 480); do
  if [[ -f "$student/QUEUE_FAILED" ]]; then exit 1;fi
  if [[ -f "$student/QUEUE_COMPLETE" ]]; then ready=1;break;fi
  sleep 30
done
test "$ready" = 1
.venv/bin/python runs/finalize_joint_core.py "$student/prepared" > "$student/artifact.log" 2>&1
tar --exclude='*.safetensors' --exclude='*/checkpoint-*' --exclude='*/__pycache__' \
  -czf "$student/review-artifacts.tar.gz" -C "$student" prepared code queue.sh queue.log QUEUE_STARTED QUEUE_COMPLETE B0.log B2.log B5.log report.log artifact.log
date -u > "$student/ARTIFACT_COMPLETE"
ready=0
for attempt in $(seq 1 720); do
  if [[ -f "$teacher/QUEUE_FAILED" ]]; then exit 1;fi
  if [[ -f "$teacher/QUEUE_COMPLETE" ]]; then ready=1;break;fi
  sleep 30
done
test "$ready" = 1
tar --exclude='*/__pycache__' -czf "$teacher/review-artifacts.tar.gz" -C "$teacher" prepared code queue.log teacher.log QUEUE_STARTED QUEUE_COMPLETE
date -u > "$teacher/ARTIFACT_COMPLETE"
