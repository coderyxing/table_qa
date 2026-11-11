#!/usr/bin/env bash
# A single-GPU student experiment, serialized after the dual-GPU teacher run.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
root=runs/optimized-student-control
mkdir -p "$root"
exec 9>"$root/stage.lock"
flock -n 9 || exit 1
trap 'echo failed_or_interrupted > "$root/stage.txt"' ERR
trap 'echo interrupted > "$root/stage.txt"; exit 143' TERM INT
frozen="$root/frozen-code"
if [[ ! -d "$frozen" ]];then
  mkdir -p "$frozen/tableqa" "$frozen/scripts"
  cp tableqa/*.py "$frozen/tableqa/"
  cp scripts/*.py "$frozen/scripts/"
  touch "$frozen/SNAPSHOT_READY"
fi
test -f "$frozen/SNAPSHOT_READY"
test -f runs/optimization-audit/report.json
echo waiting_for_interactive_teacher > "$root/stage.txt"
for attempt in $(seq 1 1440);do
  stage=$(cat runs/interactive-teacher/stage.txt)
  if [[ "$stage" == complete_development_only ]];then break;fi
  case "$stage" in failed_or_interrupted|interrupted)
    echo prerequisite_failed > "$root/stage.txt";exit 1;;esac
  sleep 30
done
if [[ "$(cat runs/interactive-teacher/stage.txt)" != complete_development_only ]];then
  echo wait_timed_out > "$root/stage.txt";exit 1
fi
source .venv/bin/activate
export PYTHONPATH="$PWD/$frozen" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo training_and_evaluating_search_fixed > "$root/stage.txt"
CUDA_VISIBLE_DEVICES=0 python "$frozen/scripts/optimized_student_control.py"
echo complete_development_only > "$root/stage.txt"
