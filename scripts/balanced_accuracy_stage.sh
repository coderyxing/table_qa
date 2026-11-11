#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
root=runs/balanced-accuracy
mkdir -p "$root"
exec 9>"$root/stage.lock"
flock -n 9 || exit 1
trap 'echo failed_or_interrupted > "$root/stage.txt"' ERR
trap 'echo interrupted > "$root/stage.txt";exit 143' TERM INT
frozen="$root/frozen-code"
if [[ ! -d "$frozen" ]];then
  mkdir -p "$frozen/tableqa" "$frozen/scripts"
  cp tableqa/*.py "$frozen/tableqa/";cp scripts/*.py "$frozen/scripts/";touch "$frozen/SNAPSHOT_READY"
fi
test -f "$frozen/SNAPSHOT_READY"
test -f runs/accuracy-training-pool/report.json
echo waiting_for_accuracy_controls > "$root/stage.txt"
ready=0
for attempt in $(seq 1 1440);do
  stage=$(cat runs/accuracy-development-v1/stage.txt)
  if [[ "$stage" == complete_validation_tuning_confirmation_pending ]];then ready=1;break;fi
  case "$stage" in failed_or_interrupted|interrupted) echo prerequisite_failed > "$root/stage.txt";exit 1;;esac
  sleep 30
done
if [[ "$ready" != 1 ]];then echo wait_timed_out > "$root/stage.txt";exit 1;fi
source .venv/bin/activate
export PYTHONPATH="$PWD/$frozen" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo training_and_evaluating_balanced_loss > "$root/stage.txt"
CUDA_VISIBLE_DEVICES=1 python "$frozen/scripts/balanced_accuracy_control.py"
echo complete_validation_tuning_confirmation_pending > "$root/stage.txt"
