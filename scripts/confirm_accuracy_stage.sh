#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
root=runs/accuracy-confirmation
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
echo waiting_for_all_tuning_conditions > "$root/stage.txt"
ready=0
for attempt in $(seq 1 1440);do
  ready=1
  for dir in accuracy-development-v1 scale-7b-accuracy balanced-accuracy;do
    s=$(cat "runs/$dir/stage.txt")
    case "$s" in failed_or_interrupted|prerequisite_failed|wait_timed_out|interrupted)
      echo prerequisite_failed > "$root/stage.txt";exit 1;;esac
    if [[ "$s" != complete_validation_tuning_confirmation_pending ]];then ready=0;fi
  done
  if [[ "$ready" == 1 ]];then break;fi
  sleep 30
done
if [[ "$ready" != 1 ]];then echo wait_timed_out > "$root/stage.txt";exit 1;fi
source .venv/bin/activate
export PYTHONPATH="$PWD/$frozen" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo selecting_then_confirming_once > "$root/stage.txt"
CUDA_VISIBLE_DEVICES=0 python "$frozen/scripts/confirm_accuracy.py"
echo complete_confirmation_no_retuning_on_this_cohort > "$root/stage.txt"
