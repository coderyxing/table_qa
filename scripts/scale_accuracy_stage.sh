#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
root=runs/scale-7b-accuracy
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
echo waiting_for_accuracy_and_verified_weights > "$root/stage.txt"
ready=0
for attempt in $(seq 1 1440);do
  a=$(cat runs/accuracy-development-v1/stage.txt);m=$(cat runs/model-7b/stage.txt)
  if [[ "$a" == failed_or_interrupted || "$a" == interrupted || "$m" == failed_or_interrupted ]];then
    echo prerequisite_failed > "$root/stage.txt";exit 1
  fi
  if [[ "$a" == complete_validation_tuning_confirmation_pending && "$m" == verified_complete ]];then ready=1;break;fi
  sleep 30
done
if [[ "$ready" != 1 ]];then echo wait_timed_out > "$root/stage.txt";exit 1;fi
source .venv/bin/activate
export PYTHONPATH="$PWD/$frozen" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo training_and_evaluating_7B > "$root/stage.txt"
CUDA_VISIBLE_DEVICES=0 python "$frozen/scripts/scale_accuracy_control.py"
echo complete_validation_tuning_confirmation_pending > "$root/stage.txt"
