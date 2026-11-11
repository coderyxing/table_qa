#!/usr/bin/env bash
# Queued, bounded experiment. It does not change or interrupt the frozen batch.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
root=runs/serialization-followup
mkdir -p "$root"
exec 9>"$root/stage.lock"
flock -n 9 || exit 1
trap 'echo failed_or_interrupted > "$root/stage.txt"' ERR
frozen="$root/frozen-code"
if [[ ! -d "$frozen" ]]; then
  mkdir -p "$frozen/tableqa" "$frozen/scripts"
  cp tableqa/*.py "$frozen/tableqa/"
  cp scripts/*.py "$frozen/scripts/"
  touch "$frozen/SNAPSHOT_READY"
fi
test -f "$frozen/SNAPSHOT_READY"
echo waiting_for_medium_batch > "$root/stage.txt"
for attempt in $(seq 1 1440); do
  stage=$(cat runs/medium-development/stage.txt)
  if [[ "$stage" == development_complete_freeform_review_still_pending ]]; then break; fi
  if [[ "$stage" == failed_or_interrupted || "$stage" == interrupted ]]; then
    echo prerequisite_failed > "$root/stage.txt"
    exit 1
  fi
  sleep 30
done
test "$(cat runs/medium-development/stage.txt)" = development_complete_freeform_review_still_pending
source .venv/bin/activate
export PYTHONPATH="$PWD/$frozen" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo generating_array_candidate > "$root/stage.txt"
CUDA_VISIBLE_DEVICES=0,1 python "$frozen/scripts/serialization_trial.py"
echo evaluating > "$root/stage.txt"
for mode in dictionary array; do
  python "$frozen/scripts/evaluate_wtq_official.py" "$root/samples.jsonl" "$root/$mode-predictions.jsonl" "$root/wtq-$mode" --evaluator data/raw/wikitq-official/WikiTableQuestions-master/evaluator.py --tagged data/raw/wikitq-official/WikiTableQuestions-master/tagged/data
done
python "$frozen/scripts/analyze_runs.py" "$root/samples.jsonl" "$root/dictionary-predictions.jsonl" "$root/analysis" --predictions-b "$root/array-predictions.jsonl" --official-a "$root/wtq-dictionary/scores.json" --official-b "$root/wtq-array/scores.json"
echo complete_development_only > "$root/stage.txt"
