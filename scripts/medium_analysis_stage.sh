#!/usr/bin/env bash
# CPU-only result stage: preserve the main batch and explicitly regrade its outputs.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
root=runs/medium-analysis
mkdir -p "$root"
exec 9>"$root/stage.lock"
flock -n 9 || exit 1
trap 'echo failed_or_interrupted > "$root/stage.txt"' ERR
trap 'echo interrupted > "$root/stage.txt"; exit 143' TERM INT
if [[ -f "$root/COMPLETE" ]]; then exit 0; fi
frozen="$root/frozen-code"
if [[ ! -d "$frozen" ]]; then
  mkdir -p "$frozen/tableqa" "$frozen/scripts"
  cp tableqa/*.py "$frozen/tableqa/"
  cp scripts/*.py "$frozen/scripts/"
  touch "$frozen/SNAPSHOT_READY"
fi
test -f "$frozen/SNAPSHOT_READY"
echo waiting_for_complete_teacher_predictions > "$root/stage.txt"
for attempt in $(seq 1 1440); do
  stage=$(cat runs/medium-development/stage.txt)
  case "$stage" in
    quality_audit|freeform_review_pending|students_B2_B5|development_complete_freeform_review_still_pending) break ;;
    failed_or_interrupted|interrupted)
      echo prerequisite_failed > "$root/stage.txt"
      exit 1 ;;
  esac
  sleep 30
done
case "$(cat runs/medium-development/stage.txt)" in
  quality_audit|freeform_review_pending|students_B2_B5|development_complete_freeform_review_still_pending) ;;
  *) echo wait_timed_out > "$root/stage.txt"; exit 1 ;;
esac
source .venv/bin/activate
export PYTHONPATH="$PWD/$frozen" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo regrading_and_analyzing > "$root/stage.txt"
for mode in A4 A0; do
  python "$frozen/scripts/evaluate_wtq_official.py" runs/medium-development/samples.jsonl "runs/medium-development/$mode-predictions.jsonl" "$root/wtq-$mode" --evaluator data/raw/wikitq-official/WikiTableQuestions-master/evaluator.py --tagged data/raw/wikitq-official/WikiTableQuestions-master/tagged/data
done
python "$frozen/scripts/analyze_runs.py" runs/medium-development/samples.jsonl runs/medium-development/A0-predictions.jsonl "$root/comparison" --predictions-b runs/medium-development/A4-predictions.jsonl --official-a "$root/wtq-A0/scores.json" --official-b "$root/wtq-A4/scores.json"
if [[ ! -d "$root/freeform-review" ]]; then
  python "$frozen/scripts/human_review.py" export runs/medium-development/samples.jsonl runs/medium-development/A4-predictions.jsonl "$root/freeform-review" --purpose train_filter
fi
echo complete_teacher_analysis_freeform_review_pending > "$root/stage.txt"
touch "$root/COMPLETE"
