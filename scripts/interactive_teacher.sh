#!/usr/bin/env bash
# Fresh training-only comparison. This B5 interface uses the 32B TEACHER, not a student.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
root=runs/interactive-teacher
mkdir -p "$root"
exec 9>"$root/stage.lock"
flock -n 9 || exit 1
trap 'echo failed_or_interrupted > "$root/stage.txt"' ERR
trap 'echo interrupted > "$root/stage.txt"; exit 143' TERM INT
if [[ -f "$root/COMPLETE" ]];then exit 0;fi
test "$(cat runs/coverage-student-control/stage.txt)" = complete_development_only
frozen="$root/frozen-code"
if [[ ! -d "$frozen" ]];then
  mkdir -p "$frozen/tableqa" "$frozen/scripts"
  cp tableqa/*.py "$frozen/tableqa/"
  cp scripts/*.py "$frozen/scripts/"
  touch "$frozen/SNAPSHOT_READY"
fi
test -f "$frozen/SNAPSHOT_READY"
source .venv/bin/activate
export PYTHONPATH="$PWD/$frozen" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ ! -f "$root/protocol.json" ]];then python "$frozen/scripts/prepare_interactive_teacher.py";fi
for mode in whole interactive;do
  echo "$mode" > "$root/stage.txt"
  CUDA_VISIBLE_DEVICES=0,1 python -m tableqa predict "$root/samples.jsonl" "$root/$mode.json" "$root/$mode-predictions.jsonl" --resume
done
echo evaluating > "$root/stage.txt"
for mode in whole interactive;do
  python "$frozen/scripts/evaluate_wtq_official.py" "$root/samples.jsonl" "$root/$mode-predictions.jsonl" "$root/wtq-$mode" --evaluator data/raw/wikitq-official/WikiTableQuestions-master/evaluator.py --tagged data/raw/wikitq-official/WikiTableQuestions-master/tagged/data
  python -m tableqa audit-trajectories "$root/samples.jsonl" "$root/$mode-predictions.jsonl" "$root/quality-$mode.jsonl"
done
python "$frozen/scripts/analyze_runs.py" "$root/samples.jsonl" "$root/whole-predictions.jsonl" "$root/comparison" --predictions-b "$root/interactive-predictions.jsonl" --official-a "$root/wtq-whole/scores.json" --official-b "$root/wtq-interactive/scores.json"
for mode in whole interactive;do
  if [[ ! -d "$root/review-$mode" ]];then
    python "$frozen/scripts/human_review.py" export "$root/samples.jsonl" "$root/$mode-predictions.jsonl" "$root/review-$mode" --purpose train_filter
  fi
done
echo complete_development_only > "$root/stage.txt"
touch "$root/COMPLETE"
