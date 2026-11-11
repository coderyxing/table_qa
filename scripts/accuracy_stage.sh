#!/usr/bin/env bash
# A/B isolate context limit, B/C isolate repair count, C/D isolate failure fallback.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
root=runs/accuracy-development-v1
mkdir -p "$root"
exec 9>"$root/stage.lock"
flock -n 9 || exit 1
children=()
cleanup(){ echo failed_or_interrupted > "$root/stage.txt"; for child in "${children[@]}";do kill -TERM "$child" 2>/dev/null || true;done; }
trap cleanup ERR
trap 'cleanup;exit 143' TERM INT
test "$(cat runs/optimized-student-control/stage.txt)" = complete_development_only
frozen="$root/frozen-code"
if [[ ! -d "$frozen" ]];then
  mkdir -p "$frozen/tableqa" "$frozen/scripts"
  cp tableqa/*.py "$frozen/tableqa/";cp scripts/*.py "$frozen/scripts/";touch "$frozen/SNAPSHOT_READY"
fi
test -f "$frozen/SNAPSHOT_READY"
source .venv/bin/activate
export PYTHONPATH="$PWD/$frozen" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo preparing > "$root/stage.txt"
if [[ ! -f "$root/protocol.json" ]];then python "$frozen/scripts/prepare_accuracy_stage.py";fi
echo evaluating_accuracy_controls > "$root/stage.txt"
CUDA_VISIBLE_DEVICES=1 python "$frozen/scripts/accuracy_condition.py" A > runs/setup/accuracy-A.log 2>&1 & pid_a=$!;children+=("$pid_a")
CUDA_VISIBLE_DEVICES=0 python "$frozen/scripts/accuracy_condition.py" C > runs/setup/accuracy-C-D.log 2>&1 & pid_c=$!;children+=("$pid_c")
wait "$pid_a"
CUDA_VISIBLE_DEVICES=1 python "$frozen/scripts/accuracy_condition.py" B > runs/setup/accuracy-B.log 2>&1 & pid_b=$!;children+=("$pid_b")
wait "$pid_b";wait "$pid_c"
echo evaluating_scores > "$root/stage.txt"
for mode in A B C D;do
  python "$frozen/scripts/evaluate_wtq_official.py" "$root/samples.jsonl" "$root/$mode-predictions.jsonl" "$root/wtq-$mode" --evaluator data/raw/wikitq-official/WikiTableQuestions-master/evaluator.py --tagged data/raw/wikitq-official/WikiTableQuestions-master/tagged/data
done
for pair in A:B B:C C:D A:D;do
  a=${pair%:*};b=${pair#*:}
  python "$frozen/scripts/analyze_runs.py" "$root/samples.jsonl" "$root/$a-predictions.jsonl" "$root/comparison-$a-vs-$b" --predictions-b "$root/$b-predictions.jsonl" --official-a "$root/wtq-$a/scores.json" --official-b "$root/wtq-$b/scores.json"
done
echo complete_validation_tuning_confirmation_pending > "$root/stage.txt"
