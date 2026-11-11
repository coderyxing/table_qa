#!/usr/bin/env bash
# Bounded development workflow, not final multi-seed thesis experiments.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -p runs/medium-development
exec 9>runs/medium-development/stage.lock
flock -n 9 || { echo 'Medium stage already running'; exit 1; }
source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
status_file=runs/medium-development/stage.txt
trap 'echo failed_or_interrupted > "$status_file"' ERR
trap 'echo interrupted > "$status_file"; exit 143' TERM INT
frozen=runs/medium-development/frozen-code
if [[ ! -d "$frozen" ]]; then
  mkdir -p "$frozen/tableqa" "$frozen/scripts"
  cp tableqa/*.py "$frozen/tableqa/"
  cp scripts/*.py scripts/*.sh "$frozen/scripts/"
  touch "$frozen/SNAPSHOT_READY"
fi
test -f "$frozen/SNAPSHOT_READY"
export PYTHONPATH="$PWD/$frozen"
if [[ ! -f runs/medium-development/protocol.json ]]; then python "$frozen/scripts/prepare_medium_stage.py"; fi
echo teacher_A4 > "$status_file"
CUDA_VISIBLE_DEVICES=0,1 python -m tableqa predict runs/medium-development/samples.jsonl runs/medium-development/A4.json runs/medium-development/A4-predictions.jsonl --resume
echo direct_A0 > "$status_file"
CUDA_VISIBLE_DEVICES=0,1 python -m tableqa predict runs/medium-development/samples.jsonl runs/medium-development/A0.json runs/medium-development/A0-predictions.jsonl --resume
echo quality_audit > "$status_file"
python -m tableqa audit-trajectories runs/medium-development/samples.jsonl runs/medium-development/A4-predictions.jsonl runs/medium-development/quality.jsonl
python -m tableqa evaluate runs/medium-development/samples.jsonl runs/medium-development/A4-predictions.jsonl runs/medium-development/teacher-evaluation.json
python -m tableqa evaluate runs/medium-development/samples.jsonl runs/medium-development/A0-predictions.jsonl runs/medium-development/direct-evaluation.json
for mode in A4 A0; do
  python "$frozen/scripts/evaluate_wtq_official.py" runs/medium-development/samples.jsonl "runs/medium-development/$mode-predictions.jsonl" "runs/medium-development/wtq-$mode" --evaluator data/raw/wikitq-official/WikiTableQuestions-master/evaluator.py --tagged data/raw/wikitq-official/WikiTableQuestions-master/tagged/data
done
echo freeform_review_pending > "$status_file"
# Convert to the shared pilot source contract while preserving raw run files.
mkdir -p runs/medium-development/trajectory-source
cp runs/medium-development/samples.jsonl runs/medium-development/trajectory-source/samples.jsonl
cp runs/medium-development/A4-predictions.jsonl runs/medium-development/trajectory-source/predictions.jsonl
cp runs/medium-development/teacher-evaluation.json runs/medium-development/trajectory-source/report.json
python "$frozen/scripts/prepare_freeform_review.py" --source runs/medium-development/trajectory-source
echo students_B2_B5 > "$status_file"
CUDA_VISIBLE_DEVICES=0 python "$frozen/scripts/paired_distillation_pilot.py" --source runs/medium-development/trajectory-source --output runs/medium-development/students --epochs 3 --rank 32 --eval-per-dataset 20
echo development_complete_freeform_review_still_pending > "$status_file"
