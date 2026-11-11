#!/usr/bin/env bash
set -Eeuo pipefail
umask 027
project=/memory/tableqa
run="$project/runs/20261110-pilot"
out="$run/chapter3-quote-v2"
py="$run/.venv/bin/python"
export TMPDIR="$project/tmp"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1

mkdir -p "$out"
status() {
    printf '%s %s\n' "$(date -Is)" "$1" | tee -a "$out/progress.log"
}
trap 'status "failed at line $LINENO (exit $?)"' ERR

cd "$run/source-v3"
status validating_source
"$py" -B -m unittest discover -s tests -p test_thesis_experiments.py -q
status queued_behind_chapter4
exec 8>"$run/pilot.lock"
flock 8
while ! nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits |
    awk '{if ($1 < 39000) busy=1} END {exit (NR != 2 || busy)}'; do
    status waiting_for_two_free_A40s
    sleep 60
done

for group in fixed dynamic; do
    cp -p "$run/chapter3/$group.jsonl" "$run/chapter3/$group.jsonl.meta.json" "$out/"
done
status extracting_verbatim_quotes
"$py" -u -m tableqa.thesis declare3 "$run/chapter3-dev60.jsonl" "$run/teacher.json" "$out/declarations.jsonl"
test "$(wc -l <"$out/declarations.jsonl")" -eq 60
"$py" - "$out/declarations.jsonl" <<'PY'
from pathlib import Path
import sys
from tableqa.data import read_jsonl

rows = read_jsonl(Path(sys.argv[1]))
bound = sum(bool(row["conditions"]) for row in rows)
resolved = sum(bool(row["conditions"]) and not row["unresolved"] for row in rows)
print(f"bound_declarations={bound}/60 fully_resolved={resolved}/60", flush=True)
if not bound or not resolved:
    raise SystemExit("The quote-binding development gate did not pass")
PY
for group in conditions row_domains no_repair; do
    status "chapter3_$group"
    "$py" -u -m tableqa.thesis predict3 "$group" "$run/chapter3-dev60.jsonl" "$run/teacher.json" \
        "$out/$group.jsonl" --declarations "$out/declarations.jsonl"
    test "$(wc -l <"$out/$group.jsonl")" -eq 60
done
status chapter3_quote_v2_pending_human_review
