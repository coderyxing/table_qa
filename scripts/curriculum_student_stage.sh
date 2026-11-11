#!/usr/bin/env bash
set -euo pipefail
stage="${1:?Prepared stage directory required}"
code="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$code"
root="$(dirname -- "$stage")"
test ! -e "$root/LAUNCHED"
touch "$root/LAUNCHED"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u "$code/scripts/train_curriculum_condition.py" "$stage" R > "$root/train-R.log" 2>&1 &
r_pid=$!
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u "$code/scripts/train_curriculum_condition.py" "$stage" M > "$root/train-M.log" 2>&1 &
m_pid=$!
r_status=0; m_status=0
wait "$r_pid" || r_status=$?
wait "$m_pid" || m_status=$?
if [[ "$r_status" != 0 || "$m_status" != 0 ]]; then
  printf 'R=%s M=%s\n' "$r_status" "$m_status" > "$root/FAILED"
  exit 1
fi
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u "$code/scripts/evaluate_curriculum_stage.py" "$stage" > "$root/evaluation.log" 2>&1
