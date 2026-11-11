#!/usr/bin/env bash
# Bounded in-task artifact completion; no model calls or test-set execution.
set -euo pipefail
root="${1:?Experiment root required}"
for attempt in $(seq 1 180); do
  if [[ -f "$root/COMPLETE" ]]; then break; fi
  if [[ -f "$root/FAILED" ]] || { [[ -f "$root/evaluation.log" ]] && grep -q '^Traceback' "$root/evaluation.log"; }; then
    echo EXPERIMENT_FAILED_BEFORE_ARTIFACT_AUDIT
    exit 1
  fi
  sleep 30
done
test -f "$root/COMPLETE"
PYTHONPATH="$root/execution-code" .venv/bin/python "$root/analysis-code-v2/scripts/audit_curriculum_completion.py" "$root"
/root/anaconda3/bin/python "$root/analysis-code-v2/scripts/plot_curriculum_stage.py" "$root"
tar -czf "$root/review-artifacts.tar.gz" -C "$root" prepared evaluation figures completion-audit.json COMPLETE \
  training-R/run.json training-R/statistics.json training-R/adapter/training_history.json training-R/adapter/training_exposure.json \
  training-M/run.json training-M/statistics.json training-M/adapter/training_history.json training-M/adapter/training_exposure.json
echo CURRICULUM_ARTIFACT_AUDIT_COMPLETE
