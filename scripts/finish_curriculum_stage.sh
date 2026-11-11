#!/usr/bin/env bash
# 脚本用途：有界等待课程任务，补齐完成审计和产物收集。
# 执行环境：Bash；先核对正文中的项目路径、虚拟环境、GPU 编号和阶段依赖。
# 阅读提示：文件中的锁、进度及完成标记用于协调阶段；保留日志便于复核失败位置。
# Bounded in-task artifact completion; no model calls or test-set execution.
# 启用脚本列出的严格模式；未处理的命令失败由阶段错误处理接管。
set -euo pipefail
# 固定本阶段的工作位置，便于把数据、源码与产物关联到同一协议。
root="${1:?Experiment root required}"
# 在有限次数内轮询前置任务；退出条件和等待间隔由循环正文定义。
for attempt in $(seq 1 180); do
  if [[ -f "$root/COMPLETE" ]]; then break; fi
  if [[ -f "$root/FAILED" ]] || { [[ -f "$root/evaluation.log" ]] && grep -q '^Traceback' "$root/evaluation.log"; }; then
    echo EXPERIMENT_FAILED_BEFORE_ARTIFACT_AUDIT
    exit 1
  fi
  # 在本次轮询之间留出间隔，降低持续检查对系统的干扰。
  sleep 30
done
# 先核验前置条件，避免在缺少配置、数据或完成标记时进入下一阶段。
test -f "$root/COMPLETE"
PYTHONPATH="$root/execution-code" .venv/bin/python "$root/analysis-code-v2/scripts/audit_curriculum_completion.py" "$root"
/root/anaconda3/bin/python "$root/analysis-code-v2/scripts/plot_curriculum_stage.py" "$root"
tar -czf "$root/review-artifacts.tar.gz" -C "$root" prepared evaluation figures completion-audit.json COMPLETE \
  training-R/run.json training-R/statistics.json training-R/adapter/training_history.json training-R/adapter/training_exposure.json \
  training-M/run.json training-M/statistics.json training-M/adapter/training_history.json training-M/adapter/training_exposure.json
echo CURRICULUM_ARTIFACT_AUDIT_COMPLETE
