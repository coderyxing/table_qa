#!/usr/bin/env bash
# 脚本用途：使用准备好的阶段目录训练课程组并评测冻结队列。
# 执行环境：Bash；先核对正文中的项目路径、虚拟环境、GPU 编号和阶段依赖。
# 阅读提示：文件中的锁、进度及完成标记用于协调阶段；保留日志便于复核失败位置。
# 启用脚本列出的严格模式；未处理的命令失败由阶段错误处理接管。
set -euo pipefail
# 固定本阶段的工作位置，便于把数据、源码与产物关联到同一协议。
stage="${1:?Prepared stage directory required}"
code="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# 显式设置本阶段的环境变量，固定代码路径、缓存或设备可见性。
export PYTHONPATH="$code"
root="$(dirname -- "$stage")"
# 先核验前置条件，避免在缺少配置、数据或完成标记时进入下一阶段。
test ! -e "$root/LAUNCHED"
touch "$root/LAUNCHED"
# 为当前命令显式选择可见 GPU，使资源配置可复核。
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u "$code/scripts/train_curriculum_condition.py" "$stage" R > "$root/train-R.log" 2>&1 &
r_pid=$!
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u "$code/scripts/train_curriculum_condition.py" "$stage" M > "$root/train-M.log" 2>&1 &
m_pid=$!
r_status=0; m_status=0
# 等待指定后台任务结束，成功后才继续依赖它的阶段。
wait "$r_pid" || r_status=$?
wait "$m_pid" || m_status=$?
if [[ "$r_status" != 0 || "$m_status" != 0 ]]; then
  printf 'R=%s M=%s\n' "$r_status" "$m_status" > "$root/FAILED"
  exit 1
fi
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u "$code/scripts/evaluate_curriculum_stage.py" "$stage" > "$root/evaluation.log" 2>&1
