#!/usr/bin/env bash
# 脚本用途：准备并检查双卡教师权重，然后执行教师试跑。
# 执行环境：Bash；先核对正文中的项目路径、虚拟环境、GPU 编号和阶段依赖。
# 阅读提示：文件中的锁、进度及完成标记用于协调阶段；保留日志便于复核失败位置。
# 启用脚本列出的严格模式；未处理的命令失败由阶段错误处理接管。
set -euo pipefail
# 进入协议指定的工作目录，后续相对路径均以这里为基准。
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# 加载项目虚拟环境或阶段配置，使后续命令使用预期依赖。
source .venv/bin/activate
# 调用当前阶段的 Python 入口，参数中的文件明确给出本次输入与输出。
python scripts/download_model.py Qwen/Qwen2.5-32B-Instruct models/Qwen2.5-32B-Instruct --metadata-only
ready=0
# 在有限次数内轮询前置任务；退出条件和等待间隔由循环正文定义。
for attempt in $(seq 1 8); do
  if python scripts/download_ranges.py Qwen/Qwen2.5-32B-Instruct models/Qwen2.5-32B-Instruct --workers 8; then
    ready=1
    break
  fi
  printf 'Download pass %s failed; preserving checked files and retrying.\n' "$attempt"
  # 在本次轮询之间留出间隔，降低持续检查对系统的干扰。
  sleep 15
done
# 先核验前置条件，避免在缺少配置、数据或完成标记时进入下一阶段。
test "$ready" -eq 1
# 显式设置本阶段的环境变量，固定代码路径、缓存或设备可见性。
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python scripts/teacher_pilot.py
