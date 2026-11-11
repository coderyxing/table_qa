#!/usr/bin/env bash
# 脚本用途：继续已授权的依赖准备流程，使用完成标记与有限等待管理阶段。
# 执行环境：Bash；先核对正文中的项目路径、虚拟环境、GPU 编号和阶段依赖。
# 阅读提示：文件中的锁、进度及完成标记用于协调阶段；保留日志便于复核失败位置。
# Continue the authorized setup without requiring the SSH client to remain open.
# 启用脚本列出的严格模式；未处理的命令失败由阶段错误处理接管。
set -euo pipefail
# 进入协议指定的工作目录，后续相对路径均以这里为基准。
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# 在有限次数内轮询前置任务；退出条件和等待间隔由循环正文定义。
for attempt in $(seq 1 240); do
  if grep -q '^OK$' runs/setup/tests.log 2>/dev/null && [[ -f models/Qwen2.5-3B-Instruct/DOWNLOAD_COMPLETE ]]; then
    # 显式设置本阶段的环境变量，固定代码路径、缓存或设备可见性。
    export CUDA_VISIBLE_DEVICES=0
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    .venv/bin/python scripts/gpu_smoke.py
    exit 0
  fi
  printf 'Waiting for environment tests and verified 3B weights (%s/240)\n' "$attempt"
  # 在本次轮询之间留出间隔，降低持续检查对系统的干扰。
  sleep 30
done
echo 'Setup wait exceeded two hours; inspect bootstrap and download logs.' >&2
exit 1
