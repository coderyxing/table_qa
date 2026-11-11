#!/usr/bin/env bash
# 脚本用途：只读检查 GPU、互联、内存、工作盘和 Python/CUDA 依赖。
# 执行环境：Bash；先核对正文中的项目路径、虚拟环境、GPU 编号和阶段依赖。
# 阅读提示：文件中的锁、进度及完成标记用于协调阶段；保留日志便于复核失败位置。
# 启用脚本列出的严格模式；未处理的命令失败由阶段错误处理接管。
set -euo pipefail
# Read-only diagnostics: no installation, driver changes or model downloads.
nvidia-smi
nvidia-smi topo -m
free -h
df -h .
# 调用当前阶段的 Python 入口，参数中的文件明确给出本次输入与输出。
python -c 'import sys, torch; print(sys.version); print("torch",torch.__version__,"runtime",torch.version.cuda); print("devices",torch.cuda.device_count()); print("bf16",torch.cuda.is_bf16_supported())'
python -m pip freeze
