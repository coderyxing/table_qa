#!/usr/bin/env bash
# 脚本用途：在磁盘与哈希检查后下载 7B 容量对照模型。
# 执行环境：Bash；先核对正文中的项目路径、虚拟环境、GPU 编号和阶段依赖。
# 阅读提示：文件中的锁、进度及完成标记用于协调阶段；保留日志便于复核失败位置。
# Public planned scale control; CPU/network only, with existing hash/disk guards.
# 启用脚本列出的严格模式；未处理的命令失败由阶段错误处理接管。
set -euo pipefail
# 进入协议指定的工作目录，后续相对路径均以这里为基准。
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# 准备当前阶段使用的目录，已存在的目录按命令参数处理。
mkdir -p runs/model-7b
# 打开实验锁对应的文件描述符，供 flock 协调重复启动或资源占用。
exec 9>runs/model-7b/download.lock
# 按当前命令指定的等待或非阻塞模式获取锁。
flock -n 9 || exit 1
# 登记退出或信号处理，异常时写入可检查的阶段状态。
trap 'echo failed_or_interrupted > runs/model-7b/stage.txt' ERR
echo downloading > runs/model-7b/stage.txt
ready=0
# 在有限次数内轮询前置任务；退出条件和等待间隔由循环正文定义。
for attempt in 1 2 3;do
  if .venv/bin/python scripts/download_model.py Qwen/Qwen2.5-7B-Instruct models/Qwen2.5-7B-Instruct --metadata-only;then ready=1;break;fi
  # 在本次轮询之间留出间隔，降低持续检查对系统的干扰。
  sleep 10
done
# 先核验前置条件，避免在缺少配置、数据或完成标记时进入下一阶段。
test "$ready" = 1
.venv/bin/python scripts/download_ranges.py Qwen/Qwen2.5-7B-Instruct models/Qwen2.5-7B-Instruct --workers 8
test -f models/Qwen2.5-7B-Instruct/DOWNLOAD_COMPLETE
echo verified_complete > runs/model-7b/stage.txt
