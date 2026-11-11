#!/usr/bin/env bash
# 脚本用途：按既定顺序运行修复版联合训练，使用冻结源码和阶段记录。
# 执行环境：Bash；先核对正文中的项目路径、虚拟环境、GPU 编号和阶段依赖。
# 阅读提示：文件中的锁、进度及完成标记用于协调阶段；保留日志便于复核失败位置。
# 启用脚本列出的严格模式；未处理的命令失败由阶段错误处理接管。
set -euo pipefail
# 进入协议指定的工作目录，后续相对路径均以这里为基准。
cd /root/tableqa-research
# 固定本阶段的工作位置，便于把数据、源码与产物关联到同一协议。
root=runs/joint-core-20261107
# 显式设置本阶段的环境变量，固定代码路径、缓存或设备可见性。
export PYTHONPATH="/root/tableqa-research/$root/code"
runner="$root/code/scripts/joint_core_stage.py"
stage="$root/prepared"
# 先核验前置条件，避免在缺少配置、数据或完成标记时进入下一阶段。
test -f "$stage/protocol.json"
test ! -e "$root/QUEUE_STARTED"
date -u > "$root/QUEUE_STARTED"
# 登记退出或信号处理，异常时写入可检查的阶段状态。
trap 'echo "queue failed at line $LINENO" > "$root/QUEUE_FAILED"' ERR
(
  # 为当前命令显式选择可见 GPU，使资源配置可复核。
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u "$runner" B0 "$stage" > "$root/B0.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u "$runner" B2 "$stage" > "$root/B2.log" 2>&1
) &
first=$!
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u "$runner" B5 "$stage" > "$root/B5.log" 2>&1 &
second=$!
status_first=0;wait "$first" || status_first=$?
status_second=0;wait "$second" || status_second=$?
if [[ "$status_first" != 0 || "$status_second" != 0 ]]; then
  echo "worker exits: $status_first $status_second" > "$root/QUEUE_FAILED"
  exit 1
fi
.venv/bin/python -u "$runner" report "$stage" > "$root/report.log" 2>&1
date -u > "$root/QUEUE_COMPLETE"
