#!/usr/bin/env bash
# 脚本用途：在冻结源码下串行执行同容量教师的依赖、证据和修复消融。
# 执行环境：Bash；先核对正文中的项目路径、虚拟环境、GPU 编号和阶段依赖。
# 阅读提示：文件中的锁、进度及完成标记用于协调阶段；保留日志便于复核失败位置。
# 启用脚本列出的严格模式；未处理的命令失败由阶段错误处理接管。
set -euo pipefail
# 进入协议指定的工作目录，后续相对路径均以这里为基准。
cd /root/tableqa-research
# 固定本阶段的工作位置，便于把数据、源码与产物关联到同一协议。
root=runs/teacher-ablation-20261107
# 显式设置本阶段的环境变量，固定代码路径、缓存或设备可见性。
export PYTHONPATH="/root/tableqa-research/$root/code"
# 先核验前置条件，避免在缺少配置、数据或完成标记时进入下一阶段。
test ! -e "$root/QUEUE_STARTED"
date -u > "$root/QUEUE_STARTED"
# 登记退出或信号处理，异常时写入可检查的阶段状态。
trap 'echo "teacher queue failed at line $LINENO" > "$root/QUEUE_FAILED"' ERR
ready=0
# 在有限次数内轮询前置任务；退出条件和等待间隔由循环正文定义。
for attempt in $(seq 1 480); do
  if [[ -f runs/joint-core-20261107/QUEUE_FAILED ]]; then
    echo STUDENT_QUEUE_FAILED > "$root/QUEUE_FAILED"
    exit 1
  fi
  if [[ -f runs/joint-core-20261107/QUEUE_COMPLETE ]]; then ready=1;break;fi
  # 在本次轮询之间留出间隔，降低持续检查对系统的干扰。
  sleep 30
done
if [[ "$ready" != 1 ]]; then echo STUDENT_WAIT_TIMEOUT > "$root/QUEUE_FAILED";exit 1;fi
# 为当前命令显式选择可见 GPU，使资源配置可复核。
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -u "$root/code/scripts/teacher_module_ablation.py" run "$root/prepared" > "$root/teacher.log" 2>&1
date -u > "$root/QUEUE_COMPLETE"
