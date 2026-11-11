#!/usr/bin/env bash
# 脚本用途：有界等待各队列完成，收集冻结产物并核对完成标记。
# 执行环境：Bash；先核对正文中的项目路径、虚拟环境、GPU 编号和阶段依赖。
# 阅读提示：文件中的锁、进度及完成标记用于协调阶段；保留日志便于复核失败位置。
# Bounded artifact collection. The models are managed by separate fixed queues.
# 启用脚本列出的严格模式；未处理的命令失败由阶段错误处理接管。
set -euo pipefail
# 进入协议指定的工作目录，后续相对路径均以这里为基准。
cd /root/tableqa-research
student=runs/joint-core-20261107
teacher=runs/teacher-ablation-20261107
# 显式设置本阶段的环境变量，固定代码路径、缓存或设备可见性。
export PYTHONPATH="/root/tableqa-research/$teacher/code"
# 登记退出或信号处理，异常时写入可检查的阶段状态。
trap 'echo "artifact collection failed at line $LINENO" > "$student/ARTIFACT_FAILED"' ERR
ready=0
# 在有限次数内轮询前置任务；退出条件和等待间隔由循环正文定义。
for attempt in $(seq 1 480); do
  if [[ -f "$student/QUEUE_FAILED" ]]; then exit 1;fi
  if [[ -f "$student/QUEUE_COMPLETE" ]]; then ready=1;break;fi
  # 在本次轮询之间留出间隔，降低持续检查对系统的干扰。
  sleep 30
done
# 先核验前置条件，避免在缺少配置、数据或完成标记时进入下一阶段。
test "$ready" = 1
.venv/bin/python runs/finalize_joint_core.py "$student/prepared" > "$student/artifact.log" 2>&1
tar --exclude='*.safetensors' --exclude='*/checkpoint-*' --exclude='*/__pycache__' \
  -czf "$student/review-artifacts.tar.gz" -C "$student" prepared code queue.sh queue.log QUEUE_STARTED QUEUE_COMPLETE B0.log B2.log B5.log report.log artifact.log
date -u > "$student/ARTIFACT_COMPLETE"
ready=0
for attempt in $(seq 1 720); do
  if [[ -f "$teacher/QUEUE_FAILED" ]]; then exit 1;fi
  if [[ -f "$teacher/QUEUE_COMPLETE" ]]; then ready=1;break;fi
  sleep 30
done
test "$ready" = 1
tar --exclude='*/__pycache__' -czf "$teacher/review-artifacts.tar.gz" -C "$teacher" prepared code queue.log teacher.log QUEUE_STARTED QUEUE_COMPLETE
date -u > "$teacher/ARTIFACT_COMPLETE"
