#!/usr/bin/env bash
# 脚本用途：等待容量和准确率开发结果，再启动独立确认阶段。
# 执行环境：Bash；先核对正文中的项目路径、虚拟环境、GPU 编号和阶段依赖。
# 阅读提示：文件中的锁、进度及完成标记用于协调阶段；保留日志便于复核失败位置。
# 启用脚本列出的严格模式；未处理的命令失败由阶段错误处理接管。
set -euo pipefail
# 进入协议指定的工作目录，后续相对路径均以这里为基准。
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# 固定本阶段的工作位置，便于把数据、源码与产物关联到同一协议。
root=runs/accuracy-confirmation
# 准备当前阶段使用的目录，已存在的目录按命令参数处理。
mkdir -p "$root"
# 打开实验锁对应的文件描述符，供 flock 协调重复启动或资源占用。
exec 9>"$root/stage.lock"
# 按当前命令指定的等待或非阻塞模式获取锁。
flock -n 9 || exit 1
# 登记退出或信号处理，异常时写入可检查的阶段状态。
trap 'echo failed_or_interrupted > "$root/stage.txt"' ERR
trap 'echo interrupted > "$root/stage.txt";exit 143' TERM INT
# 定位本轮冻结源码，历史产物通过这份快照解释。
frozen="$root/frozen-code"
if [[ ! -d "$frozen" ]];then
  mkdir -p "$frozen/tableqa" "$frozen/scripts"
  # 复制本阶段需要保存的文件；按命令中的来源与目标维护快照关系。
  cp tableqa/*.py "$frozen/tableqa/";cp scripts/*.py "$frozen/scripts/";touch "$frozen/SNAPSHOT_READY"
fi
# 先核验前置条件，避免在缺少配置、数据或完成标记时进入下一阶段。
test -f "$frozen/SNAPSHOT_READY"
echo waiting_for_all_tuning_conditions > "$root/stage.txt"
ready=0
# 在有限次数内轮询前置任务；退出条件和等待间隔由循环正文定义。
for attempt in $(seq 1 1440);do
  ready=1
  for dir in accuracy-development-v1 scale-7b-accuracy balanced-accuracy;do
    s=$(cat "runs/$dir/stage.txt")
    case "$s" in failed_or_interrupted|prerequisite_failed|wait_timed_out|interrupted)
      echo prerequisite_failed > "$root/stage.txt";exit 1;;esac
    if [[ "$s" != complete_validation_tuning_confirmation_pending ]];then ready=0;fi
  done
  if [[ "$ready" == 1 ]];then break;fi
  # 在本次轮询之间留出间隔，降低持续检查对系统的干扰。
  sleep 30
done
if [[ "$ready" != 1 ]];then echo wait_timed_out > "$root/stage.txt";exit 1;fi
# 加载项目虚拟环境或阶段配置，使后续命令使用预期依赖。
source .venv/bin/activate
# 显式设置本阶段的环境变量，固定代码路径、缓存或设备可见性。
export PYTHONPATH="$PWD/$frozen" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo selecting_then_confirming_once > "$root/stage.txt"
# 为当前命令显式选择可见 GPU，使资源配置可复核。
CUDA_VISIBLE_DEVICES=0 python "$frozen/scripts/confirm_accuracy.py"
echo complete_confirmation_no_retuning_on_this_cohort > "$root/stage.txt"
