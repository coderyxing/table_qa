#!/usr/bin/env bash
# 脚本用途：在独立目录重新判分完整开发输出并生成分析与人工审核材料。
# 执行环境：Bash；先核对正文中的项目路径、虚拟环境、GPU 编号和阶段依赖。
# 阅读提示：文件中的锁、进度及完成标记用于协调阶段；保留日志便于复核失败位置。
# CPU-only result stage: preserve the main batch and explicitly regrade its outputs.
# 启用脚本列出的严格模式；未处理的命令失败由阶段错误处理接管。
set -euo pipefail
# 进入协议指定的工作目录，后续相对路径均以这里为基准。
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# 固定本阶段的工作位置，便于把数据、源码与产物关联到同一协议。
root=runs/medium-analysis
# 准备当前阶段使用的目录，已存在的目录按命令参数处理。
mkdir -p "$root"
# 打开实验锁对应的文件描述符，供 flock 协调重复启动或资源占用。
exec 9>"$root/stage.lock"
# 按当前命令指定的等待或非阻塞模式获取锁。
flock -n 9 || exit 1
# 登记退出或信号处理，异常时写入可检查的阶段状态。
trap 'echo failed_or_interrupted > "$root/stage.txt"' ERR
trap 'echo interrupted > "$root/stage.txt"; exit 143' TERM INT
if [[ -f "$root/COMPLETE" ]]; then exit 0; fi
# 定位本轮冻结源码，历史产物通过这份快照解释。
frozen="$root/frozen-code"
if [[ ! -d "$frozen" ]]; then
  mkdir -p "$frozen/tableqa" "$frozen/scripts"
  # 复制本阶段需要保存的文件；按命令中的来源与目标维护快照关系。
  cp tableqa/*.py "$frozen/tableqa/"
  cp scripts/*.py "$frozen/scripts/"
  touch "$frozen/SNAPSHOT_READY"
fi
# 先核验前置条件，避免在缺少配置、数据或完成标记时进入下一阶段。
test -f "$frozen/SNAPSHOT_READY"
echo waiting_for_complete_teacher_predictions > "$root/stage.txt"
# 在有限次数内轮询前置任务；退出条件和等待间隔由循环正文定义。
for attempt in $(seq 1 1440); do
  stage=$(cat runs/medium-development/stage.txt)
  case "$stage" in
    quality_audit|freeform_review_pending|students_B2_B5|development_complete_freeform_review_still_pending) break ;;
    failed_or_interrupted|interrupted)
      echo prerequisite_failed > "$root/stage.txt"
      exit 1 ;;
  esac
  # 在本次轮询之间留出间隔，降低持续检查对系统的干扰。
  sleep 30
done
case "$(cat runs/medium-development/stage.txt)" in
  quality_audit|freeform_review_pending|students_B2_B5|development_complete_freeform_review_still_pending) ;;
  *) echo wait_timed_out > "$root/stage.txt"; exit 1 ;;
esac
# 加载项目虚拟环境或阶段配置，使后续命令使用预期依赖。
source .venv/bin/activate
# 显式设置本阶段的环境变量，固定代码路径、缓存或设备可见性。
export PYTHONPATH="$PWD/$frozen" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo regrading_and_analyzing > "$root/stage.txt"
# 遍历协议中显式列出的实验组或随机种子，逐项保存结果。
for mode in A4 A0; do
  # 调用当前阶段的 Python 入口，参数中的文件明确给出本次输入与输出。
  python "$frozen/scripts/evaluate_wtq_official.py" runs/medium-development/samples.jsonl "runs/medium-development/$mode-predictions.jsonl" "$root/wtq-$mode" --evaluator data/raw/wikitq-official/WikiTableQuestions-master/evaluator.py --tagged data/raw/wikitq-official/WikiTableQuestions-master/tagged/data
done
python "$frozen/scripts/analyze_runs.py" runs/medium-development/samples.jsonl runs/medium-development/A0-predictions.jsonl "$root/comparison" --predictions-b runs/medium-development/A4-predictions.jsonl --official-a "$root/wtq-A0/scores.json" --official-b "$root/wtq-A4/scores.json"
if [[ ! -d "$root/freeform-review" ]]; then
  python "$frozen/scripts/human_review.py" export runs/medium-development/samples.jsonl runs/medium-development/A4-predictions.jsonl "$root/freeform-review" --purpose train_filter
fi
echo complete_teacher_analysis_freeform_review_pending > "$root/stage.txt"
touch "$root/COMPLETE"
