#!/usr/bin/env bash
# 脚本用途：在新训练队列上比较教师交互接口，固定教师容量与执行预算。
# 执行环境：Bash；先核对正文中的项目路径、虚拟环境、GPU 编号和阶段依赖。
# 阅读提示：文件中的锁、进度及完成标记用于协调阶段；保留日志便于复核失败位置。
# Fresh training-only comparison. This B5 interface uses the 32B TEACHER, not a student.
# 启用脚本列出的严格模式；未处理的命令失败由阶段错误处理接管。
set -euo pipefail
# 进入协议指定的工作目录，后续相对路径均以这里为基准。
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# 固定本阶段的工作位置，便于把数据、源码与产物关联到同一协议。
root=runs/interactive-teacher
# 准备当前阶段使用的目录，已存在的目录按命令参数处理。
mkdir -p "$root"
# 打开实验锁对应的文件描述符，供 flock 协调重复启动或资源占用。
exec 9>"$root/stage.lock"
# 按当前命令指定的等待或非阻塞模式获取锁。
flock -n 9 || exit 1
# 登记退出或信号处理，异常时写入可检查的阶段状态。
trap 'echo failed_or_interrupted > "$root/stage.txt"' ERR
trap 'echo interrupted > "$root/stage.txt"; exit 143' TERM INT
if [[ -f "$root/COMPLETE" ]];then exit 0;fi
# 先核验前置条件，避免在缺少配置、数据或完成标记时进入下一阶段。
test "$(cat runs/coverage-student-control/stage.txt)" = complete_development_only
# 定位本轮冻结源码，历史产物通过这份快照解释。
frozen="$root/frozen-code"
if [[ ! -d "$frozen" ]];then
  mkdir -p "$frozen/tableqa" "$frozen/scripts"
  # 复制本阶段需要保存的文件；按命令中的来源与目标维护快照关系。
  cp tableqa/*.py "$frozen/tableqa/"
  cp scripts/*.py "$frozen/scripts/"
  touch "$frozen/SNAPSHOT_READY"
fi
test -f "$frozen/SNAPSHOT_READY"
# 加载项目虚拟环境或阶段配置，使后续命令使用预期依赖。
source .venv/bin/activate
# 显式设置本阶段的环境变量，固定代码路径、缓存或设备可见性。
export PYTHONPATH="$PWD/$frozen" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ ! -f "$root/protocol.json" ]];then python "$frozen/scripts/prepare_interactive_teacher.py";fi
# 遍历协议中显式列出的实验组或随机种子，逐项保存结果。
for mode in whole interactive;do
  echo "$mode" > "$root/stage.txt"
  # 为当前命令显式选择可见 GPU，使资源配置可复核。
  CUDA_VISIBLE_DEVICES=0,1 python -m tableqa predict "$root/samples.jsonl" "$root/$mode.json" "$root/$mode-predictions.jsonl" --resume
done
echo evaluating > "$root/stage.txt"
for mode in whole interactive;do
  # 调用当前阶段的 Python 入口，参数中的文件明确给出本次输入与输出。
  python "$frozen/scripts/evaluate_wtq_official.py" "$root/samples.jsonl" "$root/$mode-predictions.jsonl" "$root/wtq-$mode" --evaluator data/raw/wikitq-official/WikiTableQuestions-master/evaluator.py --tagged data/raw/wikitq-official/WikiTableQuestions-master/tagged/data
  python -m tableqa audit-trajectories "$root/samples.jsonl" "$root/$mode-predictions.jsonl" "$root/quality-$mode.jsonl"
done
python "$frozen/scripts/analyze_runs.py" "$root/samples.jsonl" "$root/whole-predictions.jsonl" "$root/comparison" --predictions-b "$root/interactive-predictions.jsonl" --official-a "$root/wtq-whole/scores.json" --official-b "$root/wtq-interactive/scores.json"
for mode in whole interactive;do
  if [[ ! -d "$root/review-$mode" ]];then
    python "$frozen/scripts/human_review.py" export "$root/samples.jsonl" "$root/$mode-predictions.jsonl" "$root/review-$mode" --purpose train_filter
  fi
done
echo complete_development_only > "$root/stage.txt"
touch "$root/COMPLETE"
