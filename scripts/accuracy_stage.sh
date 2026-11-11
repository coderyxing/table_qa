#!/usr/bin/env bash
# 脚本用途：串行执行上下文上限、修复次数和失败回退对照，用阶段文件记录进度。
# 执行环境：Bash；先核对正文中的项目路径、虚拟环境、GPU 编号和阶段依赖。
# 阅读提示：文件中的锁、进度及完成标记用于协调阶段；保留日志便于复核失败位置。
# A/B isolate context limit, B/C isolate repair count, C/D isolate failure fallback.
# 启用脚本列出的严格模式；未处理的命令失败由阶段错误处理接管。
set -euo pipefail
# 进入协议指定的工作目录，后续相对路径均以这里为基准。
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# 固定本阶段的工作位置，便于把数据、源码与产物关联到同一协议。
root=runs/accuracy-development-v1
# 准备当前阶段使用的目录，已存在的目录按命令参数处理。
mkdir -p "$root"
# 打开实验锁对应的文件描述符，供 flock 协调重复启动或资源占用。
exec 9>"$root/stage.lock"
# 按当前命令指定的等待或非阻塞模式获取锁。
flock -n 9 || exit 1
children=()
cleanup(){ echo failed_or_interrupted > "$root/stage.txt"; for child in "${children[@]}";do kill -TERM "$child" 2>/dev/null || true;done; }
# 登记退出或信号处理，异常时写入可检查的阶段状态。
trap cleanup ERR
trap 'cleanup;exit 143' TERM INT
# 先核验前置条件，避免在缺少配置、数据或完成标记时进入下一阶段。
test "$(cat runs/optimized-student-control/stage.txt)" = complete_development_only
# 定位本轮冻结源码，历史产物通过这份快照解释。
frozen="$root/frozen-code"
if [[ ! -d "$frozen" ]];then
  mkdir -p "$frozen/tableqa" "$frozen/scripts"
  # 复制本阶段需要保存的文件；按命令中的来源与目标维护快照关系。
  cp tableqa/*.py "$frozen/tableqa/";cp scripts/*.py "$frozen/scripts/";touch "$frozen/SNAPSHOT_READY"
fi
test -f "$frozen/SNAPSHOT_READY"
# 加载项目虚拟环境或阶段配置，使后续命令使用预期依赖。
source .venv/bin/activate
# 显式设置本阶段的环境变量，固定代码路径、缓存或设备可见性。
export PYTHONPATH="$PWD/$frozen" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo preparing > "$root/stage.txt"
if [[ ! -f "$root/protocol.json" ]];then python "$frozen/scripts/prepare_accuracy_stage.py";fi
echo evaluating_accuracy_controls > "$root/stage.txt"
# 为当前命令显式选择可见 GPU，使资源配置可复核。
CUDA_VISIBLE_DEVICES=1 python "$frozen/scripts/accuracy_condition.py" A > runs/setup/accuracy-A.log 2>&1 & pid_a=$!;children+=("$pid_a")
CUDA_VISIBLE_DEVICES=0 python "$frozen/scripts/accuracy_condition.py" C > runs/setup/accuracy-C-D.log 2>&1 & pid_c=$!;children+=("$pid_c")
# 等待指定后台任务结束，成功后才继续依赖它的阶段。
wait "$pid_a"
CUDA_VISIBLE_DEVICES=1 python "$frozen/scripts/accuracy_condition.py" B > runs/setup/accuracy-B.log 2>&1 & pid_b=$!;children+=("$pid_b")
wait "$pid_b";wait "$pid_c"
echo evaluating_scores > "$root/stage.txt"
# 遍历协议中显式列出的实验组或随机种子，逐项保存结果。
for mode in A B C D;do
  # 调用当前阶段的 Python 入口，参数中的文件明确给出本次输入与输出。
  python "$frozen/scripts/evaluate_wtq_official.py" "$root/samples.jsonl" "$root/$mode-predictions.jsonl" "$root/wtq-$mode" --evaluator data/raw/wikitq-official/WikiTableQuestions-master/evaluator.py --tagged data/raw/wikitq-official/WikiTableQuestions-master/tagged/data
done
for pair in A:B B:C C:D A:D;do
  a=${pair%:*};b=${pair#*:}
  python "$frozen/scripts/analyze_runs.py" "$root/samples.jsonl" "$root/$a-predictions.jsonl" "$root/comparison-$a-vs-$b" --predictions-b "$root/$b-predictions.jsonl" --official-a "$root/wtq-$a/scores.json" --official-b "$root/wtq-$b/scores.json"
done
echo complete_validation_tuning_confirmation_pending > "$root/stage.txt"
