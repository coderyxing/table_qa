#!/usr/bin/env bash
# 脚本用途：运行有界的中等规模教师与学生开发流程，记录阶段状态。
# 执行环境：Bash；先核对正文中的项目路径、虚拟环境、GPU 编号和阶段依赖。
# 阅读提示：文件中的锁、进度及完成标记用于协调阶段；保留日志便于复核失败位置。
# Bounded development workflow, not final multi-seed thesis experiments.
# 启用脚本列出的严格模式；未处理的命令失败由阶段错误处理接管。
set -euo pipefail
# 进入协议指定的工作目录，后续相对路径均以这里为基准。
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# 准备当前阶段使用的目录，已存在的目录按命令参数处理。
mkdir -p runs/medium-development
# 打开实验锁对应的文件描述符，供 flock 协调重复启动或资源占用。
exec 9>runs/medium-development/stage.lock
# 按当前命令指定的等待或非阻塞模式获取锁。
flock -n 9 || { echo 'Medium stage already running'; exit 1; }
# 加载项目虚拟环境或阶段配置，使后续命令使用预期依赖。
source .venv/bin/activate
# 显式设置本阶段的环境变量，固定代码路径、缓存或设备可见性。
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
status_file=runs/medium-development/stage.txt
# 登记退出或信号处理，异常时写入可检查的阶段状态。
trap 'echo failed_or_interrupted > "$status_file"' ERR
trap 'echo interrupted > "$status_file"; exit 143' TERM INT
# 定位本轮冻结源码，历史产物通过这份快照解释。
frozen=runs/medium-development/frozen-code
if [[ ! -d "$frozen" ]]; then
  mkdir -p "$frozen/tableqa" "$frozen/scripts"
  # 复制本阶段需要保存的文件；按命令中的来源与目标维护快照关系。
  cp tableqa/*.py "$frozen/tableqa/"
  cp scripts/*.py scripts/*.sh "$frozen/scripts/"
  touch "$frozen/SNAPSHOT_READY"
fi
# 先核验前置条件，避免在缺少配置、数据或完成标记时进入下一阶段。
test -f "$frozen/SNAPSHOT_READY"
export PYTHONPATH="$PWD/$frozen"
if [[ ! -f runs/medium-development/protocol.json ]]; then python "$frozen/scripts/prepare_medium_stage.py"; fi
echo teacher_A4 > "$status_file"
# 为当前命令显式选择可见 GPU，使资源配置可复核。
CUDA_VISIBLE_DEVICES=0,1 python -m tableqa predict runs/medium-development/samples.jsonl runs/medium-development/A4.json runs/medium-development/A4-predictions.jsonl --resume
echo direct_A0 > "$status_file"
CUDA_VISIBLE_DEVICES=0,1 python -m tableqa predict runs/medium-development/samples.jsonl runs/medium-development/A0.json runs/medium-development/A0-predictions.jsonl --resume
echo quality_audit > "$status_file"
# 调用当前阶段的 Python 入口，参数中的文件明确给出本次输入与输出。
python -m tableqa audit-trajectories runs/medium-development/samples.jsonl runs/medium-development/A4-predictions.jsonl runs/medium-development/quality.jsonl
python -m tableqa evaluate runs/medium-development/samples.jsonl runs/medium-development/A4-predictions.jsonl runs/medium-development/teacher-evaluation.json
python -m tableqa evaluate runs/medium-development/samples.jsonl runs/medium-development/A0-predictions.jsonl runs/medium-development/direct-evaluation.json
# 遍历协议中显式列出的实验组或随机种子，逐项保存结果。
for mode in A4 A0; do
  python "$frozen/scripts/evaluate_wtq_official.py" runs/medium-development/samples.jsonl "runs/medium-development/$mode-predictions.jsonl" "runs/medium-development/wtq-$mode" --evaluator data/raw/wikitq-official/WikiTableQuestions-master/evaluator.py --tagged data/raw/wikitq-official/WikiTableQuestions-master/tagged/data
done
echo freeform_review_pending > "$status_file"
# Convert to the shared pilot source contract while preserving raw run files.
mkdir -p runs/medium-development/trajectory-source
cp runs/medium-development/samples.jsonl runs/medium-development/trajectory-source/samples.jsonl
cp runs/medium-development/A4-predictions.jsonl runs/medium-development/trajectory-source/predictions.jsonl
cp runs/medium-development/teacher-evaluation.json runs/medium-development/trajectory-source/report.json
python "$frozen/scripts/prepare_freeform_review.py" --source runs/medium-development/trajectory-source
echo students_B2_B5 > "$status_file"
CUDA_VISIBLE_DEVICES=0 python "$frozen/scripts/paired_distillation_pilot.py" --source runs/medium-development/trajectory-source --output runs/medium-development/students --epochs 3 --rank 32 --eval-per-dataset 20
echo development_complete_freeform_review_still_pending > "$status_file"
