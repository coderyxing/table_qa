#!/usr/bin/env bash
# 脚本用途：等待第四章释放实验锁，重跑题面引文定位修复后的第三章校验组。
# 执行环境：Bash；先核对正文中的项目路径、虚拟环境、GPU 编号和阶段依赖。
# 阅读提示：文件中的锁、进度及完成标记用于协调阶段；保留日志便于复核失败位置。
# 启用脚本列出的严格模式；未处理的命令失败由阶段错误处理接管。
set -Eeuo pipefail
# 限定新建文件的默认权限，运行日志和缓存沿用本次进程的权限策略。
umask 027
# 固定本阶段的工作位置，便于把数据、源码与产物关联到同一协议。
project=/memory/tableqa
run="$project/runs/20261110-pilot"
out="$run/chapter3-quote-v2"
py="$run/.venv/bin/python"
# 显式设置本阶段的环境变量，固定代码路径、缓存或设备可见性。
export TMPDIR="$project/tmp"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1

# 准备当前阶段使用的目录，已存在的目录按命令参数处理。
mkdir -p "$out"
status() {
    printf '%s %s\n' "$(date -Is)" "$1" | tee -a "$out/progress.log"
}
# 登记退出或信号处理，异常时写入可检查的阶段状态。
trap 'status "failed at line $LINENO (exit $?)"' ERR

# 进入协议指定的工作目录，后续相对路径均以这里为基准。
cd "$run/source-v3"
# 写入可读取的阶段进度，供外部观察当前工作位置。
status validating_source
# 调用当前阶段的 Python 入口，参数中的文件明确给出本次输入与输出。
"$py" -B -m unittest discover -s tests -p test_thesis_experiments.py -q
status queued_behind_chapter4
# 打开实验锁对应的文件描述符，供 flock 协调重复启动或资源占用。
exec 8>"$run/pilot.lock"
# 按当前命令指定的等待或非阻塞模式获取锁。
flock 8
# 等待正文指定的资源或阶段条件，满足退出判据后继续。
while ! nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits |
    awk '{if ($1 < 39000) busy=1} END {exit (NR != 2 || busy)}'; do
    status waiting_for_two_free_A40s
    # 在本次轮询之间留出间隔，降低持续检查对系统的干扰。
    sleep 60
done

# 遍历协议中显式列出的实验组或随机种子，逐项保存结果。
for group in fixed dynamic; do
    # 复制本阶段需要保存的文件；按命令中的来源与目标维护快照关系。
    cp -p "$run/chapter3/$group.jsonl" "$run/chapter3/$group.jsonl.meta.json" "$out/"
done
status extracting_verbatim_quotes
"$py" -u -m tableqa.thesis declare3 "$run/chapter3-dev60.jsonl" "$run/teacher.json" "$out/declarations.jsonl"
# 先核验前置条件，避免在缺少配置、数据或完成标记时进入下一阶段。
test "$(wc -l <"$out/declarations.jsonl")" -eq 60
"$py" - "$out/declarations.jsonl" <<'PY'
from pathlib import Path
import sys
from tableqa.data import read_jsonl

rows = read_jsonl(Path(sys.argv[1]))
bound = sum(bool(row["conditions"]) for row in rows)
resolved = sum(bool(row["conditions"]) and not row["unresolved"] for row in rows)
print(f"bound_declarations={bound}/60 fully_resolved={resolved}/60", flush=True)
if not bound or not resolved:
    raise SystemExit("The quote-binding development gate did not pass")
PY
for group in conditions row_domains no_repair; do
    status "chapter3_$group"
    "$py" -u -m tableqa.thesis predict3 "$group" "$run/chapter3-dev60.jsonl" "$run/teacher.json" \
        "$out/$group.jsonl" --declarations "$out/declarations.jsonl"
    test "$(wc -l <"$out/$group.jsonl")" -eq 60
done
status chapter3_quote_v2_pending_human_review
