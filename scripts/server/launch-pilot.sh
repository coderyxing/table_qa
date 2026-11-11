#!/usr/bin/env bash
# 脚本用途：创建本次运行环境、固定模型与数据版本，并串行执行双章 60 题试跑。
# 执行环境：Bash；先核对正文中的项目路径、虚拟环境、GPU 编号和阶段依赖。
# 阅读提示：文件中的锁、进度及完成标记用于协调阶段；保留日志便于复核失败位置。
# 启用脚本列出的严格模式；未处理的命令失败由阶段错误处理接管。
set -Eeuo pipefail
# 限定新建文件的默认权限，运行日志和缓存沿用本次进程的权限策略。
umask 027

# 固定本阶段的工作位置，便于把数据、源码与产物关联到同一协议。
project=/memory/tableqa
run="$project/runs/20261110-pilot"
base=/home/fzus/miniconda3/envs/BREAK/bin/python
py="$run/.venv/bin/python"
# 显式设置本阶段的环境变量，固定代码路径、缓存或设备可见性。
export HF_HOME="$project/cache/huggingface"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_ETAG_TIMEOUT=30
export HF_HUB_DOWNLOAD_TIMEOUT=180
export TMPDIR="$project/tmp"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

# 打开实验锁对应的文件描述符，供 flock 协调重复启动或资源占用。
exec 8>"$run/pilot.lock"
# 按当前命令指定的等待或非阻塞模式获取锁。
flock -n 8
# 进入协议指定的工作目录，后续相对路径均以这里为基准。
cd "$run/source-v2"
status() {
    printf '%s %s\n' "$(date -Is)" "$1" | tee -a "$run/progress.log"
}
# 登记退出或信号处理，异常时写入可检查的阶段状态。
trap 'status "failed at line $LINENO (exit $?)"' ERR

# 写入可读取的阶段进度，供外部观察当前工作位置。
status preflight
if [[ ! -x "$py" ]]; then
    "$base" -m venv --system-site-packages "$run/.venv"
fi
# 调用当前阶段的 Python 入口，参数中的文件明确给出本次输入与输出。
"$py" -m pip install --disable-pip-version-check --no-deps databench-eval==4.0.1
PATH="$run/.venv/bin:$PATH" bash scripts/preflight.sh >"$run/preflight.txt" 2>&1
"$py" -B -m unittest discover -s tests -q >"$run/tests.log" 2>&1
"$py" - "$run" <<'PY'
from collections import Counter
import json
from pathlib import Path
import sys
from tableqa.data import digest, read_jsonl
from databench_eval import Evaluator
import torch

run = Path(sys.argv[1])
samples = read_jsonl(run / "chapter3-dev60.jsonl")
protocol = json.loads((run / "chapter3-dev60.protocol.json").read_text())
assert len(samples) == 60 and all(s["split"] == "validation" for s in samples)
assert Counter(s["dataset"] for s in samples) == {"wikitq": 20, "tabfact": 20, "fetaqa": 20}
assert digest(samples) == protocol["samples_sha256"]
assert torch.cuda.device_count() == 2 and torch.cuda.is_bf16_supported()
assert Evaluator(qa=[]).compare("2", "2", "number")
PY

status downloading_pinned_models_and_databench
"$py" -u - "$project" >"$run/models-download.log" 2>&1 <<'PY' &
from pathlib import Path
import sys
from huggingface_hub import snapshot_download

project = Path(sys.argv[1])
for name, revision, shards in [
    ("Qwen2.5-32B-Instruct", "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd", 17),
    ("Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1", 2),
]:
    target = project / "models" / f"{name}-{revision[:8]}"
    print(f"Downloading {name} at {revision} to {target}", flush=True)
    snapshot_download(repo_id=f"Qwen/{name}", revision=revision, local_dir=target,
                      allow_patterns=["*.safetensors", "*.json", "merges.txt", "tokenizer.model", "*.jinja"],
                      max_workers=3)
    assert len(list(target.glob("*.safetensors"))) == shards
    print(f"Verified {shards} shards for {name}", flush=True)
PY
model_pid=$!
"$py" -u - "$project" >"$run/databench-download.log" 2>&1 <<'PY' &
from pathlib import Path
import sys
from huggingface_hub import snapshot_download

project = Path(sys.argv[1])
revision = "e75d53add267d2f9cfa32efd65ad77f0807adfad"
target = project / "data" / "chapter4" / f"databench-{revision[:8]}"
print(f"Downloading DataBench at {revision} to {target}", flush=True)
snapshot_download(repo_id="cardiffnlp/databench", repo_type="dataset", revision=revision,
                  local_dir=target, allow_patterns=["data/*/all.parquet", "data/*/qa.parquet", "README.md"],
                  max_workers=5)
assert len(list((target / "data").glob("*/all.parquet"))) == 80
print("Verified 80 complete tables", flush=True)
PY
data_pid=$!

# 等待指定后台任务结束，成功后才继续依赖它的阶段。
wait "$model_pid"
model="$project/models/Qwen2.5-32B-Instruct-5ede1c97"
student="$project/models/Qwen2.5-3B-Instruct-aa8e7253"
tables="$project/data/chapter4/databench-e75d53ad/data"
printf '{"backend":"local","model":"%s","device_map":"balanced","max_input_tokens":8192,"max_new_tokens":512,"seed":42}\n' "$model" >"$run/teacher.json"
export CUDA_VISIBLE_DEVICES=0,1

wait_for_gpus() {
    # 等待正文指定的资源或阶段条件，满足退出判据后继续。
    while ! nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits |
        awk '{if ($1 < 39000) busy=1} END {exit (NR != 2 || busy)}'; do
        status waiting_for_two_free_A40s
        # 在本次轮询之间留出间隔，降低持续检查对系统的干扰。
        sleep 60
    done
}

wait_for_gpus
status chapter3_declarations
# 准备当前阶段使用的目录，已存在的目录按命令参数处理。
mkdir -p "$run/chapter3"
"$py" -u -m tableqa.thesis declare3 "$run/chapter3-dev60.jsonl" "$run/teacher.json" "$run/chapter3/declarations.jsonl"
# 遍历协议中显式列出的实验组或随机种子，逐项保存结果。
for group in fixed dynamic conditions row_domains no_repair; do
    status "chapter3_${group}"
    "$py" -u -m tableqa.thesis predict3 "$group" "$run/chapter3-dev60.jsonl" "$run/teacher.json" \
        "$run/chapter3/$group.jsonl" --declarations "$run/chapter3/declarations.jsonl"
    # 先核验前置条件，避免在缺少配置、数据或完成标记时进入下一阶段。
    test "$(wc -l <"$run/chapter3/$group.jsonl")" -eq 60
done
status chapter3_dev60_generated_pending_human_review

wait "$data_pid"
status chapter4_prepare
mkdir -p "$run/chapter4"
"$py" -u -m tableqa.thesis prepare4 "$tables" "$run/chapter4/data" --revision e75d53add267d2f9cfa32efd65ad77f0807adfad
wait_for_gpus
status chapter4_train60_teacher
"$py" -u -m tableqa.thesis predict4 L4 "$run/chapter4/data/train.jsonl" "$tables" "$student" \
    "$run/chapter4/teacher60.jsonl" --config "$run/teacher.json" --teacher --limit 60
test "$(wc -l <"$run/chapter4/teacher60.jsonl")" -eq 60
status chapter4_replay_supervision
"$py" -u -m tableqa.thesis records4 L4 "$run/chapter4/data/train.jsonl" "$tables" "$student" \
    "$run/chapter4/records.jsonl" --traces "$run/chapter4/teacher60.jsonl"
status pilot_generation_complete
