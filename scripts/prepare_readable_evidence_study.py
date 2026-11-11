# 脚本用途：从开发轨迹中确定原始证据被移出视图的样本，冻结可读证据对照。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Freeze a development cohort with an opportunity for readable evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from tableqa.cli import metadata
from tableqa.data import digest, read_jsonl, write_jsonl
from tableqa.experiments import settings
from tableqa.inference import predict
from tableqa.models import ReplayModel


# ExposureReplay：回放历史轨迹并记录后续提示中的证据展示机会。
class ExposureReplay(ReplayModel):
    # 初始化 ExposureReplay 的依赖及持久状态；具体预算和身份字段见赋值。
    # 输入：responses。
    def __init__(self, responses):
        super().__init__(responses)
        self.exposed_calls = 0

    # 按当前适配接口生成或重放响应，同时保留调用所需的观测状态。
    # 输入：messages（模型可见消息）。
    # 相关调用：super().generate。
    # 返回形式含：response。
    def generate(self, messages):
        response = super().generate(messages)
        if json.loads(messages[1]["content"]).get("original_evidence_cells"):
            self.exposed_calls += 1
        return response


# 解析当前脚本参数，按模块说明执行本次冻结阶段并保存产物。
# 相关调用：read_jsonl、write_jsonl、(source / 'completion-audit.json').read_text、(source / 'protocol.json').read_text。
# 异常约束：Historical source bindings changed
def main():
    # 定义本入口的参数解析器；位置参数、默认值和允许取值由相邻 add_argument 明确声明。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    # 解析本次命令的实际参数，后续样本路径和实验开关均由此确定。
    root = parser.parse_args().output
    source = Path("runs/teacher-ablation-20261107/prepared")
    audit = json.loads((source / "completion-audit.json").read_text(encoding="utf-8"))
    files = ["samples.jsonl", "protocol.json", "full-predictions.jsonl"]
    if not audit["passed"] or any(
        hashlib.sha256((source / name).read_bytes()).hexdigest() != audit["files_sha256"][name]
        for name in files
    ):
        raise ValueError("Historical source bindings changed")
    protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    # 读取结构化输入：source / 'samples.jsonl'，保留原有样本身份字段。
    samples = read_jsonl(source / "samples.jsonl")
    # 读取结构化输入：source / 'full-predictions.jsonl'，保留原有样本身份字段。
    predictions = read_jsonl(source / "full-predictions.jsonl")
    lookup = {(r["dataset"], r["id"]): r for r in predictions}
    options = dict(protocol["settings"])
    mode = options.pop("mode")
    selected = []
    exposures = []
    for sample in samples:
        previous = lookup[(sample["dataset"], sample["id"])]
        replay = ExposureReplay(r["text"] for r in previous["responses"])
        # 对当前样本运行完整推理入口，返回值包含答案、状态、错误和调用成本。
        result = predict(sample, replay, settings(mode, **options, readable_evidence=True))
        if any(result[k] != previous[k] for k in ["status", "operations", "answer"]):
            raise ValueError(f"Historical execution replay differs for {sample['id']}")
        exposures.append({"dataset": sample["dataset"], "id": sample["id"], "exposed_calls": replay.exposed_calls})
        if replay.exposed_calls:
            selected.append(sample)
    if not selected:
        raise ValueError("No historical states expose missing original evidence cells")
    protocol.pop("per_task", None)
    protocol.update(
        study="readable_evidence",
        conditions={"coordinates": {}, "no_evidence": {"evidence": False},
                    "readable": {"readable_evidence": True}},
        n=len(selected),
        per_task_counts=dict(Counter(s["dataset"] for s in selected)),
        samples_sha256=digest(selected),
        historical_samples_sha256=digest(samples),
        selection="All previously used validation inputs where replay with the historical full arm's recorded responses exposed original evidence values in a subsequent prompt. Uses no labels or score criteria.",
        evidence_policy="Retain coordinate list and add at most eight original cells absent from the current compact view, within a 768-character serialized allowance.",
        limits=["Exposure-selected development cohort and single greedy model run.",
                "Each arm regenerates the entire chain. Prefixes can differ after evidence changes.",
                "No-evidence arm disables tracking and exposure as in the historical ablation."],
    )
    root.mkdir(parents=True)
    # 保存逐条结构化产物到 root / 'samples.jsonl'，供后续重放或审核读取。
    write_jsonl(root / "samples.jsonl", selected)
    # 保存逐条结构化产物到 root / 'historical-exposure.jsonl'，供后续重放或审核读取。
    write_jsonl(root / "historical-exposure.jsonl", exposures)
    for name, value in [("protocol.json", protocol), ("run.json", metadata(protocol))]:
        (root / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print("READABLE_EVIDENCE_FROZEN", protocol["per_task_counts"], flush=True)


if __name__ == "__main__":
    main()
