# 脚本用途：冻结按表隔离的预处理开发集，记录题型与历史接触来源。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Freeze a table-disjoint development cohort for preprocessing controls."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re

from tableqa.data import digest, read_jsonl, table_fingerprint, write_jsonl
from tableqa.quantities import QuantityError, parse_quantity


# 从题面或表格内容检测数量特征，供开发队列分层选样。
# 输入：table（原始表头与行矩阵）。
# 返回形式含：False / True。
def has_quantity(table):
    for row in table["rows"]:
        for value in row:
            try:
                if parse_quantity(value).source_unit != "1":
                    return True
            except QuantityError:
                continue
    return False


# 解析当前脚本参数，按模块说明执行本次冻结阶段并保存产物。
# 相关调用：read_jsonl、write_jsonl、(root / 'protocol.json').write_text、digest。
# 异常约束：Preserve the existing frozen cohort
def main():
    # 定义本入口的参数解析器；位置参数、默认值和允许取值由相邻 add_argument 明确声明。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    # 解析本次命令的实际参数，后续样本路径和实验开关均由此确定。
    args = parser.parse_args()
    root = args.output
    if root.exists():
        raise ValueError("Preserve the existing frozen cohort")

    source = Path("data/normalized/joint_validation_clean.jsonl")
    # 读取结构化输入：source，保留原有样本身份字段。
    samples = read_jsonl(source)
    if any(row["split"] != "validation" for row in samples):
        raise ValueError("Expected only validation rows")
    history = sorted(Path("runs").rglob("*samples.jsonl"))
    previous_tables = set()
    history_counts = {}
    for path in history:
        count = 0
        for row in read_jsonl(path):
            if row.get("split") == "validation" and "table" in row:
                previous_tables.add(table_fingerprint(row["table"]))
                count += 1
        if count:
            history_counts[path.as_posix()] = count
    if not previous_tables:
        raise ValueError("Historical development tables are required for isolation")

    seed = 2044
    rng = random.Random(seed)
    conditions = re.compile(r"\b(?:and|both|after|before|at least|at most|lowest|highest|least|most)\b", re.IGNORECASE)
    seen = set(previous_tables)
    selected = []
    counts = Counter()
    for dataset in ["wikitq", "tabfact", "fetaqa"]:
        candidates = [(table_fingerprint(row["table"]), row)
                      for row in sorted((s for s in samples if s["dataset"] == dataset), key=lambda item: item["id"])]
        rng.shuffle(candidates)
        task_count = 0
        for stratum, target in [("quantity_in_table", 8), ("condition_terms", 8), ("remaining", 20)]:
            for fingerprint, row in candidates:
                if task_count >= 20 or counts[(dataset, stratum)] >= target:
                    break
                if fingerprint in seen:
                    continue
                if stratum == "quantity_in_table" and not has_quantity(row["table"]):
                    continue
                if stratum == "condition_terms" and not conditions.search(row["question"]):
                    continue
                selected.append({**row, "preprocessing_sampling_stratum": stratum})
                counts[(dataset, stratum)] += 1
                task_count += 1
                seen.add(fingerprint)
    if len(selected) != 60 or any(sum(row["dataset"] == name for row in selected) != 20
                                  for name in ["wikitq", "tabfact", "fetaqa"]):
        raise ValueError("Expected 20 distinct-table development cases per task")
    rng.shuffle(selected)
    protocol = {"split": "validation", "development_only": True, "seed": seed,
                "source": str(source), "source_sha256": digest(samples), "samples_sha256": digest(selected),
                "counts": {f"{task}:{stratum}": count for (task, stratum), count in counts.items()},
                "excluded_previous_tables": len(previous_tables), "historical_validation_files": history_counts,
                "sampling_uses_references_or_scores": False,
                "limits": "Question keywords and recognized table units are sampling proxies. Prior exposure check is limited to recorded sample files."}
    root.mkdir(parents=True)
    # 保存逐条结构化产物到 root / 'samples.jsonl'，供后续重放或审核读取。
    write_jsonl(root / "samples.jsonl", selected)
    (root / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected": len(selected), "counts": protocol["counts"],
                      "excluded_previous_tables": len(previous_tables)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
