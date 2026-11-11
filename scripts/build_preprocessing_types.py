# 脚本用途：为冻结开发表生成绑定原始单元格的数值类型与单位注解。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Annotate frozen development tables with source-bound numeric types."""

import argparse
from collections import Counter
import json
from pathlib import Path

from tableqa.data import digest, read_jsonl, write_jsonl
from tableqa.quantities import QuantityError, parse_quantity


# 扫描原表并给单元格附加可追溯的类型或单位标记。
# 输入：table（原始表头与行矩阵）。
# 返回形式含：annotations。
def annotate_table(table):
    annotations = []
    for row_index, row in enumerate(table["rows"]):
        for column_index, value in enumerate(row):
            cell = {"cell_id": f"r{row_index}:c{column_index}", "raw": value}
            try:
                quantity = parse_quantity(value)
            except QuantityError:
                cell.update(type="raw", status="unresolved", canonical_value=None, unit=None)
            else:
                explicit = quantity.source_unit != "1"
                cell.update(type="quantity" if explicit else "number", status="parsed",
                            canonical_value=format(quantity.value, "f"),
                            unit=quantity.base_unit if explicit else None,
                            source_unit=quantity.source_unit if explicit else None,
                            dimension=quantity.dimension if explicit else None)
            annotations.append(cell)
    return annotations


# 解析当前脚本参数，按模块说明执行本次冻结阶段并保存产物。
# 相关调用：read_jsonl、write_jsonl、(root / 'report.json').write_text、(args.development / 'protocol.json').read_text。
# 异常约束：Preserve the existing annotation output
def main():
    # 定义本入口的参数解析器；位置参数、默认值和允许取值由相邻 add_argument 明确声明。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development", type=Path)
    parser.add_argument("output", type=Path)
    # 解析本次命令的实际参数，后续样本路径和实验开关均由此确定。
    args = parser.parse_args()
    root = args.output
    if root.exists():
        raise ValueError("Preserve the existing annotation output")
    # 读取结构化输入：args.development / 'samples.jsonl'，保留原有样本身份字段。
    samples = read_jsonl(args.development / "samples.jsonl")
    protocol = json.loads((args.development / "protocol.json").read_text(encoding="utf-8"))
    if digest(samples) != protocol["samples_sha256"] or any(s["split"] != "validation" for s in samples):
        raise ValueError("Frozen validation input changed")
    outputs = []
    counts = Counter()
    for sample in samples:
        cells = annotate_table(sample["table"])
        counts.update(cell["type"] for cell in cells)
        outputs.append({"dataset": sample["dataset"], "id": sample["id"],
                        "rule_version": "quantity-v2", "table_sha256": digest(sample["table"]),
                        "cells": cells})
    report = {"samples": len(samples), "source_sha256": digest(samples),
              "annotations_sha256": digest(outputs), "cell_counts": dict(counts),
              "scope": "Syntactic cell annotations before prompt integration. Values and coordinates preserved."}
    root.mkdir(parents=True)
    # 保存逐条结构化产物到 root / 'annotations.jsonl'，供后续重放或审核读取。
    write_jsonl(root / "annotations.jsonl", outputs)
    (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
