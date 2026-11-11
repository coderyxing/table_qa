# 脚本用途：重放已知的类型及行域失败，并保存人工设计后缀的离线修复。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Reproduce and repair two known type/scope failures with hand-written operations."""

import argparse
import json
from pathlib import Path

from tableqa.core import ExecutionError, TableState, run_plan
from tableqa.data import digest, read_jsonl
from tableqa.inference import parse
from tableqa.quantities import quantity_text


# 解析当前脚本参数，按模块说明执行本次冻结阶段并保存产物。
# 相关调用：args.output.write_text、run_plan、read_jsonl、digest。
# 异常约束：Preserve the existing replay report
def main():
    # 定义本入口的参数解析器；位置参数、默认值和允许取值由相邻 add_argument 明确声明。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("output", type=Path)
    # 解析本次命令的实际参数，后续样本路径和实验开关均由此确定。
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve the existing replay report")

    keys = {("tabfact", "2-18823880-1.html.csv:9"), ("wikitq", "nt-11054")}
    # 读取结构化输入：args.stage / 'samples.jsonl'，保留原有样本身份字段。
    samples = {(r["dataset"], r["id"]): r for r in read_jsonl(args.stage / "samples.jsonl")
               if (r["dataset"], r["id"]) in keys}
    # 读取结构化输入：args.stage / 'B5/predictions.jsonl'，保留原有样本身份字段。
    predictions = {(r["dataset"], r["id"]): r for r in read_jsonl(args.stage / "B5/predictions.jsonl")
                   if (r["dataset"], r["id"]) in keys}
    if set(samples) != keys or set(predictions) != keys:
        raise ValueError("Expected both historical regression cases")

    rows = []
    for key in sorted(keys):
        sample, previous = samples[key], predictions[key]
        old = parse(previous["responses"][0]["text"])["operations"]
        initial = TableState.from_table(sample["table"])
        try:
            run_plan(initial, old)
        except ExecutionError as error:
            failure = str(error)
        else:
            raise ValueError("Historical failed block unexpectedly executed")
        if sample["dataset"] == "tabfact":
            if old[0].get("column") != "c2" or old[0].get("value") != "3.33 ghz":
                raise ValueError("Unexpected historical frequency filter")
            derive = {"op": "derive", "as": "frequency_ghz",
                      "expression": {"op": "quantity", "args": [{"col": "c2"}], "unit": "GHz"}}
            normalized_filter = {**old[0], "column": "frequency_ghz",
                                 "value": quantity_text(old[0]["value"], "GHz")}
            revised = [derive, normalized_filter, *old[1:]]
            terminal = {"op": "answer", "assert": {"kind": "row_count", "cmp": "gt", "value": 0}}
        else:
            if old[0].get("column") != "c3" or old[1].get("column") != "c0":
                raise ValueError("Unexpected historical finisher plan")
            # The question explicitly asks about a finisher.
            exclude_retired = {"op": "filter", "column": "c0", "cmp": "ne", "value": "Ret", "type": "text"}
            revised = [old[0], exclude_retired, *old[1:]]
            terminal = {"op": "answer", "columns": ["c1"]}
        # 在明确给出的表状态上重放操作链，核对执行结果和来源追踪。
        before_answer, trace = run_plan(initial, revised)
        result, _ = run_plan(before_answer, [terminal])
        if result.answer != sample["reference"]:
            raise ValueError(f"Diagnostic answer disagrees with the stored reference for {sample['id']}")
        rows.append({"dataset": sample["dataset"], "id": sample["id"], "sample_hash": digest(sample),
                     "historical_prediction_hash": digest(previous), "question": sample["question"],
                     "historical_block": old, "historical_error": failure,
                     "manual_replacement_block": revised, "manual_terminal": terminal,
                     "rows_after_each_operation": [len(step["after"]["rows"]) for step in trace],
                     "replay_answer": result.answer, "reference": sample["reference"]})
    report = {"scope": "offline_diagnostic_with_manual_operations", "cases": rows}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([{"id": r["id"], "answer": r["replay_answer"]} for r in rows], ensure_ascii=False))


if __name__ == "__main__":
    main()
