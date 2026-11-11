"""Reproduce and repair two known type/scope failures with hand-written operations."""

import argparse
import json
from pathlib import Path

from tableqa.core import ExecutionError, TableState, run_plan
from tableqa.data import digest, read_jsonl
from tableqa.inference import parse
from tableqa.quantities import quantity_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve the existing replay report")

    keys = {("tabfact", "2-18823880-1.html.csv:9"), ("wikitq", "nt-11054")}
    samples = {(r["dataset"], r["id"]): r for r in read_jsonl(args.stage / "samples.jsonl")
               if (r["dataset"], r["id"]) in keys}
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
