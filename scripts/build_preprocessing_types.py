"""Annotate frozen development tables with source-bound numeric types."""

import argparse
from collections import Counter
import json
from pathlib import Path

from tableqa.data import digest, read_jsonl, write_jsonl
from tableqa.quantities import QuantityError, parse_quantity


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output
    if root.exists():
        raise ValueError("Preserve the existing annotation output")
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
    write_jsonl(root / "annotations.jsonl", outputs)
    (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
