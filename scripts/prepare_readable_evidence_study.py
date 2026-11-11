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


class ExposureReplay(ReplayModel):
    def __init__(self, responses):
        super().__init__(responses)
        self.exposed_calls = 0

    def generate(self, messages):
        response = super().generate(messages)
        if json.loads(messages[1]["content"]).get("original_evidence_cells"):
            self.exposed_calls += 1
        return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
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
    samples = read_jsonl(source / "samples.jsonl")
    predictions = read_jsonl(source / "full-predictions.jsonl")
    lookup = {(r["dataset"], r["id"]): r for r in predictions}
    options = dict(protocol["settings"])
    mode = options.pop("mode")
    selected = []
    exposures = []
    for sample in samples:
        previous = lookup[(sample["dataset"], sample["id"])]
        replay = ExposureReplay(r["text"] for r in previous["responses"])
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
    write_jsonl(root / "samples.jsonl", selected)
    write_jsonl(root / "historical-exposure.jsonl", exposures)
    for name, value in [("protocol.json", protocol), ("run.json", metadata(protocol))]:
        (root / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print("READABLE_EVIDENCE_FROZEN", protocol["per_task_counts"], flush=True)


if __name__ == "__main__":
    main()
