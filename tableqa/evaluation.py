"""Diagnostic metrics, explicit external-score import and paired uncertainty."""
import math
import random
import re
import statistics
from collections import Counter


def normalize(v):
    return re.sub(r"\s+", " ", str(v).strip().casefold())


def label(v):
    if isinstance(v, list) and len(v) == 1: v = v[0]
    v = normalize(v)
    if v in {"1", "true", "entailed", "supported"}: return 1
    if v in {"0", "false", "refuted", "unsupported"}: return 0
    return None


def diagnostic_score(prediction, sample):
    if prediction["status"] != "ok": return 0.
    p, r = prediction["answer"], sample["reference"]
    if sample["task"] == "verification": return float(label(p) is not None and label(p) == label(r))
    if sample["task"] == "short":
        p = p if isinstance(p, list) else [p]; r = r if isinstance(r, list) else [r]
        return float(set(map(normalize, p)) == set(map(normalize, r)))
    # Token F1 is diagnostic only; not FeTaQA official BLEU/ROUGE.
    p = p if isinstance(p, str) else " ".join(map(str, p)) if isinstance(p, list) else str(p)
    a, b = Counter(normalize(p).split()), Counter(normalize(r).split())
    overlap = sum((a & b).values()); total = sum(a.values()) + sum(b.values())
    return 2 * overlap / total if total else 1.


def quantile(values, p):
    if not values: return None
    x = sorted(values); pos = (len(x) - 1) * p; low = math.floor(pos); high = math.ceil(pos)
    return x[low] + (x[high] - x[low]) * (pos - low)


def evaluate(predictions, samples):
    by_id = {(r["dataset"], r["id"]): r for r in samples}
    keys = [(r["dataset"], r["id"]) for r in predictions]
    if len(set(keys)) != len(keys): raise ValueError("Duplicate predictions")
    if set(keys) != set(by_id): raise ValueError("Predictions must cover EXACT reference set, including failures")
    result = {}
    for dataset in sorted({r["dataset"] for r in samples}):
        ps = [p for p in predictions if p["dataset"] == dataset]
        scores = [diagnostic_score(p, by_id[(p["dataset"], p["id"])]) for p in ps]
        latencies = [p["metrics"]["total_seconds"] for p in ps]
        tokens = [p["metrics"].get("input_tokens") for p in ps]
        result[dataset] = {"n": len(ps), "diagnostic_score": statistics.mean(scores),
            "official_score": None, "official_evaluation_required": dataset == "wikitq",
            "execution_success_rate": sum(p["status"] == "ok" and p.get('answer_source')!='direct_fallback' for p in ps) / len(ps),
            "pipeline_success_rate":sum(p['status']=='ok' for p in ps)/len(ps),
            "fallback_answers":sum(p.get('answer_source')=='direct_fallback' for p in ps),
            "p50_seconds": quantile(latencies, .5), "p95_seconds": quantile(latencies, .95),
            "mean_calls": statistics.mean(p["metrics"]["model_calls"] for p in ps),
            "mean_input_tokens": statistics.mean(tokens) if all(t is not None for t in tokens) else None,
            "cache_hits": sum(p["metrics"].get("cache_hits", 0) for p in ps)}
    return result


def paired_bootstrap(a, b, repeats=2000, seed=42):
    if not a or len(a) != len(b): raise ValueError("Paired nonempty equal-length scores required")
    rng = random.Random(seed); differences = []
    for _ in range(repeats):
        indices = [rng.randrange(len(a)) for _ in a]
        differences.append(statistics.mean(b[i] - a[i] for i in indices))
    return {"difference_b_minus_a": statistics.mean(y-x for x,y in zip(a,b)),
            "ci95": [quantile(differences, .025), quantile(differences, .975)]}


def subgroup_report(predictions, samples):
    lookup = {(r["dataset"], r["id"]): r for r in samples}; groups = {}
    for p in predictions:
        r = lookup[(p["dataset"], p["id"])]
        n = len(r["table"]["rows"]); length = "rows<=20" if n <= 20 else "rows21-100" if n <=100 else "rows>100"
        kinds = {o["op"] for o in p.get("operations", [])}
        for tag in [length, f"operations={len(p.get('operations', []))}"] + sorted(kinds):
            groups.setdefault((r["dataset"], tag), []).append(diagnostic_score(p, r))
    return [{"dataset": d, "subgroup": k, "n": len(v), "diagnostic_score": statistics.mean(v)}
            for (d,k),v in sorted(groups.items())]


def select_cases(predictions, seed=42, per_category=5):
    groups = {}; rng = random.Random(seed)
    for p in sorted(predictions, key=lambda x: (x["dataset"], x["id"])):
        category = "failed" if p["status"] != "ok" else "repaired" if p["errors"] else "executable_unreviewed"
        groups.setdefault((p["dataset"], category), []).append(p)
    out = []
    for (d, cat), ps in sorted(groups.items()):
        rng.shuffle(ps)
        for p in ps[:per_category]:
            out.append({"dataset": d, "id": p["id"], "category": cat,
                        "reviewer1": None, "reviewer2": None, "evidence_correct": None,
                        "unsupported_claims": None, "selection_seed": seed})
    return out
