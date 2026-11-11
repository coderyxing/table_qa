# 模块职责：计算诊断答案分数、完整队列成功率、延迟分位数和按样本配对的不确定性。
# 接口约定：样本、操作和结果采用本项目的结构化字典；细节见各入口及字段检查。
"""Diagnostic metrics, explicit external-score import and paired uncertainty."""
import math
import random
import re
import statistics
from collections import Counter


# 归一化诊断答案的大小写和空白，固定简单集合匹配口径。
# 输入：v。
def normalize(v):
    return re.sub(r"\s+", " ", str(v).strip().casefold())


# 将允许的真假表示解析为 0/1；无法识别时返回 None。
# 输入：v。
# 返回形式含：None / 1 / 0。
def label(v):
    if isinstance(v, list) and len(v) == 1: v = v[0]
    v = normalize(v)
    if v in {"1", "true", "entailed", "supported"}: return 1
    if v in {"0", "false", "refuted", "unsupported"}: return 0
    return None


# 按短答、验证或自由文本计算诊断分数，运行失败计零。
# 输入：prediction（单条预测或教师轨迹）；sample（规范样本记录）。
# 返回形式含：0.0。
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


# 在排序后的观测上做线性插值，空列表保留缺失值。
# 输入：values（待统计或序列化值）；p。
def quantile(values, p):
    if not values: return None
    x = sorted(values); pos = (len(x) - 1) * p; low = math.floor(pos); high = math.ceil(pos)
    return x[low] + (x[high] - x[low]) * (pos - low)


# 强制完整 ID 覆盖后，按数据集汇总分数、成功率与调用成本。
# 输入：predictions（完整预测集合）；samples（固定样本集合）。
# 返回形式含：result。
# 异常约束：Duplicate predictions
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


# 对相同样本的两组分数一起重采样，估计 B 减 A 的区间。
# 输入：a；b；repeats（重采样次数）；seed（随机种子）。
# 返回形式含：字段 difference_b_minus_a、ci95。
# 异常约束：Paired nonempty equal-length scores required
def paired_bootstrap(a, b, repeats=2000, seed=42):
    if not a or len(a) != len(b): raise ValueError("Paired nonempty equal-length scores required")
    rng = random.Random(seed); differences = []
    for _ in range(repeats):
        indices = [rng.randrange(len(a)) for _ in a]
        differences.append(statistics.mean(b[i] - a[i] for i in indices))
    return {"difference_b_minus_a": statistics.mean(y-x for x,y in zip(a,b)),
            "ci95": [quantile(differences, .025), quantile(differences, .975)]}


# 按表长和预测操作特征生成事后诊断子组。
# 输入：predictions（完整预测集合）；samples（固定样本集合）。
# 返回形式含：列表记录。
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


# 固定随机种子按运行状态选取案例，供后续人工分类。
# 输入：predictions（完整预测集合）；seed（随机种子）；per_category。
# 返回形式含：out。
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
