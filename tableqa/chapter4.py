# 模块职责：组织 DataBench 完整表评测、长短表增益、探查成本、配对防泄露及学生可见状态重放。
# 接口约定：样本、操作和结果采用本项目的结构化字典；细节见各入口及字段检查。
"""DataBench cohort checks, supervision replay and Chapter 4 reports."""
from collections import defaultdict
import json
import math
import unicodedata
from random import Random
from statistics import mean

from .data import digest, table_fingerprint
from .evaluation import quantile
from .inference import parse
from .longtable import INPUT_LIMIT, dump, predict_longtable, replay_plan
from .models import ReplayModel


# 用 DataBench 官方比较函数判定单题；运行失败的样本得零分。
# 输入：prediction（单条预测或教师轨迹）；sample（规范样本记录）。
def official_score(prediction, sample):
    from databench_eval import Evaluator

    if prediction["status"] != "ok":
        return False
    return bool(Evaluator(qa=[]).compare(prediction["answer"], sample["reference"], sample["semantic"]))


# 建立样本 ID 索引，遇到重复立即报错，防止静默覆盖。
# 输入：rows。
# 返回形式含：lookup。
# 异常约束：Duplicate sample IDs
def id_index(rows):
    lookup = {r["id"]: r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate sample IDs")
    return lookup


# 要求预测与参考集合逐 ID 完整相等，包括执行失败样本。
# 输入：samples（固定样本集合）；predictions（完整预测集合）。
# 返回形式含：2 项组成的结果。
# 异常约束：Every example needs one prediction, including failures
def _cohort(samples, predictions):
    data = id_index(samples)
    outcomes = id_index(predictions)
    if not data or set(data) != set(outcomes):
        raise ValueError("Every example needs one prediction, including failures")
    return data, outcomes


# 计算非空观测列表的均值；缺少观测时返回 None 以保留缺失含义。
# 输入：values（待统计或序列化值）。
def _rate(values):
    return mean(values) if values else None


# 从短长表的配对差值与样本数计算长表增益减短表增益。
# 输入：clusters。
def _length_gain(clusters):
    sums = [sum(v[i] for v in clusters) for i in range(4)]
    return sums[2] / sums[3] - sums[0] / sums[1] if sums[1] and sums[3] else None


# 汇总 L0 至 L4 的全量及长短表准确率，按原表聚类重采样比较增益。
# 输入：samples（固定样本集合）；groups；score；repeats（重采样次数）；seed（随机种子）。
# 返回形式含：字段 input_limit、table、paired_table_bootstrap。
# 异常约束：The main comparison requires L0 through L4 and positive bootstrap repeats
def main_report(samples, groups, score=official_score, *, repeats=2000, seed=42):
    if set(groups) != {f"L{i}" for i in range(5)} or repeats < 1:
        raise ValueError("The main comparison requires L0 through L4 and positive bootstrap repeats")
    by_group, table = {}, []
    lengths = None
    for group, rows in sorted(groups.items()):
        data, outcomes = _cohort(samples, rows)
        tokens = {k: p["complete_input_tokens"] for k, p in outcomes.items()}
        if any(type(n) is not int or n < 1 for n in tokens.values()):
            raise ValueError("Expected positive pinned tokenizer lengths")
        if lengths is None:
            lengths = tokens
        if lengths != tokens or any(p["group"] != group or p["ablation"] != "none" for p in rows):
            raise ValueError("Main groups must use identical complete-table token lengths and no ablation")
        scores = {k: int(score(outcomes[k], s)) for k, s in data.items()}
        by_group[group] = scores
        short = [v for k, v in scores.items() if lengths[k] <= INPUT_LIMIT]
        long = [v for k, v in scores.items() if lengths[k] > INPUT_LIMIT]
        table.append({"group": group, "all_n": len(scores), "all_accuracy": mean(scores.values()),
                      "short_n": len(short), "short_accuracy": _rate(short),
                      "long_n": len(long), "long_accuracy": _rate(long)})
    differences = {}
    for group in ("L3", "L4"):
        cluster = defaultdict(lambda: [0, 0, 0, 0])
        for uid, sample in data.items():
            cell = cluster[sample["table_id"]]
            i = 2 if lengths[uid] > INPUT_LIMIT else 0
            cell[i] += by_group[group][uid] - by_group["L2"][uid]
            cell[i + 1] += 1
        cells = [cluster[k] for k in sorted(cluster)]
        estimate = _length_gain(cells)
        rng = Random(seed)
        draws = []
        if estimate is not None:
            for _ in range(repeats):
                draw = _length_gain(rng.choices(cells, k=len(cells)))
                if draw is not None:
                    draws.append(draw)
        support = [sum(c[i] > 0 for c in cells) for i in (1, 3)]
        ci = ([quantile(draws, .025), quantile(draws, .975)]
              if min(support) >= 2 and len(draws) >= .95 * repeats else None)
        differences[group] = {"reference": "L2", "long_minus_short_gain": estimate,
                              "ci95": ci, "table_clusters": len(cells), "short_clusters": support[0],
                              "long_clusters": support[1], "usable_draws": len(draws),
                              "supports_larger_long_gain": ci is not None and ci[0] > 0}
    return {"input_limit": INPUT_LIMIT, "table": table, "paired_table_bootstrap": differences}


# 核对单个处理组后汇总探查、查询、模型与总耗时，计入全部失败。
# 输入：samples（固定样本集合）；predictions（完整预测集合）；score。
# 返回形式含：字段 n、group、ablation、accuracy、seconds、probe_rate、repair_rate。
# 异常约束：Report one treatment and seed at a time
def cost_report(samples, predictions, score=official_score):
    data, outcomes = _cohort(samples, predictions)
    group = {(p["group"], p["ablation"]) for p in predictions}
    if len(group) != 1:
        raise ValueError("Report one treatment and seed at a time")
    metrics = [p["metrics"] for p in predictions]
    if any(m["cache_hits"] for m in metrics):
        raise ValueError("Timing comparison requires uncached model responses")
    time_fields = ("cpu_setup_seconds", "cpu_probe_seconds", "cpu_query_seconds",
                   "model_seconds", "total_seconds")
    times = {}
    for field in time_fields:
        values = [m[field] for m in metrics]
        active = [v for v in values if v > 0]
        times[field] = {"sum": sum(values), "p50": quantile(values, .5), "p95": quantile(values, .95),
                        "active_n": len(active), "active_p50": quantile(active, .5),
                        "active_p95": quantile(active, .95)}
    elapsed = times["total_seconds"]["sum"]
    return {"n": len(outcomes), "group": predictions[0]["group"], "ablation": predictions[0]["ablation"],
            "accuracy": mean(score(outcomes[k], s) for k, s in data.items()), "seconds": times,
            "probe_rate": mean(m["probes"] > 0 for m in metrics),
            "repair_rate": mean(m["repairs"] > 0 for m in metrics),
            "mean_calls": mean(m["model_calls"] for m in metrics),
            "cpu_scan_fraction": (times["cpu_probe_seconds"]["sum"] + times["cpu_query_seconds"]["sum"]) /
                elapsed if elapsed else None,
            "cache_states": sorted({m["cache_state"] for m in metrics}),
            "table_rows": {p: quantile([m["table_rows"] for m in metrics], p) for p in (0, .5, .95, 1)},
            **{key: sum(m[key] for m in metrics) if all(m[key] is not None for m in metrics) else None
               for key in ("input_tokens", "output_tokens")}}


# 统一 Unicode、大小写、空白与末尾标点，用于配对题面的重复检查。
# 输入：question（原题文本）。
def normalized_question(question):
    text = unicodedata.normalize("NFKC", question).casefold()
    return " ".join(text.split()).rstrip(" ?!.,;:。！？；：")


# 按官方划分排除训练和已用开发表的 ID、指纹及重复题，返回可用来源。
# 输入：train；held_out；phase；dev。
# 异常约束：Provide nonempty training and held-out manifests and an explicit phase
def validate_pair_sources(train, held_out, phase, dev=()):
    expected = {"dev": "validation", "test": "test"}.get(phase)
    if not train or not held_out or expected is None:
        raise ValueError("Provide nonempty training and held-out manifests and an explicit phase")
    if phase == "test" and not dev:
        raise ValueError("Formal pairs also require the development manifest")
    for cohort, split in ((train, "train"), (held_out, expected), (dev, "validation")):
        id_index(cohort)
        if any(r["dataset"] != "databench" or r["split"] != split for r in cohort):
            raise ValueError(f"Expected official DataBench {split} cohort")
    forbidden = [*train, *dev]
    # 原表身份、内容指纹及题面重复分别检查；同表改写题会被内容层直接排除。
    excluded_ids = {r["table_id"] for r in forbidden}
    excluded_hashes = {r["table_fingerprint"] for r in forbidden}
    excluded_questions = {normalized_question(r["question"]) for r in forbidden}
    return {r["id"]: r for r in held_out if r["table_id"] not in excluded_ids
            and r["table_fingerprint"] not in excluded_hashes
            and normalized_question(r["question"]) not in excluded_questions}


# 核验人工批准的配对哈希，保证每对只改变规定的视图、隐藏数据或题意范围。
# 输入：base；pair。
# 相关调用：digest。
# 异常约束：Pair variants require distinct IDs and the original answer type
def validate_pair(base, pair):
    """Check both variants against their held-out source before model inference."""
    a, b = pair["a"], pair["b"]
    kind = pair["kind"]
    if a["id"] == b["id"] or any(s.get("semantic") != base["semantic"] for s in (a, b)):
        raise ValueError("Pair variants require distinct IDs and the original answer type")
    reviewed = pair["review"]
    review_payload = {"kind": kind, "source_id": base["id"], "a": a, "b": b}
    if (reviewed.get("answers_and_scope_verified") is not True
            or reviewed.get("pair_hash") != digest(review_payload)):
        raise ValueError("Both answers and allowed scopes need hash-bound human review")
    if (a["question"], a["table"], a["reference"]) != (base["question"], base["table"], base["reference"]):
        raise ValueError("First variant must use the held-out question and original table")
    av, bv = a["view"], b["view"]
    for sample, view in ((a, av), (b, bv)):
        table = sample["table"]
        if view.get("coverage") != "sample" or len(view["columns"]) > 8 or len(view["rows"]) > 16:
            raise ValueError("Invalid diagnostic view")
        if not view["columns"] or any(c not in {f"c{i}" for i in range(len(table["headers"]))}
                                      for c in view["columns"]):
            raise ValueError("Invalid diagnostic view columns")
        for row in view["rows"]:
            row_id = row["id"]
            if not row_id.startswith("r") or not row_id[1:].isdigit():
                raise ValueError("Diagnostic row IDs must resolve to original table coordinates")
            index = int(row_id[1:])
            if index >= len(table["rows"]) or row["values"] != [
                    table["rows"][index][int(c[1:])] for c in view["columns"]]:
                raise ValueError("Diagnostic view is inconsistent with its original table")
    same_table, same_question = a["table"] == b["table"], a["question"] == b["question"]
    if kind == "view_swap":
        valid = same_table and same_question and av != bv and av["columns"] == bv["columns"] and a["reference"] == b["reference"]
    elif kind == "hidden_answer":
        valid = (not same_table and same_question and av == bv and a["reference"] != b["reference"]
                 and a["table"]["headers"] == b["table"]["headers"]
                 and len(a["table"]["rows"]) == len(b["table"]["rows"])
                 and all(a["table"]["rows"][i] == b["table"]["rows"][i]
                         for i in (int(row["id"][1:]) for row in av["rows"])))
    elif kind == "scope_swap":
        valid = same_table and not same_question and av == bv
    else:
        raise ValueError("Unknown diagnostic kind")
    if not valid:
        raise ValueError(f"Pair does not isolate {kind}")


# 对两侧首次计划做规则开关重放，并计算答案稳定性及成对正确性。
# 输入：source（原始来源或原链）；pair；predictions（完整预测集合）；score。
# 相关调用：digest。
# 返回形式含：字段 kind、source_id、correct、both_correct、stable、fixed_first_plan_replay、first_plan_results。
# 异常约束：Diagnostic inference must use the frozen view
def paired_diagnostic(source, pair, predictions, score=official_score):
    """Score both variants and replay the actual first plans on their original tables."""
    base = source[pair["source_id"]]
    validate_pair(base, pair)
    a, b, kind, reviewed = pair["a"], pair["b"], pair["kind"], pair["review"]
    samples, outcomes = _cohort([a, b], predictions)
    if any(outcomes[k]["view"] != s["view"] for k, s in samples.items()):
        raise ValueError("Diagnostic inference must use the frozen view")
    first_plan_results = []
    for key, sample in samples.items():
        row = outcomes[key]
        plan = row["first_plan"]
        if plan is None:
            continue
        probe_spec = next((parse(t["response"]["text"])["probe"] for t in row["trace"]
                           if t.get("action") == "probe" and t["accepted"]), None)
        audit_key = digest(plan)
        first_plan_results.append({
            "id": key, "with_scope": replay_plan(sample, plan, probe_spec=probe_spec),
            "without_scope": replay_plan(sample, plan, enforce_scope=False, probe_spec=probe_spec),
            "scope_legal": reviewed.get("scope_legal", {}).get(key, {}).get(audit_key),
            "unsupported_enum": reviewed.get("unsupported_enum", {}).get(key, {}).get(audit_key),
        })
    correctness = [bool(score(outcomes[k], s)) for k, s in samples.items()]
    row_a, row_b = outcomes[a["id"]], outcomes[b["id"]]
    stable = row_a["status"] == row_b["status"] == "ok" and score(row_b, {**b, "reference": row_a["answer"]})
    fixed = None
    if kind == "hidden_answer" and row_a["first_plan"] is not None:
        probe = next((parse(t["response"]["text"])["probe"] for t in row_a["trace"]
                      if t.get("action") == "probe" and t["accepted"]), None)
        fixed = [replay_plan(s, row_a["first_plan"], probe_spec=probe) for s in (a, b)]
    return {"kind": kind, "source_id": base["id"], "correct": correctness, "both_correct": all(correctness),
            "stable": bool(stable) if kind == "view_swap" else None,
            "fixed_first_plan_replay": fixed, "first_plan_results": first_plan_results}


# 按配对类型汇总答案、首次枚举、人工范围裁决与规则拦截的不同指标。
# 输入：rows。
# 返回形式含：result。
def diagnostic_summary(rows):
    result = {}
    for kind in sorted({r["kind"] for r in rows}):
        subset = [r for r in rows if r["kind"] == kind]
        candidates = [p for r in subset for p in r["first_plan_results"]]
        blocked = lambda p: p["with_scope"]["error"] in {"UNSUPPORTED_ENUM_FILTER", "UNSUPPORTED_FILTER"}
        known = [p for p in candidates if type(p["scope_legal"]) is bool]
        result[kind] = {
            "pairs": len(subset), "pair_accuracy": mean(r["both_correct"] for r in subset),
            "answer_accuracy": mean(v for r in subset for v in r["correct"]),
            "answer_stability": _rate([r["stable"] for r in subset if r["stable"] is not None]),
            "first_plans": len(candidates), "human_reviewed_first_plans": len(known),
            "first_plan_unsupported_enum_rate": _rate([p["unsupported_enum"] for p in candidates
                                                      if type(p["unsupported_enum"]) is bool]),
            "first_plan_enum_rejection_rate": _rate([p["with_scope"]["error"] == "UNSUPPORTED_ENUM_FILTER" for p in candidates]),
            "range_error_interception_rate": _rate([blocked(p) for p in known if not p["scope_legal"]]),
            "legal_filter_false_reject_rate": _rate([blocked(p) for p in known if p["scope_legal"]]),
        }
    return result


# 从训练题重放教师原子，逐步核对学生可见提示，再生成目标 JSON。
# 输入：sample（规范样本记录）；prediction（单条预测或教师轨迹）；tokenizer（学生分词器）；protocol（冻结实验协议）；score。
# 返回形式含：列表记录。
# 异常约束：Chapter 4 supervision is restricted to DataBench training questions
def supervision(sample, prediction, tokenizer, protocol, score=official_score):
    if sample["split"] != "train" or sample["dataset"] != "databench":
        raise ValueError("Chapter 4 supervision is restricted to DataBench training questions")
    if protocol.group == "L1":
        raise ValueError("L1 is the zero-shot base model")
    if protocol.group == "L0":
        responses = [{"answer": sample["reference"]}]
    else:
        responses = [t["response"]["text"] for t in prediction["trace"]]
    replayed = predict_longtable(sample, ReplayModel(responses), tokenizer, protocol)
    # 重放重新生成每一步消息，防止教师使用学生当时尚未探查到的私有值。
    if not score(replayed, sample):
        raise ValueError("Teacher trajectory must replay and match the answer")
    if protocol.group != "L0":
        original = prediction["trace"]
        fresh = replayed["trace"]
        if len(original) != len(fresh) or any(a["messages"] != b["messages"] for a, b in zip(original, fresh)):
            raise ValueError("Teacher target depends on a different student-visible state")
    return [{"id": sample["id"], "dataset": "databench", "split": "train",
             "kind": "block" if t["phase"] in {"view", "plan"} else "answer", "stage": t["phase"],
             "messages": t["messages"], "target": dump(parse(t["response"]["text"]))}
            for t in replayed["trace"] if t["accepted"] and not (
                protocol.ablation == "no_repair_training" and json.loads(t["messages"][-1]["content"]).get("feedback"))]


# 把表单元格变成可序列化的值，保留合法零值并明确处理非有限数。
# 输入：value。
# 返回形式含：None / value。
def normalize_cell(value):
    if value is None or isinstance(value, float) and not math.isfinite(value):
        return None
    if type(value) in {bool, int, float, str}:
        return value
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


# 从完整 Parquet 表恢复列、类型和行，供原表执行器及指纹计算复用。
# 输入：path（输入或输出路径）。
# 返回形式含：字段 headers、types、rows。
def load_parquet_table(path):
    import pyarrow.parquet as pq
    import pyarrow.types as types

    table = pq.read_table(path)
    headers = table.column_names
    schema = ["boolean" if types.is_boolean(f.type) else
              "number" if types.is_integer(f.type) or types.is_floating(f.type) else "text" for f in table.schema]
    rows = [[normalize_cell(c) for c in row] for row in zip(*(column.to_pylist() for column in table.columns))]
    return {"headers": headers, "types": schema, "rows": rows}


# 读取指定原表并核对清单指纹，发现内容变化时终止。
# 输入：sample（规范样本记录）；tables。
# 返回形式含：字段 table。
def materialize(sample, tables):
    table = load_parquet_table(original_path(tables, sample["table_id"]))
    if table_fingerprint(table) != sample["table_fingerprint"]:
        raise ValueError(f"Original table changed for {sample['id']}")
    return {**sample, "table": table}


# 限制表 ID 为单个安全目录名，再构造 all.parquet 路径。
# 输入：tables；table_id。
# 异常约束：Invalid original DataBench table ID
def original_path(tables, table_id):
    if not isinstance(table_id, str) or not table_id or any(c in table_id for c in "/\\:") or table_id in {".", ".."}:
        raise ValueError("Invalid original DataBench table ID")
    return tables / table_id / "all.parquet"
