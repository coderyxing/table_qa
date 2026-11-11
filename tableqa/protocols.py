# 模块职责：实现训练预算、经审核的语义扰动、机制摘要和离线替换诊断等实验协议。
# 接口约定：样本、操作和结果采用本项目的结构化字典；细节见各入口及字段检查。
"""Auditable perturbations, training budget control, annotation and repairs."""
from copy import deepcopy
import random
from .data import digest


# 只执行已批准的语义扰动，并清理变换后失效的坐标注解。
# 输入：sample（规范样本记录）；kind；seed（随机种子）；approved。
# 相关调用：digest。
# 返回形式含：result。
# 异常约束：Perturbation requires explicit semantic-preservation review
def perturb(sample, kind, seed=42, approved=False):
    """Require semantic review: order-sensitive questions must not be shuffled."""
    if not approved: raise ValueError("Perturbation requires explicit semantic-preservation review")
    result=deepcopy(sample); table=result["table"]
    rng=random.Random(f"{seed}:{sample['dataset']}:{sample['id']}:{kind}")
    if kind=="row_shuffle": rng.shuffle(table["rows"])
    elif kind=="column_shuffle":
        order=list(range(len(table["headers"]))); rng.shuffle(order)
        table["headers"]=[table["headers"][i] for i in order]
        table["rows"]=[[row[i] for i in order] for row in table["rows"]]
    elif kind=="irrelevant_column":
        name="__reviewed_irrelevant_control__"
        if name in table["headers"]: raise ValueError("Control column already exists")
        table["headers"].append(name)
        for row in table["rows"]: row.append("control")
    else: raise ValueError(kind)
    # Original gold coordinates cannot be reused after a coordinate permutation.
    result.pop("highlighted_cells",None)
    result["perturbation"]={"kind":kind,"seed":seed,"source_table_hash":digest(sample["table"]),
                            "semantic_review_approved":True}
    return result


# 按实际目标 token 选择完整训练记录，记录预算剩余量。
# 输入：records（规范记录序列）；tokenizer（学生分词器）；target_tokens（目标监督 token 预算）；seed（随机种子）。
# 返回形式含：2 项组成的结果。
# 异常约束：Budget uses training records only
def budget_records(records, tokenizer, target_tokens=None, seed=42):
    """Never split a supervised target. Count EOS; report residual exact budget."""
    if any(r["split"]!="train" for r in records): raise ValueError("Budget uses training records only")
    items=list(records); random.Random(seed).shuffle(items)
    out=[]; used=0
    for r in items:
        n=len(tokenizer.encode(r["target"],add_special_tokens=False))+1
        if target_tokens is not None and used+n>target_tokens: continue
        out.append(r); used+=n
    return out,{"supervised_target_tokens":used,"requested_budget":target_tokens,
               "residual":None if target_tokens is None else target_tokens-used,
               "records":len(out),"seed":seed}


# 统计错误消息、首次观测失败位置和操作分布。
# 输入：predictions（完整预测集合）。
# 返回形式含：字段 error_messages、first_failure_step、operation_counts、note。
def mechanism_report(predictions):
    from collections import Counter
    errors=Counter(); first=Counter(); kinds=Counter()
    for p in predictions:
        for e in p.get("errors",[]): errors[e["message"]]+=1
        if p.get("errors"): first[str(p["errors"][0]["step"])]+=1
        for op in p.get("operations",[]): kinds[op["op"]]+=1
    return {"error_messages":dict(errors),"first_failure_step":dict(first),"operation_counts":dict(kinds),
            "note":"First failure is observed execution/format error, not necessarily first semantic error."}


# 只重放经人工确认的单步替换，保存离线诊断结果。
# 输入：sample（规范样本记录）；operations（白名单操作序列）；index；replacement；reviewed。
# 相关调用：run_plan。
# 返回形式含：字段 id、dataset、diagnostic_only、replacement_index、answer、trace、eligible_for_main_metrics。
# 异常约束：Counterfactual replacement needs human confirmation
def replacement_diagnostic(sample, operations, index, replacement, reviewed=False):
    from .core import TableState,run_plan
    if not reviewed: raise ValueError("Counterfactual replacement needs human confirmation")
    if not 0<=index<len(operations): raise ValueError("Invalid replacement index")
    modified=deepcopy(operations); modified[index]=replacement
    result,trace=run_plan(TableState.from_table(sample["table"]),modified)
    return {"id":sample["id"],"dataset":sample["dataset"],"diagnostic_only":True,
            "replacement_index":index,"answer":result.answer,"trace":trace,
            "eligible_for_main_metrics":False}
