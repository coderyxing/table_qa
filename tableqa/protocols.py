"""Auditable perturbations, training budget control, annotation and repairs."""
from copy import deepcopy
import random
from .data import digest


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


def mechanism_report(predictions):
    from collections import Counter
    errors=Counter(); first=Counter(); kinds=Counter()
    for p in predictions:
        for e in p.get("errors",[]): errors[e["message"]]+=1
        if p.get("errors"): first[str(p["errors"][0]["step"])]+=1
        for op in p.get("operations",[]): kinds[op["op"]]+=1
    return {"error_messages":dict(errors),"first_failure_step":dict(first),"operation_counts":dict(kinds),
            "note":"First failure is observed execution/format error, not necessarily first semantic error."}


def replacement_diagnostic(sample, operations, index, replacement, reviewed=False):
    from .core import TableState,run_plan
    if not reviewed: raise ValueError("Counterfactual replacement needs human confirmation")
    if not 0<=index<len(operations): raise ValueError("Invalid replacement index")
    modified=deepcopy(operations); modified[index]=replacement
    result,trace=run_plan(TableState.from_table(sample["table"]),modified)
    return {"id":sample["id"],"dataset":sample["dataset"],"diagnostic_only":True,
            "replacement_index":index,"answer":result.answer,"trace":trace,
            "eligible_for_main_metrics":False}
