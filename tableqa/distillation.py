# 模块职责：重放教师链、保守删步、切分操作块，并生成与线上提示一致的学生监督记录。
# 接口约定：样本、操作和结果采用本项目的结构化字典；细节见各入口及字段检查。
"""Conservative, replay-verified rewrites and explicit feedback boundaries."""
from copy import deepcopy
from .core import ExecutionError, run_plan, TableState


# 尝试有限删步与合并规则，每个候选都须在当前原表上保持完整执行签名。
# 输入：state（当前执行状态）；operations（白名单操作序列）。
# 相关调用：run_plan。
# 返回形式含：result。
def compress(state: TableState, operations: list[dict]) -> list[dict]:
    original, _ = run_plan(state, operations)
    result = deepcopy(operations)
    required_feedback=[op for op in operations if op.get('requires_feedback')]
    changed = True
    while changed:
        changed = False
        for i, op in enumerate(result):
            candidates = []
            before, _ = run_plan(state, result[:i])
            if op["op"] == "project" and op["columns"] == before.columns:
                candidates.append(result[:i] + result[i+1:])
            if i + 1 < len(result):
                nxt = result[i+1]
                if op["op"] == nxt["op"] == "project":
                    candidates.append(result[:i] + [nxt] + result[i+2:])
                if op["op"] == nxt["op"] == "sort" and op == nxt:
                    candidates.append(result[:i] + [op] + result[i+2:])
            for candidate in candidates:
                if [op for op in candidate if op.get('requires_feedback')]!=required_feedback:continue
                try:
                    final, _ = run_plan(state, candidate)
                except ExecutionError:
                    continue
                if final.signature() == original.signature():
                    result = candidate; changed = True; break
            if changed: break
    return result


# 根据最大块长、答案和显式反馈需求切分操作序列。
# 输入：operations（白名单操作序列）；max_size（最大操作块长）。
# 返回形式含：result。
# 异常约束：Block size must be positive
def blocks(operations: list[dict], max_size=4) -> list[list[dict]]:
    if max_size < 1: raise ValueError("Block size must be positive")
    result, current = [], []
    for op in operations:
        # Feedback boundaries are explicitly annotated by teacher/human and
        # retained in traces. They are not inferred from mere read/write overlap.
        if op.get("requires_feedback", False) and current:
            result.append(current); current = []
        current.append(op)
        if len(current) == max_size or op["op"] == "answer":
            result.append(current); current = []
    if current: result.append(current)
    return result


# 检查允许划分与审核状态，重放链并生成与线上状态一致的提示和目标。
# 输入：sample（规范样本记录）；operations（白名单操作序列）；mode（实验模式）；max_size（最大操作块长）；allow_validation；compression_policy；partition；prepared_operations；typed_quantities；semantic_extract；tool_demonstrations。
# 相关调用：run_plan。
# 返回形式含：records。
# 异常约束：Quantity supervision requires the explicit typed_quantities prompt
def build_records(sample, operations, mode="B5", max_size=4, allow_validation=False, *,
                  compression_policy='legacy', partition=None, prepared_operations=None,
                  typed_quantities=False, semantic_extract=False, tool_demonstrations=False):
    from .inference import context, system_prompt,compact_state
    SYSTEM=system_prompt(mode, typed_quantities, semantic_extract, tool_demonstrations)
    from .quantities import uses_quantities
    if mode not in {'B0', 'B1'} and uses_quantities(operations) and not typed_quantities:
        raise ValueError('Quantity supervision requires the explicit typed_quantities prompt')
    from .semantic_columns import uses_extraction
    if mode not in {'B0', 'B1'} and uses_extraction(operations) and not semantic_extract:
        raise ValueError('Extraction supervision requires the explicit semantic_extract prompt')
    permitted={"train", "validation"} if allow_validation else {"train"}
    if sample["split"] not in permitted: raise ValueError("Training records require train split; validation must be explicitly enabled")
    if sample['task']=='freeform' and mode not in {'B0','B1'}:
        from .review import review_approved
        if not review_approved(sample,operations=operations):
            raise ValueError('Freeform operation supervision requires a hash-bound dual human review')
    state = TableState.from_table(sample["table"])
    if compression_policy not in {'legacy','search','guarded_search'}:raise ValueError('Unknown compression policy')
    compressed_mode=mode in {"B3", "B4", "B5", "B5_no_blocks"}
    if prepared_operations is not None:
        if not compressed_mode:raise ValueError('Prepared rewrites require a compressed supervision mode')
        from .optimization import verify_subsequence_rewrite
        verify_subsequence_rewrite(state,operations,prepared_operations)
        if compression_policy=='guarded_search':
            protected={'filter','sort','select','group_aggregate','answer'}
            if [o for o in operations if o['op'] in protected]!=[o for o in prepared_operations if o['op'] in protected]:
                raise ValueError('Guarded compression removed a question-dependent operation')
        ops=deepcopy(prepared_operations)
    elif compressed_mode:
        if compression_policy=='legacy':ops=compress(state,operations)
        else:
            from .optimization import optimize_operations
            protect=('filter','sort','select','group_aggregate') if compression_policy=='guarded_search' else ()
            ops,_=optimize_operations(state,operations,preserve_kinds=protect)
    else:
        if compression_policy!='legacy':raise ValueError('Search compression requires a compressed supervision mode')
        ops=operations
    if partition is not None:
        if mode not in {'B4','B5'}:raise ValueError('Custom blocks require a block-supervised mode')
        if [op for group in partition for op in group]!=ops:raise ValueError('Partition changed the optimized operation chain')
        if any(not group or len(group)>max_size or any(op.get('requires_feedback') for op in group[1:]) for group in partition):
            raise ValueError('Partition violates size or feedback boundaries')
        grouped=deepcopy(partition)
    else:grouped = blocks(ops, max_size if mode in {"B4", "B5"} else 1)
    history, records, full_trace = [], [], []
    # 用当前公开快照构造下一步训练或推理的提示。
    def prompt():
        return context(sample, state, history if compact_state(mode) else full_trace,
                       mode, max_block=max_size if mode in {"B4", "B5"} else 1)
    if mode not in {"B0", "B1"}:
        for block in grouped:
            # Answer is trained separately, from the exact pre-answer state.
            pending = []
            for op in block:
                if op["op"] == "answer":
                    if pending:
                        records.append(_record(sample, "block", SYSTEM, prompt(), {"operations": pending}))
                        state, steps = run_plan(state, pending); full_trace += steps; history += pending; pending = []
                    records.append(_record(sample, "answer", SYSTEM, prompt(), {"operations": [op]}))
                    state, steps = run_plan(state, [op]); full_trace += steps; history.append(op)
                else: pending.append(op)
            if pending:
                records.append(_record(sample, "block", SYSTEM, prompt(), {"operations": pending}))
                state, steps = run_plan(state, pending); full_trace += steps; history += pending
    elif mode == "B0":
        records.append(_record(sample, "answer", SYSTEM, context(sample, state, [], mode), {"answer": sample["reference"]}))
    else:
        rationale = sample.get("teacher_rationale")
        if not rationale: raise ValueError("B1 requires separately generated teacher_rationale, not operation JSON relabelled as CoT")
        records.append(_record(sample, "answer", SYSTEM, context(sample, state, [], mode),
                               {"rationale": rationale, "answer": sample["reference"]}))
    return records


# 封装单条训练记录，保存样本身份、监督类型、消息与目标 JSON。
# 输入：sample（规范样本记录）；kind；system；prompt；target。
# 返回形式含：字段 id、dataset、split、kind、messages、target。
def _record(sample, kind, system, prompt, target):
    import json
    return {"id": sample["id"], "dataset": sample["dataset"], "split": sample["split"], "kind": kind,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "target": json.dumps(target, ensure_ascii=False)}
