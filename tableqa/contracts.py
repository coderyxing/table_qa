# 模块职责：把题面原文绑定为显式操作义务，在执行中保护待满足条件及其输入行域。
# 接口约定：样本、操作和结果采用本项目的结构化字典；细节见各入口及字段检查。
"""Explicit, source-bound condition obligations; independent experimental API.

Checks only DECLARED obligations. A copied question span and a matching program
do not prove that conditions were extracted completely or understood correctly.
This module is not enabled by the curriculum pilot or default inference.
"""
from dataclasses import dataclass, replace
import json

from .core import ExecutionError, reads, writes, run_plan, validate_plan


# ContractError：条件或行域校验错误；保留原始状态与拒绝原因。
class ContractError(ExecutionError):
    pass


# 规范化操作 JSON，独立处理反馈标记，使条件匹配具有确定性。
# 输入：operation（单个操作参数）。
# 返回形式含：text。
# 异常约束：Condition operation exceeds the supported bound
def fingerprint(operation):
    # Feedback annotations are separately enforced by block/partition policies.
    value={k:v for k,v in operation.items() if k!='requires_feedback'}
    # 按执行器真实默认值补齐指纹，避免把省略 type 的文本筛选误认为另一条操作。
    # 只统一已证明等价的缺省写法，显式 number、比较符和数值策略仍严格区分。
    if value.get('op') in {'filter','sort'}:
        value.setdefault('type','text')
    if value.get('op')=='filter':
        value.setdefault('invalid_number','error')
    if value.get('op')=='sort':
        value.setdefault('descending',False)
    try:
        text=json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False)
    except (TypeError,ValueError) as exc:
        raise ContractError('Condition operation must be finite JSON data') from exc
    if len(text)>16384:raise ContractError('Condition operation exceeds the supported bound')
    return text


# Condition：不可变的题面跨度、操作指纹、必需列及前置条件记录。
@dataclass(frozen=True)
class Condition:
    # 条件 ID 在同一题内唯一；跨度按 Python 字符索引，结束位置不包含在引文中。
    key: str
    question_start: int
    question_end: int
    quote: str
    # 操作以规范 JSON 指纹绑定；该绑定本身不证明题意抽取完整。
    operation_json: str
    required_columns: frozenset
    after: frozenset


# ObligationTracker：不可变的条件履行集合；每次成功操作返回新的跟踪器。
@dataclass(frozen=True)
class ObligationTracker:
    # conditions 是冻结的声明，fulfilled 只记录已经成功执行满足的条件 ID。
    conditions: tuple
    fulfilled: frozenset = frozenset()

    # 构造后立即检查字段和允许组合，使无效配置尽早失败。
    # 异常约束：Tracker requires a bounded tuple of declared conditions
    def __post_init__(self):
        if not isinstance(self.conditions,tuple) or not 1<=len(self.conditions)<=64 or any(not isinstance(c,Condition) for c in self.conditions):
            raise ContractError('Tracker requires a bounded tuple of declared conditions')
        keys={c.key for c in self.conditions}
        if len(keys)!=len(self.conditions) or not isinstance(self.fulfilled,frozenset) or not self.fulfilled<=keys:
            raise ContractError('Tracker fulfilled IDs must belong to its unique declared conditions')

    # 核验声明字段、原文跨度及前置关系，建立新的条件跟踪器。
    # 输入：question（原题文本）；contracts。
    # 异常约束：A question and 1..64 declared conditions are required
    @classmethod
    def from_question(cls,question,contracts):
        if not isinstance(question,str) or not isinstance(contracts,list) or not 1<=len(contracts)<=64:
            raise ContractError('A question and 1..64 declared conditions are required')
        conditions=[];keys=set()
        for item in contracts:
            if not isinstance(item,dict) or set(item)-{'id','question_span','quote','operation','after'}:
                raise ContractError('Unexpected condition fields; no references or labels are accepted')
            key=item.get('id');span=item.get('question_span');quote=item.get('quote')
            if not isinstance(key,str) or not key or len(key)>64 or key in keys:
                raise ContractError('Condition IDs must be bounded and unique')
            if not isinstance(span,list) or len(span)!=2 or any(type(x) is not int for x in span):
                raise ContractError('Condition requires two integer question offsets')
            start,end=span
            if not 0<=start<end<=len(question) or not isinstance(quote,str) or not quote.strip() or question[start:end]!=quote:
                raise ContractError('Condition quote does not match its question span')
            operation=item.get('operation')
            if not isinstance(operation,dict) or operation.get('op') not in {'filter','project','sort','select','derive','group_aggregate','answer'}:
                raise ContractError('Condition must bind a whitelisted operation')
            if operation.get('op')=='answer' and 'value' in operation:
                raise ContractError('Literal final answers are not condition obligations')
            try:
                required=reads(operation)
            except (ExecutionError,KeyError,TypeError,ValueError) as exc:
                raise ContractError('Invalid declared operation read footprint') from exc
            if any(not isinstance(c,str) or not c for c in required):
                raise ContractError('Required columns must have explicit IDs')
            after=item.get('after',[])
            if not isinstance(after,list) or any(not isinstance(a,str) for a in after) or len(set(after))!=len(after):
                raise ContractError('Condition predecessors must be unique explicit IDs')
            conditions.append(Condition(key,start,end,quote,fingerprint(operation),frozenset(required),frozenset(after)))
            keys.add(key)
        by_id={c.key:c for c in conditions}
        # 遍历条件前置关系，检查未知引用和依赖环。
        # 输入：key；active；done。
        # 异常约束：Cyclic condition dependency
        def visit(key,active,done):
            if key in active:raise ContractError('Cyclic condition dependency')
            if key in done:return
            if key not in by_id:raise ContractError('Unknown condition predecessor')
            for parent in by_id[key].after:visit(parent,active|{key},done)
            done.add(key)
        done=set()
        for key in by_id:visit(key,set(),done)
        return cls(tuple(conditions))

    # 在执行前检查前置条件、待用列和终点覆盖，并返回下一跟踪状态。
    # 输入：state（当前执行状态）；operation（单个操作参数）。
    # 返回形式含：2 项组成的结果。
    def before_operation(self,state,operation):
        signature=fingerprint(operation)
        matched={c.key for c in self.conditions if c.key not in self.fulfilled and c.operation_json==signature}
        for condition in self.conditions:
            if condition.key in matched and not condition.after<=self.fulfilled:
                raise ContractError(f'Condition {condition.key} executes before its declared predecessors')
        pending=[c for c in self.conditions if c.key not in self.fulfilled|matched]
        kind=operation.get('op');lost=set()
        if kind=='project':lost=set(state.columns)-set(operation['columns'])
        elif kind=='group_aggregate':
            retained=set(operation.get('by',[]))|writes(operation)
            lost=(set(state.columns)-retained)|(writes(operation)&set(state.columns))
        endangered={c.key for c in pending if c.required_columns&lost}
        if endangered:raise ContractError(f'Operation removes columns required by pending conditions: {sorted(endangered)}')
        if kind=='answer' and pending:
            raise ContractError(f'Answer omits declared conditions: {sorted(c.key for c in pending)}')
        return replace(self,fulfilled=self.fulfilled|frozenset(matched)),matched

    # 导出当前声明履行与缺失情况，保留未被自动证明的语义状态。
    # 返回形式含：字段 declared_conditions、fulfilled_conditions、pending_ids、declared_condition_execution_coverage、question_condition_completeness、semantic_correctness。
    def report(self):
        declared={c.key for c in self.conditions}
        return {'declared_conditions':len(declared),'fulfilled_conditions':len(self.fulfilled),
                'pending_ids':sorted(declared-self.fulfilled),
                'declared_condition_execution_coverage':len(self.fulfilled)/len(declared),
                'question_condition_completeness':'not_proven','semantic_correctness':'not_proven'}


# 将当前普通行或聚合行的来源展开为原表行 ID 集合。
# 输入：state（当前执行状态）。
def input_row_ids(state):
    if all(rid.startswith("r") for rid in state.row_ids):
        return set(state.row_ids)
    return {cell.split(":", 1)[0] for row in state.lineage.values()
            for cells in row.values() for cell in cells}


# 在候选状态上同时推进条件义务与操作执行，整块通过后才返回可提交状态。
# 输入：state（当前执行状态）；operations（白名单操作序列）；tracker（条件履行状态）；evidence；row_domains。
# 相关调用：run_plan。
# 返回形式含：3 项组成的结果。
def run_contracted_block(state,operations,tracker,*,evidence=True,row_domains=None):
    """Atomic state AND obligation update; no oracle future plan or answer used."""
    validate_plan(state,operations)
    current=state;working=tracker;trace=[]
    for index,operation in enumerate(operations):
        try:
            next_tracker,matched=working.before_operation(current,operation)
            actual_rows = input_row_ids(current)
            for key in matched:
                if key in (row_domains or {}) and actual_rows != row_domains[key]:
                    raise ContractError(f"Condition {key} has an incorrect input row domain")
            current,steps=run_plan(current,[operation],dependency=False,evidence=evidence)
        except ExecutionError as exc:
            exc.operation_index=index
            raise
        working=next_tracker
        steps[0]['condition_obligations']={'satisfied_here':sorted(matched),**working.report()}
        if row_domains is not None:
            steps[0]['input_row_ids'] = sorted(actual_rows)
        trace+=steps
    return current,working,trace


# 分别重放原链和候选链，核对本表状态、证据和声明义务是否保留。
# 输入：state（当前执行状态）；source（原始来源或原链）；candidate（待核验候选）；tracker（条件履行状态）。
# 返回形式含：字段 signature_equal_on_this_table、declared_conditions_preserved、universal_equivalence_proven、question_condition_completeness。
# 异常约束：Rewrite changed an explicit feedback operation
def verify_condition_rewrite(state,source,candidate,tracker):
    """Additional offline rewrite check; not a universal semantic equivalence proof."""
    if [op for op in source if op.get('requires_feedback')]!=[op for op in candidate if op.get('requires_feedback')]:
        raise ContractError('Rewrite changed an explicit feedback operation')
    original,source_tracker,_=run_contracted_block(state,source,tracker)
    rewritten,candidate_tracker,_=run_contracted_block(state,candidate,tracker)
    if source_tracker.report()['pending_ids'] or candidate_tracker.report()['pending_ids']:
        raise ContractError('Complete rewrite comparison requires all declared obligations')
    if original.signature()!=rewritten.signature():raise ContractError('Rewrite changed this-table state or provenance')
    return {'signature_equal_on_this_table':True,'declared_conditions_preserved':True,
            'universal_equivalence_proven':False,'question_condition_completeness':'not_proven'}
