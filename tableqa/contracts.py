"""Explicit, source-bound condition obligations; independent experimental API.

Checks only DECLARED obligations. A copied question span and a matching program
do not prove that conditions were extracted completely or understood correctly.
This module is not enabled by the curriculum pilot or default inference.
"""
from dataclasses import dataclass, replace
import json

from .core import ExecutionError, reads, writes, run_plan, validate_plan


class ContractError(ExecutionError):
    pass


def fingerprint(operation):
    # Feedback annotations are separately enforced by block/partition policies.
    value={k:v for k,v in operation.items() if k!='requires_feedback'}
    try:
        text=json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False)
    except (TypeError,ValueError) as exc:
        raise ContractError('Condition operation must be finite JSON data') from exc
    if len(text)>16384:raise ContractError('Condition operation exceeds the supported bound')
    return text


@dataclass(frozen=True)
class Condition:
    key: str
    question_start: int
    question_end: int
    quote: str
    operation_json: str
    required_columns: frozenset
    after: frozenset


@dataclass(frozen=True)
class ObligationTracker:
    conditions: tuple
    fulfilled: frozenset = frozenset()

    def __post_init__(self):
        if not isinstance(self.conditions,tuple) or not 1<=len(self.conditions)<=64 or any(not isinstance(c,Condition) for c in self.conditions):
            raise ContractError('Tracker requires a bounded tuple of declared conditions')
        keys={c.key for c in self.conditions}
        if len(keys)!=len(self.conditions) or not isinstance(self.fulfilled,frozenset) or not self.fulfilled<=keys:
            raise ContractError('Tracker fulfilled IDs must belong to its unique declared conditions')

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
        def visit(key,active,done):
            if key in active:raise ContractError('Cyclic condition dependency')
            if key in done:return
            if key not in by_id:raise ContractError('Unknown condition predecessor')
            for parent in by_id[key].after:visit(parent,active|{key},done)
            done.add(key)
        done=set()
        for key in by_id:visit(key,set(),done)
        return cls(tuple(conditions))

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

    def report(self):
        declared={c.key for c in self.conditions}
        return {'declared_conditions':len(declared),'fulfilled_conditions':len(self.fulfilled),
                'pending_ids':sorted(declared-self.fulfilled),
                'declared_condition_execution_coverage':len(self.fulfilled)/len(declared),
                'question_condition_completeness':'not_proven','semantic_correctness':'not_proven'}


def input_row_ids(state):
    if all(rid.startswith("r") for rid in state.row_ids):
        return set(state.row_ids)
    return {cell.split(":", 1)[0] for row in state.lineage.values()
            for cells in row.values() for cell in cells}


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
