"""Typed table state, safe execution, dependency checks and cell lineage.

Cell IDs are original rN:cN coordinates. `lineage` records value contributors;
`evidence` additionally records filtering/comparison/order dependencies. No eval.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import json
from typing import Any


class ExecutionError(ValueError):
    pass


def number(value: Any) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ExecutionError(f"Not a number: {value!r}")
    text = str(value).strip().replace(",", "")
    percent = text.endswith("%")
    try:
        n = Decimal(text[:-1] if percent else text)
    except InvalidOperation as exc:
        raise ExecutionError(f"Ambiguous numeric value: {value!r}") from exc
    if not n.is_finite():
        raise ExecutionError("Non-finite number")
    return n / 100 if percent else n


def scalar(n: Decimal) -> int | float:
    return int(n) if n == n.to_integral_value() else float(n)


def compare(a, b, cmp):
    if cmp == 'eq': return a == b
    if cmp == 'ne': return a != b
    if cmp == 'gt': return a > b
    if cmp == 'ge': return a >= b
    if cmp == 'lt': return a < b
    if cmp == 'le': return a <= b
    raise ExecutionError(f'Unknown comparator {cmp}')


@dataclass
class TableState:
    columns: list[str]
    rows: list[dict[str, Any]]
    row_ids: list[str]
    lineage: dict[str, dict[str, set[str]]]
    evidence: set[str] = field(default_factory=set)
    answer: Any = None
    answered: bool = False

    @classmethod
    def from_table(cls, table: dict) -> "TableState":
        headers = table["headers"]
        if not headers or not all(isinstance(h, str) for h in headers):
            raise ExecutionError("Headers must be nonempty strings")
        # Stable internal column IDs also support duplicate visible headers.
        columns = [f"c{i}" for i in range(len(headers))]
        rows, lineage, ids = [], {}, []
        for i, values in enumerate(table["rows"]):
            if len(values) != len(columns):
                raise ExecutionError(f"Ragged row {i}")
            rid = f"r{i}"
            ids.append(rid)
            rows.append(dict(zip(columns, values)))
            lineage[rid] = {c: {f"{rid}:{c}"} for c in columns}
        return cls(columns, rows, ids, lineage)

    def view(self, headers: list[str] | None = None) -> dict:
        result = {"columns": self.columns, "row_ids": self.row_ids,
                  "rows": self.rows, "answered": self.answered}
        if headers is not None:
            result["original_headers"] = {f"c{i}": h for i, h in enumerate(headers) if f"c{i}" in self.columns}
        return result

    def signature(self) -> str:
        # Ignore derived row IDs; include cell lineage and predicate evidence.
        return json.dumps({"columns": self.columns, "rows": self.rows,
                           "lineage": [ {c: sorted(self.lineage[r][c]) for c in self.columns}
                                       for r in self.row_ids],
                           "evidence": sorted(self.evidence), "answer": self.answer,
                           "answered": self.answered}, sort_keys=True, ensure_ascii=False)


def expression_columns(expr: Any) -> set[str]:
    if not isinstance(expr, dict):
        return set()
    if "col" in expr:
        return {expr["col"]}
    return set().union(*(expression_columns(x) for x in expr.get("args", [])))


def reads(op: dict) -> set[str]:
    kind = op.get("op")
    if kind == "project":
        return set(op["columns"])
    if kind in {"filter", "sort", "select"}:
        return {op["column"]} if "column" in op else set()
    if kind == "group_aggregate":
        return set(op.get("by", [])) | {a["column"] for a in op["aggregations"] if a.get("column")}
    if kind == "derive":
        return expression_columns(op["expression"])
    if kind == "answer":
        assertion=op.get('assert',{})
        return set(op.get("columns", [])) | ({assertion['column']} if assertion.get('kind')=='cell' else set()) | expression_columns(assertion.get('expression'))
    raise ExecutionError(f"Unknown operation {kind!r}")


def writes(op: dict) -> set[str]:
    if op["op"] == "derive":
        return {op["as"]}
    if op["op"] == "group_aggregate":
        return {a["as"] for a in op["aggregations"]}
    return set()


def validate_plan(state: TableState, operations: list[dict]) -> None:
    available = set(state.columns)
    for i, op in enumerate(operations):
        required = reads(op)
        if not required <= available:
            raise ExecutionError(f"Step {i}: unavailable columns {sorted(required - available)}")
        if op["op"] == "project":
            kept = set(op["columns"])
            # Account for columns that will be created later.
            created: set[str] = set()
            needed: set[str] = set()
            for later in operations[i + 1:]:
                needed |= reads(later) - created
                created |= writes(later)
            if (needed & available) - kept:
                raise ExecutionError(f"Projection removes future dependency: {(needed & available) - kept}")
            available = kept
        elif op["op"] == "group_aggregate":
            available = set(op.get("by", [])) | writes(op)
        else:
            available |= writes(op)
        if op["op"] == "answer" and i != len(operations) - 1:
            raise ExecutionError("Answer must terminate a block")


def evaluate_expression(expr: Any, row: dict) -> Any:
    if not isinstance(expr, dict):
        return expr
    if set(expr) == {"col"}:
        return row[expr["col"]]
    if set(expr) == {"literal"}:
        return expr["literal"]
    raw_args=expr.get('args',[]); kind=expr.get('op')
    if kind == 'extract':
        from .semantic_columns import extract_from_row
        try:
            return extract_from_row(expr, row)['value']
        except ValueError as exc:
            raise ExecutionError(str(exc)) from exc
    if kind=='if':
        if len(raw_args)!=3: raise ExecutionError('if needs condition, true branch, false branch')
        condition=evaluate_expression(raw_args[0],row)
        if not isinstance(condition,bool): raise ExecutionError('if requires a boolean condition')
        return evaluate_expression(raw_args[1] if condition else raw_args[2],row)
    values=[evaluate_expression(x,row) for x in raw_args]
    if kind == 'quantity':
        from .quantities import quantity_text
        if len(values) != 1 or 'unit' not in expr:
            raise ExecutionError('Quantity needs one argument and an explicit target unit')
        try:
            return quantity_text(values[0], expr['unit'], expr.get('source_unit'))
        except ValueError as exc:
            raise ExecutionError(str(exc)) from exc
    if kind in {'eq','ne','gt','ge','lt','le'}:
        if len(values)!=2: raise ExecutionError('Comparison requires two arguments')
        if expr.get('type','number')=='number': a,b=map(number,values)
        elif expr.get('type')=='text': a,b=(str(v).casefold() for v in values)
        else: raise ExecutionError('Unknown expression comparison type')
        return compare(a,b,kind)
    if kind in {'and','or'}:
        if len(values)!=2 or not all(isinstance(v,bool) for v in values): raise ExecutionError('Boolean operation requires two booleans')
        return all(values) if kind=='and' else any(values)
    args = [number(v) for v in values]
    if len(args) != 2 or expr.get("op") not in {"add", "sub", "mul", "div"}:
        raise ExecutionError("Expression must be a binary whitelist arithmetic tree")
    a, b = args
    if expr["op"] == "div" and b == 0:
        raise ExecutionError("Division by zero")
    if expr["op"] == "add": result = a + b
    elif expr["op"] == "sub": result = a - b
    elif expr["op"] == "mul": result = a * b
    else: result = a / b
    return scalar(result)


def execute(state: TableState, op: dict, track_evidence: bool = True) -> TableState:
    s = deepcopy(state)  # Atomic operation; callers retain immutable snapshots.
    if s.answered:
        raise ExecutionError("Cannot operate after answer")
    kind = op.get("op")
    required = reads(op)
    if not required <= set(s.columns):
        raise ExecutionError(f"Unavailable columns: {required - set(s.columns)}")
    def contribute(columns: set[str]) -> None:
        if track_evidence:
            for rid in s.row_ids:
                for c in columns:
                    s.evidence.update(s.lineage[rid][c])
    def retain(indices: list[int]) -> None:
        s.rows = [s.rows[i] for i in indices]
        s.row_ids = [s.row_ids[i] for i in indices]
        s.lineage = {r: s.lineage[r] for r in s.row_ids}
    try:
        if kind == "project":
            cols = op["columns"]
            if not cols or len(cols) != len(set(cols)):
                raise ExecutionError("Projection must have unique nonempty columns")
            s.columns = list(cols)
            s.rows = [{c: row[c] for c in cols} for row in s.rows]
            s.lineage = {r: {c: s.lineage[r][c] for c in cols} for r in s.row_ids}
        elif kind == "filter":
            c, cmp = op["column"], op["cmp"]
            contribute({c})
            def match(row):
                a, b = row[c], op.get("value")
                if cmp == "is_null": return a is None or a == ""
                if cmp=='is_number':
                    try: number(a); return True
                    except ExecutionError: return False
                if cmp == "contains": return str(b).casefold() in str(a).casefold()
                if cmp in {'in','not_in'}:
                    if not isinstance(b,list): raise ExecutionError('Membership requires a list literal')
                    if op.get('type','text')=='number': a,items=number(a),[number(v) for v in b]
                    elif op.get('type','text')=='text': a,items=str(a).casefold(),[str(v).casefold() for v in b]
                    else: raise ExecutionError('Invalid membership type')
                    return (a in items) if cmp=='in' else (a not in items)
                if op.get("type", "text") == "number": a, b = number(a), number(b)
                elif op.get("type", "text") == "text": a, b = str(a).casefold(), str(b).casefold()
                else: raise ExecutionError("Supported filter types: text, number")
                if cmp == "eq": return a == b
                if cmp == "ne": return a != b
                if cmp == "gt": return a > b
                if cmp == "ge": return a >= b
                if cmp == "lt": return a < b
                if cmp == "le": return a <= b
                raise ExecutionError(f"Unknown comparator {cmp}")
            retain([i for i, row in enumerate(s.rows) if match(row)])
        elif kind == "sort":
            c = op["column"]
            contribute({c})
            value_type = op.get("type", "text")
            if value_type not in {"number", "text"}: raise ExecutionError("Unsupported sort type")
            key = (lambda i: number(s.rows[i][c])) if value_type == "number" else (lambda i: str(s.rows[i][c]))
            retain(sorted(range(len(s.rows)), key=key, reverse=op.get("descending", False)))
        elif kind == "select":
            mode = op.get("mode", "extreme")
            if mode in {"first", "last"}:
                n = op.get("n", 1)
                if not isinstance(n, int) or n < 0: raise ExecutionError("Invalid first n")
                retain(list(range(min(n, len(s.rows)))) if mode=='first' else list(range(max(0,len(s.rows)-n),len(s.rows))))
            elif mode == "extreme":
                c = op["column"]
                contribute({c})
                values = [number(row[c]) for row in s.rows]
                if values:
                    which = op.get("which", "max")
                    if which not in {"min", "max"}: raise ExecutionError("Invalid extreme")
                    target = min(values) if which == "min" else max(values)
                    retain([i for i, v in enumerate(values) if v == target])
            else: raise ExecutionError("Unsupported selection mode")
        elif kind == "derive":
            name = op["as"]
            if name in s.columns: raise ExecutionError("Derived column already exists")
            for rid, row in zip(s.row_ids, s.rows):
                row[name] = evaluate_expression(op["expression"], row)
                s.lineage[rid][name] = set().union(*(s.lineage[rid][c] for c in required))
            s.columns.append(name)
        elif kind == "group_aggregate":
            by, aggs = op.get("by", []), op["aggregations"]
            names = list(by) + [a["as"] for a in aggs]
            if len(names) != len(set(names)) or not aggs: raise ExecutionError("Invalid aggregate output names")
            contribute(required)
            groups: dict[str, list[int]] = {}
            for i, row in enumerate(s.rows):
                groups.setdefault(json.dumps([row[c] for c in by], ensure_ascii=False), []).append(i)
            if not s.rows and not by: groups["[]"] = []
            out_rows, out_ids, out_lineage = [], [], {}
            for k, indices in enumerate(groups.values()):
                rid = f"g{k}"
                row, lin = {}, {}
                for c in by:
                    row[c] = s.rows[indices[0]][c]
                    lin[c] = set().union(*(s.lineage[s.row_ids[i]][c] for i in indices))
                for agg in aggs:
                    fn, c, name = agg["fn"], agg.get("column"), agg["as"]
                    if fn == "count":
                        row[name] = len(indices)
                        # Count(*) depends on retained row identity, represented by its cells.
                        lin[name] = set().union(*(s.lineage[s.row_ids[i]][col]
                                                 for i in indices for col in ( [c] if c else s.columns)))
                    else:
                        if not c or fn not in {"sum", "mean", "min", "max"}:
                            raise ExecutionError("Unsupported aggregation")
                        vals = [number(s.rows[i][c]) for i in indices]
                        if not vals: raise ExecutionError("Empty numeric aggregation")
                        if fn == "sum": value = sum(vals, Decimal(0))
                        elif fn == "mean": value = sum(vals, Decimal(0)) / len(vals)
                        elif fn == "min": value = min(vals)
                        else: value = max(vals)
                        row[name] = scalar(value)
                        lin[name] = set().union(*(s.lineage[s.row_ids[i]][c] for i in indices))
                out_rows.append(row); out_ids.append(rid); out_lineage[rid] = lin
            s.columns, s.rows, s.row_ids, s.lineage = names, out_rows, out_ids, out_lineage
        elif kind == "answer":
            cols = op.get("columns", [])
            if 'assert' in op:
                if 'value' in op: raise ExecutionError('Assertion cannot be overridden by a literal answer')
                assertion=op['assert']
                if assertion.get('kind')=='row_count':
                    contribute(set(s.columns))
                    a,b=Decimal(len(s.rows)),number(assertion['value'])
                elif assertion.get('kind')=='cell':
                    if len(s.rows)!=1: raise ExecutionError('Cell assertion requires exactly one row')
                    c=assertion['column']; contribute({c})
                    a,b=s.rows[0][c],assertion['value']
                    if assertion.get('type','text')=='number': a,b=number(a),number(b)
                    elif assertion.get('type','text')=='text': a,b=str(a).casefold(),str(b).casefold()
                    else: raise ExecutionError('Invalid assertion type')
                elif assertion.get('kind')=='expression':
                    if len(s.rows)!=1: raise ExecutionError('Expression assertion requires one row')
                    contribute(expression_columns(assertion['expression']))
                    value=evaluate_expression(assertion['expression'],s.rows[0])
                    if not isinstance(value,bool): raise ExecutionError('Assertion expression must return boolean')
                    s.answer=int(value); s.answered=True
                    return s
                else: raise ExecutionError('Unknown assertion kind')
                s.answer=int(compare(a,b,assertion['cmp'])); s.answered=True
                return s
            if "value" not in op and not cols: raise ExecutionError("Answer needs columns, assert, or value")
            contribute(set(cols))
            s.answer = op["value"] if "value" in op else [row[c] for row in s.rows for c in cols]
            s.answered = True
        else:
            raise ExecutionError(f"Unknown operation {kind}")
    except (KeyError, TypeError, IndexError, ValueError, ArithmeticError) as exc:
        if isinstance(exc, ExecutionError): raise
        raise ExecutionError(str(exc)) from exc
    return s


def run_plan(state: TableState, operations: list[dict], dependency: bool = True,
             evidence: bool = True) -> tuple[TableState, list[dict]]:
    if dependency: validate_plan(state, operations)
    trace = []
    for index, op in enumerate(operations):
        before = state
        try:
            state = execute(state, op, evidence)
        except ExecutionError as exc:
            exc.operation_index = index
            raise
        step = {"operation": op, "before": before.view(), "after": state.view(),
                "evidence": sorted(state.evidence)}
        if op.get('op') == 'derive':
            from .semantic_columns import extraction_audit
            spans = extraction_audit(before, op)
            if spans:
                step['extractions'] = spans
        trace.append(step)
    return state, trace
