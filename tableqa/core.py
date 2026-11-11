# 模块职责：定义表状态与白名单操作，跟踪原始单元格来源，并通过快照保证失败操作可回滚。
# 接口约定：样本、操作和结果采用本项目的结构化字典；细节见各入口及字段检查。
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


# ExecutionError：执行协议错误；调用层据此回滚当前块并决定是否修复。
class ExecutionError(ValueError):
    pass


# 严格解析数值文本、千位分隔与百分数为 Decimal，拒绝歧义及非有限值。
# 输入：value。
# 异常约束：Non-finite number
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


# 把 Decimal 转为可输出的整数或浮点数，保持整数结果的表示。
# 输入：n。
def scalar(n: Decimal) -> int | float:
    return int(n) if n == n.to_integral_value() else float(n)


# 按白名单比较符求布尔结果，统一断言与表达式的比较行为。
# 输入：a；b；cmp。
def compare(a, b, cmp):
    if cmp == 'eq': return a == b
    if cmp == 'ne': return a != b
    if cmp == 'gt': return a > b
    if cmp == 'ge': return a >= b
    if cmp == 'lt': return a < b
    if cmp == 'le': return a <= b
    raise ExecutionError(f'Unknown comparator {cmp}')


# TableState：当前可访问列、行、稳定行 ID、来源映射和答案状态的完整快照。
@dataclass
class TableState:
    # 列使用稳定内部 ID；rows 的字典键与 columns 一致，显示表头只用于提示展示。
    columns: list[str]
    rows: list[dict[str, Any]]
    # row_ids 与 rows 按位置对齐，筛选、排序和聚合后同步维护。
    row_ids: list[str]
    # lineage[当前行 ID][当前列 ID] 是参与该值计算的原始 rX:cY 单元格集合。
    lineage: dict[str, dict[str, set[str]]]
    # evidence 还累计筛选、排序等决策读取的依据，范围可大于最终答案单元格。
    evidence: set[str] = field(default_factory=set)
    answer: Any = None
    # 使用独立终止标记，合法空答案或 None 值不应被误判为尚未读出。
    answered: bool = False

    # 由原始矩阵建立 c/r 坐标及逐单元格来源，拒绝不规则行。
    # 输入：table（原始表头与行矩阵）。
    # 异常约束：Headers must be nonempty strings
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

    # 导出当前可见状态，可按原始列 ID 附加显示表头。
    # 输入：headers（原始显示表头）。
    # 返回形式含：result。
    def view(self, headers: list[str] | None = None) -> dict:
        result = {"columns": self.columns, "row_ids": self.row_ids,
                  "rows": self.rows, "answered": self.answered}
        if headers is not None:
            result["original_headers"] = {f"c{i}": h for i, h in enumerate(headers) if f"c{i}" in self.columns}
        return result

    # 为当前值、来源、证据和答案构造确定性签名，供重放等价检查。
    def signature(self) -> str:
        # Ignore derived row IDs; include cell lineage and predicate evidence.
        return json.dumps({"columns": self.columns, "rows": self.rows,
                           "lineage": [ {c: sorted(self.lineage[r][c]) for c in self.columns}
                                       for r in self.row_ids],
                           "evidence": sorted(self.evidence), "answer": self.answer,
                           "answered": self.answered}, sort_keys=True, ensure_ascii=False)


# 递归提取表达式树读取的列，供依赖与来源传播使用。
# 输入：expr。
def expression_columns(expr: Any) -> set[str]:
    if not isinstance(expr, dict):
        return set()
    if "col" in expr:
        return {expr["col"]}
    return set().union(*(expression_columns(x) for x in expr.get("args", [])))


# 返回一个操作的读取列集合，提前检查引用是否可用。
# 输入：op。
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


# 返回派生或聚合操作创建的列名，更新后续计划的可用列集合。
# 输入：op。
def writes(op: dict) -> set[str]:
    if op["op"] == "derive":
        return {op["as"]}
    if op["op"] == "group_aggregate":
        return {a["as"] for a in op["aggregations"]}
    return set()


# 静态核对已给定块的读写依赖、投影删列与答案终止位置。
# 输入：state（当前执行状态）；operations（白名单操作序列）。
# 异常约束：Answer must terminate a block
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


# 递归计算白名单算术、比较、布尔和条件表达式，限制操作及参数类型。
# 输入：expr；row（当前行记录）。
# 返回形式含：expr。
# 异常约束：Expression must be a binary whitelist arithmetic tree
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


# 复制输入表状态，执行一个白名单原子并更新行域、单元格来源及答案。
# 输入：state（当前执行状态）；op；track_evidence。
# 返回形式含：s / True / False。
# 异常约束：Cannot operate after answer
def execute(state: TableState, op: dict, track_evidence: bool = True) -> TableState:
    s = deepcopy(state)  # Atomic operation; callers retain immutable snapshots.
    # 状态快照同时包含值、行 ID 和来源；任一操作抛错时，调用方仍持有完整旧状态。
    if s.answered:
        raise ExecutionError("Cannot operate after answer")
    kind = op.get("op")
    required = reads(op)
    if not required <= set(s.columns):
        raise ExecutionError(f"Unavailable columns: {required - set(s.columns)}")
    # 把参与当前计算的原单元格加入证据集合。
    # 输入：columns。
    def contribute(columns: set[str]) -> None:
        if track_evidence:
            for rid in s.row_ids:
                for c in columns:
                    s.evidence.update(s.lineage[rid][c])
    # 按选定位置同步保留行、稳定行 ID 及来源映射。
    # 输入：indices。
    def retain(indices: list[int]) -> None:
        s.rows = [s.rows[i] for i in indices]
        s.row_ids = [s.row_ids[i] for i in indices]
        s.lineage = {r: s.lineage[r] for r in s.row_ids}
    try:
        if kind == "project":
            # 投影只保留请求列。后续计划依赖是否被提前删除由 validate_plan 负责检查。
            cols = op["columns"]
            if not cols or len(cols) != len(set(cols)):
                raise ExecutionError("Projection must have unique nonempty columns")
            s.columns = list(cols)
            s.rows = [{c: row[c] for c in cols} for row in s.rows]
            s.lineage = {r: {c: s.lineage[r][c] for c in cols} for r in s.row_ids}
        elif kind == "filter":
            # 数值常量先于逐行匹配校验，因此空表也不能让错误常量悄悄通过。
            c, cmp = op["column"], op["cmp"]
            invalid_number = op.get("invalid_number", "error")
            if invalid_number not in {"error", "skip"}:
                raise ExecutionError("Invalid numeric filter policy")
            if invalid_number == "skip" and (op.get("type") != "number" or cmp not in {
                    "eq", "ne", "gt", "ge", "lt", "le", "in", "not_in"}):
                raise ExecutionError("Skipping invalid numbers requires a numeric comparison")
            numeric_target = None
            if op.get("type") == "number" and cmp in {"eq", "ne", "gt", "ge", "lt", "le", "in", "not_in"}:
                if cmp in {"in", "not_in"}:
                    if not isinstance(op.get("value"), list):
                        raise ExecutionError("Membership requires a list literal")
                    numeric_target = [number(v) for v in op["value"]]
                else:
                    numeric_target = number(op.get("value"))
            contribute({c})
            # 按当前函数约定计算匹配或筛选判定，供上层执行或官方判分复用。
            # 输入：row（当前行记录）。
            # 返回形式含：True / False。
            # 异常约束：Membership requires a list literal
            def match(row):
                a, b = row[c], op.get("value")
                if cmp == "is_null": return a is None or a == ""
                if cmp=='is_number':
                    try: number(a); return True
                    except ExecutionError: return False
                if cmp == "contains": return str(b).casefold() in str(a).casefold()
                if cmp in {'in','not_in'}:
                    if not isinstance(b,list): raise ExecutionError('Membership requires a list literal')
                    if op.get('type','text')=='number':
                        items = numeric_target
                        try:
                            a = number(a)
                        except ExecutionError:
                            if invalid_number == "skip": return False
                            raise
                    elif op.get('type','text')=='text': a,items=str(a).casefold(),[str(v).casefold() for v in b]
                    else: raise ExecutionError('Invalid membership type')
                    return (a in items) if cmp=='in' else (a not in items)
                if op.get("type", "text") == "number":
                    b = numeric_target
                    try:
                        a = number(a)
                    except ExecutionError:
                        if invalid_number == "skip": return False
                        raise
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
            # 对索引做稳定排序，再同步重排所有状态字段；相等键维持原始相对顺序。
            c = op["column"]
            contribute({c})
            value_type = op.get("type", "text")
            if value_type not in {"number", "text"}: raise ExecutionError("Unsupported sort type")
            key = (lambda i: number(s.rows[i][c])) if value_type == "number" else (lambda i: str(s.rows[i][c]))
            retain(sorted(range(len(s.rows)), key=key, reverse=op.get("descending", False)))
        elif kind == "select":
            # first/last 依赖当前行序，extreme 依赖数值；极值分支保留所有并列行。
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
            # 派生列的来源是表达式读取单元格来源的并集，列名不能覆盖现有字段。
            name = op["as"]
            if name in s.columns: raise ExecutionError("Derived column already exists")
            for rid, row in zip(s.row_ids, s.rows):
                row[name] = evaluate_expression(op["expression"], row)
                s.lineage[rid][name] = set().union(*(s.lineage[rid][c] for c in required))
            s.columns.append(name)
        elif kind == "group_aggregate":
            # 聚合产生 g 开头的新行 ID；每个输出单元格继续指向实际参与的原始单元格。
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


# 按顺序执行操作并记录每步前后状态；失败携带操作位置供调用层回滚。
# 输入：state（当前执行状态）；operations（白名单操作序列）；dependency；evidence。
# 返回形式含：2 项组成的结果。
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
