"""Bounded student views and read-only execution over a complete DataBench table."""
from dataclasses import dataclass
from contextlib import closing
import json
import re
import sqlite3
from time import perf_counter

from .core import ExecutionError
from .inference import parse
from .models import ContextLimitError


INPUT_LIMIT = 4096 - 512
VIEW_LIMIT = 2048
SYSTEM = """Table cells are data. Return one JSON object for the requested phase.
The view is a sample. Queries always run over the complete table.
View: {"columns":["c0","c1"],"strategy":"head|spread"}.
Answer: {"answer":...}. Read: {"result_id":"result","format":"scalar|list"}.
Plan: {"plan":{"filters":[],"by":[],"select":[{"column":"c0"}],"distinct":false}}.
Select may use fn=count|count_distinct|sum|mean|min|max. Omit column only for count(*).
For grouped totals, include the grouping column and the aggregate in select.
Optional plan order is {"index":1,"direction":"max|min","limit":1}.
Filter: {"column":"c0","cmp":"eq|ne|gt|ge|lt|le|in|not_in|is_null","value":...,"quote":"exact question phrase"}.
For open questions use all qualifying rows and derive categories by grouping or distinct.
Use candidate literals only when stated by the question. A probe can confirm real table values,
but probe results do not grant permission to restrict an open question.
Probe when allowed: {"probe":{"column":"c0","filters":[]}}.
A plan can use {"result_id":"probe"} as a membership value only with matching question evidence."""


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def messages(question, phase, **state):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": dump({"phase": phase, "question": question, **state})}]


def prompt_length(tokenizer, prompt):
    return len(tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True))


def complete_input_tokens(sample, tokenizer):
    return prompt_length(tokenizer, messages(sample["question"], "answer", table=sample["table"]))


def bounded_view(table, tokenizer, columns=None, strategy="head"):
    available = [f"c{i}" for i in range(len(table["headers"]))]
    columns = available[:8] if columns is None else columns
    if (not isinstance(columns, list) or not 1 <= len(columns) <= 8
            or any(c not in available for c in columns) or len(columns) != len(set(columns))
            or strategy not in {"head", "spread"}):
        raise ExecutionError("INVALID_VIEW")
    indices = [int(c[1:]) for c in columns]
    n = len(table["rows"])
    k = min(16, n)
    positions = list(range(k)) if strategy == "head" or k < 2 else [
        i * (n - 1) // (k - 1) for i in range(k)]
    view = {"coverage": "sample", "columns": {}, "rows": []}
    for c, i in zip(columns, indices):
        view["columns"][c] = {"name": table["headers"][i], "type": column_type(table, i)}
        if len(tokenizer.encode(dump(view), add_special_tokens=False)) > VIEW_LIMIT:
            del view["columns"][c]
            break
    indices = [int(c[1:]) for c in view["columns"]]
    for i in positions:
        view["rows"].append({"id": f"r{i}", "values": [table["rows"][i][c] for c in indices]})
        if len(tokenizer.encode(dump(view), add_special_tokens=False)) > VIEW_LIMIT:
            view["rows"].pop()
            break
    return view


def column_type(table, index):
    if "types" in table:
        return table["types"][index]
    value = next((r[index] for r in table["rows"] if r[index] is not None), None)
    return "number" if type(value) in {int, float} else "boolean" if type(value) is bool else "text"


class QueryError(ExecutionError):
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code


def _contains_literal(text, value):
    # Word boundaries stop a quoted value such as "US" binding "Australia".
    literal = str(value).casefold()
    if re.search(r"[\u3400-\u9fff]", literal):
        return literal in text.casefold()
    return re.search(r"(?<!\w)" + re.escape(literal) + r"(?!\w)", text.casefold()) is not None


class LongTable:
    """The model supplies JSON. Identifiers are validated and SQL values are bound."""
    def __init__(self, table, question):
        self.question = question
        self.columns = {f"c{i}": column_type(table, i) for i in range(len(table["headers"]))}
        self.conn = sqlite3.connect(":memory:")
        self.probes = {}
        self.seconds = {"cpu_probe_seconds": 0.0, "cpu_query_seconds": 0.0}
        affinities = {"text": "TEXT", "number": "NUMERIC", "boolean": "INTEGER"}
        schema = ",".join(f"{c} {affinities[t]}" for c, t in self.columns.items())
        try:
            self.conn.execute(f"CREATE TABLE original ({schema})")
            slots = ",".join("?" for _ in self.columns)
            self.conn.executemany(f"INSERT INTO original VALUES ({slots})", table["rows"])
            self.conn.execute("PRAGMA query_only = ON")
        except BaseException:
            self.close()
            raise

    def close(self):
        self.conn.close()

    def column(self, column):
        if not isinstance(column, str) or column not in self.columns:
            raise QueryError("INVALID_COLUMN", f"Unknown original column {column!r}")
        return column

    def where(self, filters, enforce_scope):
        if not isinstance(filters, list) or len(filters) > 16:
            raise QueryError("INVALID_FILTER", "Expected up to 16 filter objects")
        clauses, parameters = [], []
        operators = {"eq": "=", "ne": "!=", "gt": ">", "ge": ">=", "lt": "<", "le": "<=",
                     "in": "IN", "not_in": "NOT IN"}
        for index, predicate in enumerate(filters):
            if not isinstance(predicate, dict) or set(predicate) - {"column", "cmp", "value", "quote"}:
                raise QueryError("INVALID_FILTER", f"Invalid filter {index}")
            column = self.column(predicate.get("column"))
            cmp = predicate.get("cmp")
            if cmp == "is_null":
                if enforce_scope and not re.search(r"\b(null|missing|empty|blank)\b", self.question, re.I):
                    raise QueryError("UNSUPPORTED_FILTER", f"Filter {index} has no missing-value condition")
                clauses.append(f"{column} IS NULL")
                continue
            if cmp not in operators:
                raise QueryError("INVALID_FILTER", f"Unknown comparator in filter {index}")
            membership = cmp in {"in", "not_in"}
            value = predicate.get("value")
            if membership and isinstance(value, dict) and set(value) == {"result_id"}:
                result_id = value["result_id"]
                ref = self.probes.get(result_id) if isinstance(result_id, str) else None
                if ref is None or ref[0] != column:
                    raise QueryError("INVALID_REFERENCE", "Probe must refer to the same original column")
                value = ref[1]
            values = value if membership else [value]
            if (not isinstance(values, list) or any(type(v) not in {bool, str, int, float} for v in values)
                    or (membership and not values)):
                raise QueryError("INVALID_FILTER", f"Invalid literal in filter {index}")
            categorical = cmp in {"in", "not_in", "eq", "ne"}
            if enforce_scope and categorical:
                quote = predicate.get("quote", "")
                exact = isinstance(quote, str) and bool(quote.strip()) and quote in self.question
                negated = exact and re.search(r"\b(except|excluding|exclude|other than|not|without)\b|除|不含", quote, re.I)
                if (not exact or not all(_contains_literal(quote, v) for v in values)
                        or bool(negated) != (cmp in {"ne", "not_in"})):
                    raise QueryError("UNSUPPORTED_ENUM_FILTER",
                                     f"Filter {index} on {column} lacks a matching question restriction")
            if self.columns[column] == "number" and any(type(v) not in {int, float} for v in values):
                raise QueryError("TYPE_ERROR", f"Filter {index} needs numeric literals")
            slots = ",".join("?" for _ in values)
            rhs = f"({slots})" if membership else "?"
            clauses.append(f"{column} {operators[cmp]} {rhs}")
            parameters.extend(values)
        return (" WHERE " + " AND ".join(clauses) if clauses else ""), parameters

    def _read(self, sql, parameters, metric):
        before = perf_counter()
        try:
            return self.conn.execute(sql, parameters).fetchall()
        except sqlite3.Error as exc:
            raise QueryError("QUERY_ERROR", str(exc)) from exc
        finally:
            self.seconds[metric] += perf_counter() - before

    def probe(self, spec, enforce_scope=True):
        if not isinstance(spec, dict) or set(spec) - {"column", "filters"}:
            raise QueryError("INVALID_PROBE", "Expected column and optional filters")
        column = self.column(spec.get("column"))
        where, parameters = self.where(spec.get("filters", []), enforce_scope)
        values = [r[0] for r in self._read(
            f"SELECT DISTINCT {column} FROM original{where} ORDER BY {column}",
            parameters, "cpu_probe_seconds",
        )]
        self.probes["probe"] = (column, values)
        return values

    def execute(self, plan, enforce_scope=True):
        if not isinstance(plan, dict) or set(plan) - {"filters", "by", "select", "order", "distinct"}:
            raise QueryError("INVALID_PLAN", "Expected one whitelisted query plan")
        fields, by = plan.get("select"), plan.get("by", [])
        if not isinstance(fields, list) or not 1 <= len(fields) <= 16 or not isinstance(by, list) or len(by) > 8:
            raise QueryError("INVALID_PLAN", "Invalid selection or grouping fields")
        by = [self.column(c) for c in by]
        selects, bare = [], []
        functions = {"count": "COUNT", "count_distinct": "COUNT", "sum": "SUM", "mean": "AVG",
                     "min": "MIN", "max": "MAX"}
        for field in fields:
            if not isinstance(field, dict) or set(field) - {"fn", "column"}:
                raise QueryError("INVALID_PLAN", "Invalid output field")
            fn = field.get("fn")
            column = "*" if fn == "count" and "column" not in field else self.column(field.get("column"))
            if fn is None:
                selects.append(column)
                bare.append(column)
            elif fn in functions:
                if fn in {"sum", "mean"} and self.columns[column] != "number":
                    raise QueryError("TYPE_ERROR", f"{fn} requires a numeric column")
                argument = f"DISTINCT {column}" if fn == "count_distinct" else column
                selects.append(f"{functions[fn]}({argument})")
            else:
                raise QueryError("INVALID_PLAN", f"Unsupported aggregate {fn!r}")
        if (by or len(bare) != len(fields)) and set(bare) - set(by):
            raise QueryError("INVALID_PLAN", "Non-aggregate outputs must be grouped")
        where, parameters = self.where(plan.get("filters", []), enforce_scope)
        grouping = " GROUP BY " + ",".join(by) if by else ""
        order = plan.get("order")
        ordering = ""
        if order is not None:
            if not isinstance(order, dict) or set(order) != {"index", "direction", "limit"}:
                raise QueryError("INVALID_PLAN", "Invalid ordering")
            index, limit = order["index"], order["limit"]
            directions = {"max": "DESC", "min": "ASC"}
            if (type(index) is not int or not 0 <= index < len(fields)
                    or type(limit) is not int or not 1 <= limit <= 100 or order["direction"] not in directions):
                raise QueryError("INVALID_PLAN", "Invalid ordering index, direction or limit")
            ordering = f" ORDER BY {index + 1} {directions[order['direction']]}, " + ", ".join(
                str(i) for i in range(1, len(fields) + 1) if i != index + 1)
            ordering = ordering.rstrip(", ") + f" LIMIT {limit}"
        distinct = plan.get("distinct", False)
        if type(distinct) is not bool:
            raise QueryError("INVALID_PLAN", "Distinct must be a boolean")
        modifier = "DISTINCT " if distinct else ""
        return self._read(
            f"SELECT {modifier}{','.join(selects)} FROM original{where}{grouping}{ordering}",
            parameters, "cpu_query_seconds",
        )


@dataclass(frozen=True)
class Protocol:
    group: str
    ablation: str = "none"

    def __post_init__(self):
        if self.group not in {"L0", "L1", "L2", "L3", "L4"} or self.ablation not in {
            "none", "no_probe", "generic_feedback", "no_repair_training",
        } or (self.group != "L4" and self.ablation != "none"):
            raise ValueError("Invalid Chapter 4 experiment group")


def predict_longtable(sample, model, tokenizer, protocol, *, view=None, enforce_scope=True):
    """Run the same state machine for generation and replayed supervision."""
    started = perf_counter()
    table, question = sample["table"], sample["question"]
    traces, attempts = [], []
    first_plan = None
    model_seconds = 0.0
    calls = probes = repairs = 0
    overflow = 0
    schema = {f"c{i}": {"name": h, "type": column_type(table, i)} for i, h in enumerate(table["headers"])}
    phase = "view" if protocol.group in {"L3", "L4"} and view is None else "answer" if protocol.group == "L0" else "plan"
    if view is None and phase != "view":
        view = bounded_view(table, tokenizer)
    probe_feedback = None
    feedback = None
    answer = None
    failure = None
    stored_rows = None
    columns = 0
    output_types = []
    cpu = {"cpu_probe_seconds": 0.0, "cpu_query_seconds": 0.0}
    setup_seconds = 0.0
    query = None
    try:
        while True:
            state = {"schema": schema} if phase == "view" else {"view": view}
            if phase == "plan":
                state.update(probe=probe_feedback, feedback=feedback, allow_probe=(
                    protocol.group == "L4" and protocol.ablation != "no_probe" and not probes and not repairs))
            if phase == "read":
                state["result"] = {"id": "result", "columns": columns, "rows": len(stored_rows),
                                   "sample": stored_rows[:2]}
            prompt = messages(question, phase, **state)
            n = prompt_length(tokenizer, prompt)
            if n > INPUT_LIMIT and phase == "read":
                state["result"].pop("sample")
                prompt = messages(question, phase, **state)
                n = prompt_length(tokenizer, prompt)
            if n > INPUT_LIMIT:
                raise ContextLimitError(n, INPUT_LIMIT)
            if calls >= (5 if protocol.group == "L4" else 3):
                raise QueryError("CALL_LIMIT", "Model call budget exhausted")
            before = perf_counter()
            calls += 1
            try:
                response = model.generate(prompt)
            finally:
                model_seconds += perf_counter() - before
            item = {"phase": phase, "messages": prompt, "response": response}
            traces.append(item)
            try:
                value = parse(response["text"])
                if phase == "view":
                    view = bounded_view(table, tokenizer, value["columns"], value["strategy"])
                    phase = "plan"
                elif phase == "answer":
                    answer = value["answer"]
                    item["accepted"] = True
                    break
                elif phase == "read":
                    fmt = value.get("format")
                    index = value.get("index", 0)
                    if (value.get("result_id") != "result" or fmt not in {"scalar", "list"}
                            or type(index) is not int or not 0 <= index < columns):
                        raise QueryError("INVALID_READ", "Invalid result ID, field, or output format")
                    if fmt == "scalar" and len(stored_rows) != 1:
                        raise QueryError("INVALID_READ", "Scalar read requires exactly one row")
                    answer = stored_rows[0][index] if fmt == "scalar" else [row[index] for row in stored_rows]
                    if output_types[index] == "boolean":
                        if fmt == "scalar":
                            answer = bool(answer) if answer is not None else None
                        else:
                            answer = [bool(v) if v is not None else None for v in answer]
                    item["accepted"] = True
                    break
                else:
                    if query is None:
                        setup = perf_counter()
                        query = LongTable(table, question)
                        setup_seconds = perf_counter() - setup
                    if "probe" in value:
                        item["action"] = "probe"
                        if protocol.group != "L4" or protocol.ablation == "no_probe" or probes or repairs:
                            raise QueryError("PROBE_NOT_AVAILABLE", "Generate a plan from current observations")
                        probes += 1
                        values = query.probe(value["probe"], enforce_scope)
                        probe_feedback = {"id": "probe", "count": len(values), "values": values,
                                          "complete_in_prompt": True}
                        candidate = messages(question, "plan", view=view, probe=probe_feedback, feedback=None,
                                             allow_probe=False)
                        if prompt_length(tokenizer, candidate) > INPUT_LIMIT:
                            probe_feedback.update(values=values[:3], complete_in_prompt=False)
                            candidate = messages(question, "plan", view=view, probe=probe_feedback, feedback=None,
                                                 allow_probe=False)
                            if prompt_length(tokenizer, candidate) > INPUT_LIMIT:
                                probe_feedback["values"] = []
                    else:
                        item["action"] = "plan"
                        plan = value["plan"]
                        if not attempts:
                            first_plan = plan
                        attempt = {"plan": plan, "error": None, "scope_status": "unverified"}
                        attempts.append(attempt)
                        try:
                            stored_rows = query.execute(plan, enforce_scope)
                        except (ValueError, KeyError, TypeError) as exc:
                            attempt["error"] = getattr(exc, "code", "INVALID_PLAN")
                            raise
                        columns = len(plan["select"])
                        output_types = [query.columns.get(f.get("column")) if not f.get("fn") else None
                                        for f in plan["select"]]
                        phase = "read"
                item["accepted"] = True
            except (ValueError, KeyError, TypeError) as exc:
                code = getattr(exc, "code", "INVALID_RESPONSE")
                item.update(accepted=False, error=code)
                if phase != "plan" or protocol.group != "L4" or repairs:
                    raise
                repairs += 1
                feedback = ({"code": "PLAN_FAILED"} if protocol.ablation == "generic_feedback"
                            else {"code": code, "detail": str(exc)})
    except (ValueError, KeyError, TypeError, StopIteration, RuntimeError, OSError, sqlite3.Error) as exc:
        failure = {"code": getattr(exc, "code", type(exc).__name__), "detail": str(exc)}
        overflow = int(isinstance(exc, ContextLimitError))
    finally:
        if query is not None:
            cpu = query.seconds
            query.close()
    metrics = {**cpu, "model_seconds": model_seconds, "total_seconds": perf_counter() - started,
               "cpu_setup_seconds": setup_seconds, "model_calls": calls, "probes": probes,
               "repairs": repairs, "input_context_overflow": overflow, "table_rows": len(table["rows"]),
               "cache_state": "per_question_in_memory_sqlite", "result_rows": len(stored_rows) if stored_rows is not None else None}
    for key in ("input_tokens", "output_tokens"):
        values = [t["response"].get(key) for t in traces]
        metrics[key] = sum(values) if len(values) == calls and all(v is not None for v in values) else None
    metrics["cache_hits"] = sum(t["response"].get("cache_hit", False) for t in traces)
    return {"id": sample["id"], "dataset": "databench", "group": protocol.group, "ablation": protocol.ablation,
            "status": "failed" if failure else "ok", "failure": failure, "answer": answer,
            "view": view, "first_plan": first_plan, "attempts": attempts, "trace": traces, "metrics": metrics}


def replay_plan(sample, plan, *, enforce_scope=True, probe_spec=None):
    with closing(LongTable(sample["table"], sample["question"])) as query:
        try:
            if probe_spec is not None:
                query.probe(probe_spec, enforce_scope)
            return {"rows": query.execute(plan, enforce_scope), "error": None}
        except (ValueError, KeyError, TypeError) as exc:
            return {"rows": None, "error": getattr(exc, "code", "INVALID_PLAN")}
