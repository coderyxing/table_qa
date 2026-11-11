"""用回放和真实 SQLite 验证 v6 提案的结果断言；不是模型效果评测。"""
from copy import deepcopy
import unittest
import tableqa.longtable as module
from tableqa.models import ReplayModel


class Tokenizer:
    """预算不是本组检查的变量；以固定长度近似构造小表的测试提示。"""
    def encode(self, text, **kwargs):
        return list(range((len(text) + 3) // 4))

    def apply_chat_template(self, messages, **kwargs):
        return self.encode("".join(m["content"] for m in messages))


TABLE = {"headers": ["Name", "Department", "Recommended", "Score"],
         "types": ["text", "text", "boolean", "number"],
         "rows": [["A", "Sales", True, 7], ["A", "R&D", False, 9], ["B", "R&D", True, 1]]}


def run(plan, read, *, table=None, group="L4", question="Check the executed result"):
    """模型回复预置，查询及读出走真实状态机，并保留实际调用预算。"""
    sample = {"id": "fixture", "dataset": "databench", "question": question, "table": table or TABLE}
    responses = [{"plan": plan}, {"result_id": "result", **read}]
    if group in {"L3", "L4"}:
        responses.insert(0, {"columns": ["c0", "c1", "c2", "c3"], "strategy": "head"})
    return module.predict_longtable(sample, ReplayModel(responses), Tokenizer(), module.Protocol(group))


class ResultReadTests(unittest.TestCase):
    # 存在性与空集直接由完整查询行数决定，输出必须是 bool。
    def test_exists_and_empty(self):
        for threshold, exists in [(0, True), (10, False)]:
            for fmt, expected in [("exists", exists), ("empty", not exists)]:
                with self.subTest(threshold=threshold, fmt=fmt):
                    p = run({"select": [{"column": "c3"}], "filters": [
                        {"column": "c3", "cmp": "gt", "value": threshold}]}, {"format": fmt})
                    self.assertEqual(p["status"], "ok")
                    self.assertIs(p["answer"], expected)
                    self.assertEqual(p["metrics"]["model_calls"], 3)

    # 唯一性比较 COUNT(*) 和 COUNT(DISTINCT) 的实际值，不让模型直接填真假。
    def test_unique_name_comparison(self):
        p = run({"select": [{"fn": "count"}, {"fn": "count_distinct", "column": "c0"}]},
                {"format": "assert", "index": 0, "cmp": "eq", "right_index": 1})
        self.assertEqual(p["status"], "ok")
        self.assertIs(p["answer"], False)

    # “同一部门”的结构常数 1 只作用于真实计数结果。
    def test_one_distinct_category(self):
        plan = {"select": [{"fn": "count_distinct", "column": "c1"}]}
        read = {"format": "assert", "index": 0, "cmp": "eq", "value": 1}
        self.assertIs(run(plan, read)["answer"], False)
        same = deepcopy(TABLE)
        for row in same["rows"]:
            row[1] = "Sales"
        self.assertIs(run(plan, read, table=same)["answer"], True)

    # 常数字面量、任意答案及越界索引均不能绕过结果证据。
    def test_invalid_assertions_fail(self):
        plan = {"select": [{"fn": "count"}, {"fn": "count_distinct", "column": "c0"}]}
        base = {"format": "assert", "index": 0, "cmp": "eq"}
        for extra in [{"value": 7}, {"value": True}, {"right_index": True}, {"right_index": 5},
                      {"right_index": 1, "value": 1}, {"right_index": 1, "answer": True}]:
            with self.subTest(extra=extra):
                p = run(plan, {**base, **extra})
                self.assertEqual(p["status"], "failed")
                self.assertEqual(p["failure"]["code"], "INVALID_READ")
        p = run({"select": [{"fn": "sum", "column": "c3"}]}, {**base, "value": 1})
        self.assertEqual(p["failure"]["code"], "INVALID_READ")

    # 旧的单值读出仍要求恰好一行；新存在性不能接收模型答案字面量。
    def test_existing_scalar_guard_and_no_literal_exists(self):
        plan = {"select": [{"column": "c0"}]}
        for read in [{"format": "scalar"}, {"format": "exists", "answer": True}]:
            self.assertEqual(run(plan, read)["failure"]["code"], "INVALID_READ")

    # 布尔极值保持布尔类型；均值和求和分别返回比例与数量。
    def test_boolean_aggregate_types(self):
        for fn, expected in [("min", False), ("max", True), ("sum", 2)]:
            p = run({"select": [{"fn": fn, "column": "c2"}]}, {"format": "scalar"})
            self.assertEqual(p["status"], "ok")
            self.assertEqual(p["answer"], expected)
            self.assertIs(type(p["answer"]), type(expected))
        p = run({"select": [{"fn": "mean", "column": "c2"}]}, {"format": "scalar"})
        self.assertAlmostEqual(p["answer"], 2 / 3)
        self.assertIs(type(p["answer"]), float)

    # 缺失的数值聚合结果不能被默认为一个真假标签。
    def test_missing_numeric_result_rejected(self):
        missing = deepcopy(TABLE)
        for row in missing["rows"]:
            row[3] = None
        p = run({"select": [{"fn": "min", "column": "c3"}, {"fn": "max", "column": "c3"}]},
                {"format": "assert", "index": 0, "cmp": "eq", "right_index": 1}, table=missing)
        self.assertEqual(p["failure"]["code"], "INVALID_READ")

    # L1–L4 共用读出语义，不把新能力只给予某个消融组。
    def test_all_plan_groups_share_exists(self):
        for group in ["L1", "L2", "L3", "L4"]:
            p = run({"select": [{"column": "c0"}]}, {"format": "exists"}, group=group)
            self.assertIs(p["answer"], True)

    # 题意范围检查不随新增读出放宽：未在问题出现的枚举值仍应拦截。
    def test_scope_check_unchanged(self):
        from contextlib import closing
        with closing(module.LongTable(TABLE, "Which departments are present?")) as query:
            with self.assertRaises(module.QueryError) as error:
                query.execute({"select": [{"column": "c0"}], "filters": [
                    {"column": "c1", "cmp": "eq", "value": "Sales", "quote": "departments"}]})
            self.assertEqual(error.exception.code, "UNSUPPORTED_ENUM_FILTER")

    # 缺失 value 的反馈应指明字段；一次修复仍占原有调用预算，且不带参考答案。
    def test_missing_value_feedback_allows_bounded_repair(self):
        sample = {"id": "fixture", "dataset": "databench", "table": TABLE,
                  "question": "Are there people in Sales?"}
        bad = {"plan": {"select": [{"column": "c0"}], "filters": [
            {"column": "c1", "cmp": "eq", "quote": "Sales"}]}}
        good = deepcopy(bad)
        good["plan"]["filters"][0]["value"] = "Sales"
        p = module.predict_longtable(sample, ReplayModel([
            {"columns": ["c0", "c1"], "strategy": "head"}, bad, good,
            {"result_id": "result", "format": "exists"}]), Tokenizer(), module.Protocol("L4"))
        self.assertEqual(p["status"], "ok")
        self.assertIs(p["answer"], True)
        self.assertEqual(p["metrics"]["repairs"], 1)
        self.assertEqual(p["metrics"]["model_calls"], 4)
        self.assertEqual(p["attempts"][0]["error"], "MISSING_FILTER_VALUE")
        self.assertIn("value literal", p["trace"][2]["messages"][-1]["content"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
