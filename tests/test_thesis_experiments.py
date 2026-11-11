# 测试范围：验证对应模块的数据隔离、输入约束与历史回归行为。
# 运行方式：python -m unittest discover -s tests -p test_thesis_experiments.py -v
# 阅读顺序：先看样本或替身模型的构造，再看每个用例前的预期约束。
import unittest

from tableqa.chapter3 import bind_condition_quotes, teacher_audit, teacher_prediction
from tableqa.chapter4 import cost_report, main_report, paired_diagnostic, supervision, validate_pair_sources
from tableqa.contracts import ContractError
from tableqa.data import digest
from tableqa.longtable import INPUT_LIMIT, Protocol, bounded_view, predict_longtable, replay_plan
from tableqa.models import ReplayModel


# FixtureTokenizer：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
class FixtureTokenizer:
    # 用确定性的测试编码规则返回 token 序列，避免依赖实际模型权重。
    # 输入：text（待解析文本）；**kwargs。
    def encode(self, text, **kwargs):
        return list(range(len(text)))

    # 用确定性的测试聊天模板表示消息，便于核对提示边界和成本。
    # 输入：messages（模型可见消息）；**kwargs。
    def apply_chat_template(self, messages, **kwargs):
        return self.encode("".join(m["content"] for m in messages))


# 按执行成功且答案完全一致的规则给合成夹具评分。
# 输入：prediction（单条预测或教师轨迹）；sample（规范样本记录）。
def exact(prediction, sample):
    return prediction["status"] == "ok" and prediction["answer"] == sample["reference"]


# ThesisExperiments：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
class ThesisExperiments(unittest.TestCase):
    # 准备本组测试的独立样本与执行状态，避免用例之间共享可变结果。
    def setUp(self):
        self.tokenizer = FixtureTokenizer()
        self.table = {"headers": ["Brand", "Sales"], "types": ["text", "number"],
                      "rows": [["Apple", 2]] * 16 + [["Xiaomi", 40]]}
        self.sample = {"dataset": "databench", "split": "train", "id": "one",
                       "table": self.table, "question": "Which brand has the highest total sales?",
                       "reference": "Xiaomi", "semantic": "category"}
        self.plan = {"by": ["c0"], "select": [{"column": "c0"}, {"fn": "sum", "column": "c1"}],
                     "order": {"index": 1, "direction": "max", "limit": 1}}
        self.read = {"result_id": "result", "format": "scalar"}

    # 回归用例：test_sample_literal_rejected_but_original_table_used
    # 应相等：replay_plan(self.sample, constrained)['error']，'UNSUPPORTED_ENUM_FILTER'
    # 应相等：replay_plan(self.sample, constrained, enforce_scope=False)['rows']，[('Apple', 32)]
    # 应相等：replay_plan(limited, legitimate)['rows']，[('Xiaomi', 40)]
    def test_sample_literal_rejected_but_original_table_used(self):
        constrained = {**self.plan, "filters": [
            {"column": "c0", "cmp": "eq", "value": "Apple", "quote": "Apple"}]}
        self.assertEqual(replay_plan(self.sample, constrained)["error"], "UNSUPPORTED_ENUM_FILTER")
        self.assertEqual(replay_plan(self.sample, constrained, enforce_scope=False)["rows"], [("Apple", 32)])
        limited = {**self.sample, "question": "Among Apple and Xiaomi, which has the highest total sales?"}
        legitimate = {**self.plan, "filters": [
            {"column": "c0", "cmp": "in", "value": ["Apple", "Xiaomi"], "quote": "Among Apple and Xiaomi"}]}
        self.assertEqual(replay_plan(limited, legitimate)["rows"], [("Xiaomi", 40)])
        view = bounded_view(self.table, self.tokenizer)
        self.assertEqual({r["values"][0] for r in view["rows"]}, {"Apple"})
        pred = predict_longtable(self.sample, ReplayModel([{"plan": self.plan}, self.read]),
                                 self.tokenizer, Protocol("L2"))
        self.assertEqual(pred["answer"], "Xiaomi")
        self.assertGreater(pred["metrics"]["cpu_query_seconds"], 0)
        self.assertEqual(pred["metrics"]["cpu_probe_seconds"], 0)
        self.assertEqual(len(supervision(self.sample, pred, self.tokenizer, Protocol("L2"), exact)), 2)
        with self.assertRaises(ValueError):
            supervision({**self.sample, "split": "test"}, pred, self.tokenizer, Protocol("L2"), exact)

    # 回归用例：test_probe_repair_and_cost_reporting
    # 应相等：pred['answer']，self.sample['reference']
    # 应相等：pred['metrics']['model_calls']，5
    # 应相等：pred['metrics']['repairs']，1
    def test_probe_repair_and_cost_reporting(self):
        wrong = {**self.plan, "filters": [{"column": "c0", "cmp": "eq", "value": "Apple"}]}
        responses = [{"columns": ["c0", "c1"], "strategy": "head"}, {"probe": {"column": "c0"}},
                     {"plan": wrong}, {"plan": self.plan}, self.read]
        pred = predict_longtable(self.sample, ReplayModel(responses), self.tokenizer, Protocol("L4"))
        self.assertEqual(pred["answer"], self.sample["reference"])
        self.assertEqual(pred["metrics"]["model_calls"], 5)
        self.assertEqual(pred["metrics"]["repairs"], 1)
        self.assertGreater(pred["metrics"]["cpu_probe_seconds"], 0)
        self.assertEqual(pred["attempts"][0]["error"], "UNSUPPORTED_ENUM_FILTER")
        self.assertEqual(cost_report([self.sample], [pred], exact)["accuracy"], 1)
        no_repair = supervision(self.sample, pred, self.tokenizer, Protocol("L4", "no_repair_training"), exact)
        self.assertEqual([r["stage"] for r in no_repair], ["view", "plan", "read"])

    # 回归用例：test_legal_zero_count_and_high_cardinality_probe
    # 应相等：replay_plan(sample, count)['rows']，[(0,)]
    # 应相等：pred['answer']，1500
    # 应包含：'"complete_in_prompt":false'，pred['trace'][2]['messages'][-1]['content']
    def test_legal_zero_count_and_high_cardinality_probe(self):
        count = {"select": [{"fn": "count"}], "filters": [
            {"column": "c0", "cmp": "eq", "value": "Nokia", "quote": "Nokia"}]}
        sample = {**self.sample, "question": "How many Nokia rows are there?"}
        self.assertEqual(replay_plan(sample, count)["rows"], [(0,)])
        table = {**self.table, "rows": [[f"brand {i}", i] for i in range(1500)]}
        sample = {**sample, "table": table, "question": "How many distinct brands are there?"}
        pred = predict_longtable(sample, ReplayModel([
            {"columns": ["c0"], "strategy": "head"}, {"probe": {"column": "c0"}},
            {"plan": {"select": [{"fn": "count_distinct", "column": "c0"}]}}, self.read,
        ]), self.tokenizer, Protocol("L4"))
        self.assertEqual(pred["answer"], 1500)
        self.assertIn('"complete_in_prompt":false', pred["trace"][2]["messages"][-1]["content"])

    # 回归用例：test_table_cluster_bootstrap_and_threshold
    # 应相等：report['table'][3]['short_n']，4
    # 应相等：report['paired_table_bootstrap']['L3']['ci95']，[1, 1]
    # 应成立：report['paired_table_bootstrap']['L3']['supports_larger_long_gain']
    def test_table_cluster_bootstrap_and_threshold(self):
        samples = [{**self.sample, "id": str(i), "table_id": str(i)} for i in range(8)]
        groups = {}
        for group in ("L0", "L1", "L2", "L3", "L4"):
            groups[group] = [{"id": s["id"], "group": group, "ablation": "none", "status": "ok",
                              "answer": s["reference"] if group in {"L3", "L4"} and i >= 4 else "Apple",
                              "complete_input_tokens": INPUT_LIMIT + int(i >= 4)}
                             for i, s in enumerate(samples)]
        report = main_report(samples, groups, exact, repeats=200)
        self.assertEqual(report["table"][3]["short_n"], 4)
        self.assertEqual(report["paired_table_bootstrap"]["L3"]["ci95"], [1, 1])
        self.assertTrue(report["paired_table_bootstrap"]["L3"]["supports_larger_long_gain"])

    # 回归用例：test_test_pairs_exclude_train_fingerprint_and_dev_id
    # 应相等：set(validate_pair_sources(train, test, 'test', dev))，{'c'}
    # 应抛出预期异常：ValueError
    def test_test_pairs_exclude_train_fingerprint_and_dev_id(self):
        # 构造带划分、原表身份和指纹的泄露检查样本。
        # 输入：uid；split（数据划分）；table（原始表头与行矩阵）；fingerprint。
        # 返回形式含：字段 id、split、table_id、table_fingerprint、question。
        def row(uid, split, table, fingerprint):
            return {**self.sample, "id": uid, "split": split, "table_id": table,
                    "table_fingerprint": fingerprint, "question": uid}

        train = [row("tr", "train", "tr_table", "tr_fp")]
        dev = [row("de", "validation", "de_table", "de_fp")]
        test = [row("a", "test", "another", "tr_fp"), row("b", "test", "de_table", "new"),
                row("c", "test", "fresh", "newer")]
        self.assertEqual(set(validate_pair_sources(train, test, "test", dev)), {"c"})
        with self.assertRaises(ValueError):
            validate_pair_sources(train, train, "test", dev)

    # 回归用例：test_hidden_answer_changes_under_fixed_first_plan
    # 应成立：result['both_correct']
    # 应相等：[r['rows'][0][0] for r in result['fixed_first_plan_replay']]，['Xiaomi', 'Apple']
    def test_hidden_answer_changes_under_fixed_first_plan(self):
        view = bounded_view(self.table, self.tokenizer)
        a = {**self.sample, "id": "a", "view": view}
        altered = {**self.table, "rows": self.table["rows"][:-1] + [["Xiaomi", 3]]}
        b = {**a, "id": "b", "table": altered, "reference": "Apple"}
        payload = {"source_id": self.sample["id"], "kind": "hidden_answer", "a": a, "b": b}
        pair = {**payload, "review": {"answers_and_scope_verified": True, "pair_hash": digest(payload)}}
        predictions = [predict_longtable(s, ReplayModel([{"plan": self.plan}, self.read]), self.tokenizer,
                                         Protocol("L2"), view=view) for s in (a, b)]
        result = paired_diagnostic({self.sample["id"]: self.sample}, pair, predictions, exact)
        self.assertTrue(result["both_correct"])
        self.assertEqual([r["rows"][0][0] for r in result["fixed_first_plan_replay"]], ["Xiaomi", "Apple"])

    # 回归用例：test_condition_quote_is_bound_to_original_question
    # 应相等：bind_condition_quotes('查找 2020 年', [condition])[0]['question_span']，[3, 7]
    # 应抛出含指定原因的异常：ContractError，'does not occur'
    # 应抛出含指定原因的异常：ContractError，'ambiguous'
    def test_condition_quote_is_bound_to_original_question(self):
        condition = {"id": "year", "quote": "2020", "question_span": [0, 4],
                     "operation": {"op": "filter", "column": "c0", "cmp": "eq", "value": "2020"}}
        self.assertEqual(bind_condition_quotes("查找 2020 年", [condition])[0]["question_span"], [3, 7])
        with self.assertRaisesRegex(ContractError, "does not occur"):
            bind_condition_quotes("查找 2021 年", [condition])
        with self.assertRaisesRegex(ContractError, "ambiguous"):
            bind_condition_quotes("比较 2020 年与 2020 年", [condition])

    # 回归用例：test_teacher_row_domain_gate_and_review
    # 应相等：baseline['status']，'failed'
    # 应相等：review['answer']，[1]
    # 应成立：report['run_llm_review']
    def test_teacher_row_domain_gate_and_review(self):
        question = "How many A items?"
        count = {"op": "group_aggregate", "aggregations": [{"fn": "count", "as": "n"}]}
        answer = {"op": "answer", "columns": ["n"]}
        restrict = {"op": "filter", "column": "c0", "cmp": "eq", "value": "A"}
        sample = {"id": "t", "dataset": "wikitq", "split": "validation", "task": "short",
                  "question": question, "table": {"headers": ["Item"], "rows": [["A"], ["B"]]},
                  "reference": ["1"]}
        declarations = {"conditions": [{"id": "count", "question_span": [0, 8], "quote": "How many",
                                        "operation": count}], "row_domains": {"count": ["r1"]}}
        candidate = {"operations": [restrict, count, answer]}
        baseline = teacher_prediction(sample, ReplayModel([candidate, candidate]), "row_domains", declarations)
        review = teacher_prediction(sample, ReplayModel([candidate]), "llm_review", declarations,
                                    ReplayModel([{"allow": True}]))
        self.assertEqual(baseline["status"], "failed")
        self.assertEqual(review["answer"], [1])
        judgments = [{"dataset": "wikitq", "id": "t", "candidate_hash": baseline["attempts"][0]["candidate_hash"],
                      "legal": True, "condition_omitted": False, "row_domain_error": False,
                      "evidence_supported": True}]
        report = teacher_audit([sample], [baseline, review], judgments, split="validation")
        self.assertTrue(report["run_llm_review"])
        self.assertEqual(report["groups"]["llm_review"]["false_reject_after"], 0)


if __name__ == "__main__":
    unittest.main()
