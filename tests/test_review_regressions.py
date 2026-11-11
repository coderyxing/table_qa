# 测试范围：验证对应模块的数据隔离、输入约束与历史回归行为。
# 运行方式：python -m unittest discover -s tests -p test_review_regressions.py -v
# 阅读顺序：先看样本或替身模型的构造，再看每个用例前的预期约束。
from dataclasses import fields
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
from threading import Barrier
import unittest
from unittest.mock import patch

from tableqa.chapter3 import teacher_audit, teacher_prediction
from tableqa.chapter4 import validate_pair_sources
from tableqa.core import ExecutionError, TableState, execute
from tableqa.data import example
from tableqa.experiments import manifest, settings
from tableqa.inference import Settings, predict
from tableqa.longtable import LongTable, QueryError
from tableqa.models import CachedModel, ReplayModel, make_model


# ReviewRegressions：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
class ReviewRegressions(unittest.TestCase):
    # 验证 API 配置的 max_new_tokens 真正传给后端，避免实验生成预算失效。
    def test_api_model_receives_configured_generation_budget(self):
        with patch("tableqa.models.APIModel") as api:
            make_model({"backend": "api", "base_url": "https://example.invalid",
                        "model": "fixture", "max_new_tokens": 37})
        api.assert_called_once_with("https://example.invalid", "fixture", "TABLEQA_API_KEY", 37)

    # 将两个线程同时停在原子提交前，验证同一缓存键拥有独立临时文件。
    def test_cache_concurrent_writes_use_independent_temporary_files(self):
        barrier = Barrier(2)
        original_replace = Path.replace

        # 仅在测试内同步两个提交，不改变 replace 的实际文件系统行为。
        def synchronized_replace(source, target):
            barrier.wait(timeout=5)
            return original_replace(source, target)

        class Backend:
            # 后端无需模型权重，两个线程对同一消息生成相同内容。
            def generate(self, messages):
                return {"text": '{"answer":1}', "cache_hit": False}

        with tempfile.TemporaryDirectory() as directory:
            model = CachedModel(Backend(), directory, {"model": "fixture"})
            with patch.object(Path, "replace", synchronized_replace):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    results = list(pool.map(lambda _: model.generate([]), range(2)))
            self.assertEqual(len(results), 2)
            self.assertTrue(model.generate([])["cache_hit"])
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 1)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    # 后端抛异常的调用仍消耗时间；token 未知应明确记录为 None。
    def test_failed_generation_attempt_is_counted_with_unknown_tokens(self):
        sample = example("wikitq", "validation", "one", "t", ["Name"], [["A"]], "Which name?", ["A"])
        model = unittest.mock.Mock()
        model.generate.side_effect = RuntimeError("fixture generation failed")
        outcome = predict(sample, model, settings("B5"))
        self.assertEqual(outcome["status"], "failed")
        self.assertEqual(outcome["metrics"]["model_calls"], 1)
        self.assertGreater(outcome["metrics"]["model_seconds"], 0)
        self.assertIsNone(outcome["metrics"]["input_tokens"])
        self.assertIsNone(outcome["metrics"]["output_tokens"])

    # 回归用例：test_probe_ids_preserve_each_column_and_reject_ambiguous_legacy_id
    # 应相等：(cities['id'], brands['id'])，('probe_1', 'probe_2')
    # 应相等：[row[0] for row in rows]，probe['values']
    # 应抛出含指定原因的异常：QueryError，'same original column'
    def test_probe_ids_preserve_each_column_and_reject_ambiguous_legacy_id(self):
        table = {"headers": ["City", "Brand"], "rows": [["Paris", "Apple"], ["Rome", "Nokia"]]}
        query = LongTable(table, "Compare Paris and Rome, Apple and Nokia")
        self.addCleanup(query.close)
        cities = query.probe({"column": "c0"})
        brands = query.probe({"column": "c1"})
        self.assertEqual((cities["id"], brands["id"]), ("probe_1", "probe_2"))
        for column, probe in (("c0", cities), ("c1", brands)):
            rows = query.execute({
                "filters": [{"column": column, "cmp": "in", "value": {"result_id": probe["id"]},
                             "quote": query.question}],
                "select": [{"column": column}],
            })
            self.assertEqual([row[0] for row in rows], probe["values"])
        with self.assertRaisesRegex(QueryError, "same original column"):
            query.execute({"filters": [{"column": "c0", "cmp": "in", "value": {"result_id": "probe"}}],
                           "select": [{"column": "c0"}]})

    # 回归用例：test_incomplete_reviews_disable_full_metrics_and_review_gate
    # 应相等：group['review_coverage']，0.5
    # 应相等：group['missing_reviews']，1
    # 应为 None：group['false_reject_before']
    def test_incomplete_reviews_disable_full_metrics_and_review_gate(self):
        data = [example("wikitq", "validation", str(i), str(i), ["Item"], [["A"]],
                        "Which item?", ["A"]) for i in range(2)]
        candidate = {"operations": [{"op": "answer", "columns": ["c0"]}]}
        outcomes = [teacher_prediction(s, ReplayModel([candidate]), "row_domains", {}) for s in data]
        reviews = [{"dataset": "wikitq", "id": "0", "candidate_hash": outcomes[0]["attempts"][0]["candidate_hash"],
                    "legal": True, "condition_omitted": False, "row_domain_error": False, "evidence_supported": True}]
        report = teacher_audit(data, outcomes, reviews, split="validation")
        group = report["groups"]["row_domains"]
        self.assertEqual(group["review_coverage"], .5)
        self.assertEqual(group["missing_reviews"], 1)
        self.assertIsNone(group["false_reject_before"])
        self.assertEqual(group["reviewed_subset_rates"]["false_reject_before"], 0)
        self.assertIsNone(report["run_llm_review"])

    # 回归用例：test_manifest_keeps_metadata_outside_prediction_settings
    # 应成立：any(('implementation' in row for row in runs))
    # 应成立：any(('ablation' in row for row in runs))
    # 应相等：set(row['prediction_settings'])，names
    def test_manifest_keeps_metadata_outside_prediction_settings(self):
        names = {field.name for field in fields(Settings)}
        runs = manifest()
        self.assertTrue(any("implementation" in row for row in runs))
        self.assertTrue(any("ablation" in row for row in runs))
        for row in runs:
            self.assertEqual(set(row["prediction_settings"]), names)
            if row.get("parameter") in {"max_block", "max_repairs"}:
                self.assertEqual(row["prediction_settings"][row["parameter"]], row["value"])
        with self.assertRaisesRegex(ValueError, "Unknown setting implementation"):
            settings("B5", implementation="adapted")

    # 回归用例：test_repair_omits_long_failed_response_and_uses_two_messages
    # 应相等：outcome['status']，'ok'
    # 应相等：outcome['answer']，['A']
    # 应相等：len(model.prompts[1])，2
    def test_repair_omits_long_failed_response_and_uses_two_messages(self):
        sample = example("wikitq", "validation", "one", "t", ["Name"], [["A"]], "Which name?", ["A"])

        # BudgetReplay：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
        class BudgetReplay(ReplayModel):
            # 初始化 BudgetReplay 的依赖及持久状态；具体预算和身份字段见赋值。
            def __init__(self):
                super().__init__([
                    {"operations": [{"op": "unknown"}], "padding": "x" * 6000},
                    {"operations": [{"op": "answer", "columns": ["c0"]}]},
                ])
                self.prompts = []

            # 按当前适配接口生成或重放响应，同时保留调用所需的观测状态。
            # 输入：messages（模型可见消息）。
            # 相关调用：super().generate。
            def generate(self, messages):
                self.prompts.append(messages)
                if len(self.prompts) > 1:
                    self.assert_budget(messages)
                return super().generate(messages)

            # 核对修复提示是否仍在测试预算内，暴露完整错误输出造成的膨胀。
            # 输入：messages（模型可见消息）。
            # 异常约束：Repair prompt exceeded fixture budget
            def assert_budget(self, messages):
                original = sum(len(m["content"]) for m in self.prompts[0])
                if sum(len(m["content"]) for m in messages) > original + 1000:
                    raise RuntimeError("Repair prompt exceeded fixture budget")

        model = BudgetReplay()
        outcome = predict(sample, model, settings("B5"))
        self.assertEqual(outcome["status"], "ok")
        self.assertEqual(outcome["answer"], ["A"])
        self.assertEqual(len(model.prompts[1]), 2)
        self.assertNotIn("padding", model.prompts[1][-1]["content"])
        self.assertIn("repair_feedback", model.prompts[1][-1]["content"])

    # 回归用例：test_numeric_skip_is_explicit_and_invalid_constants_still_fail
    # 应相等：execute(state, {**op, 'invalid_number': 'skip'}).row_ids，['r1', 'r2']
    # 应相等：execute(state, {**op, 'invalid_number': 'skip', 'cmp': 'not_in', 'value': [10]}).row_ids，['r1']
    # 应相等：state.row_ids，['r0', 'r1', 'r2']
    def test_numeric_skip_is_explicit_and_invalid_constants_still_fail(self):
        state = TableState.from_table({"headers": ["Number"], "rows": [["N/A"], [0], [10]]})
        op = {"op": "filter", "column": "c0", "cmp": "gt", "type": "number", "value": -1}
        with self.assertRaises(ExecutionError):
            execute(state, op)
        self.assertEqual(execute(state, {**op, "invalid_number": "skip"}).row_ids, ["r1", "r2"])
        with self.assertRaises(ExecutionError):
            execute(state, {**op, "invalid_number": "skip", "value": "bad"})
        empty = TableState.from_table({"headers": ["Number"], "rows": []})
        with self.assertRaises(ExecutionError):
            execute(empty, {**op, "invalid_number": "skip", "value": "bad"})
        self.assertEqual(execute(state, {**op, "invalid_number": "skip", "cmp": "not_in", "value": [10]}).row_ids, ["r1"])
        self.assertEqual(state.row_ids, ["r0", "r1", "r2"])

    # 回归用例：test_leakage_checks_table_fingerprint_and_terminal_punctuation
    # 应相等：set(validate_pair_sources(train, test, 'test', dev))，{'fresh'}
    def test_leakage_checks_table_fingerprint_and_terminal_punctuation(self):
        # 构造带划分、原表身份和指纹的泄露检查样本。
        # 输入：uid；split（数据划分）；table_id；fingerprint；question（原题文本）。
        # 返回形式含：字段 id、dataset、split、table_id、table_fingerprint、question。
        def row(uid, split, table_id, fingerprint, question):
            return {"id": uid, "dataset": "databench", "split": split, "table_id": table_id,
                    "table_fingerprint": fingerprint, "question": question}

        train = [row("train", "train", "a", "hash-a", "Which city?")]
        dev = [row("dev", "validation", "b", "hash-b", "What brand?")]
        test = [
            row("same-table", "test", "renamed", "hash-a", "A wholly different question"),
            row("punctuation", "test", "c", "hash-c", "  WHICH   CITY？！ "),
            row("fresh", "test", "d", "hash-d", "What is the total?"),
        ]
        self.assertEqual(set(validate_pair_sources(train, test, "test", dev)), {"fresh"})


if __name__ == "__main__":
    unittest.main()
