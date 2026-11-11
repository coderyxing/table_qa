# 测试范围：验证对应模块的数据隔离、输入约束与历史回归行为。
# 运行方式：python -m unittest discover -s tests -p test_tool_curriculum.py -v
# 阅读顺序：先看样本或替身模型的构造，再看每个用例前的预期约束。
import unittest
from scripts.build_tool_curriculum import make_pool
from tableqa.core import TableState, run_plan
from tableqa.data import digest
from tableqa.evaluation import diagnostic_score
from tableqa.semantic_columns import uses_extraction
from tableqa.quantities import uses_quantities


# CurriculumTests：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
class CurriculumTests(unittest.TestCase):
    # 回归用例：test_all_six_families_match_independent_answers_and_retain_raw_tables
    # 应相等：len(samples)，24
    # 应相等：len({r['template_family'] for r in samples})，6
    # 应相等：{r['reference'] for r in samples if r['dataset'] == 'tabfact'}，{0, 1}
    def test_all_six_families_match_independent_answers_and_retain_raw_tables(self):
        samples, traces = make_pool(4)
        self.assertEqual(len(samples), 24)
        self.assertEqual(len({r['template_family'] for r in samples}), 6)
        self.assertEqual({r['reference'] for r in samples if r['dataset']=='tabfact'}, {0, 1})
        self.assertTrue(any(uses_extraction(t['operations']) for t in traces))
        self.assertTrue(any(uses_quantities(t['operations']) for t in traces))
        for sample, trace in zip(samples, traces):
            before = digest(sample)
            result, _ = run_plan(TableState.from_table(sample['table']), trace['operations'])
            self.assertEqual(diagnostic_score({'status': 'ok', 'answer': result.answer}, sample), 1)
            self.assertEqual(before, digest(sample))
            self.assertTrue(sample['synthetic']); self.assertEqual(sample['split'], 'train')
            self.assertLessEqual(len(trace['operations']), 8)

    # 回归用例：test_reproducible_generation_and_missing_rank_precondition
    # 应相等：digest(a)，digest(b)
    # 应不同：digest(a)，digest(c)
    # 应抛出预期异常：ValueError
    def test_reproducible_generation_and_missing_rank_precondition(self):
        a, traces = make_pool(2, 12); b, _ = make_pool(2, 12); c, _ = make_pool(2, 13)
        self.assertEqual(digest(a), digest(b)); self.assertNotEqual(digest(a), digest(c))
        for sample, trace in zip(a, traces):
            if sample['template_family'] == 'ordinal_scope':
                self.assertEqual(trace['operations'][0]['op'], 'filter')
                self.assertEqual(trace['operations'][0]['value'], '\u2014')
                self.assertEqual(sample['table']['rows'][0][1], '\u2014')
        with self.assertRaises(ValueError): make_pool(0)


if __name__ == '__main__': unittest.main()
