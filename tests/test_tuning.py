# 测试范围：按固定规则选择准确率开发和确认队列，记录历史已接触来源。
# 运行方式：python -m unittest discover -s tests -p test_tuning.py -v
# 阅读顺序：先看样本或替身模型的构造，再看每个用例前的预期约束。
import unittest
from tableqa.data import example,table_fingerprint
from tableqa.tuning import accuracy_cohorts


# TuningTests：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
class TuningTests(unittest.TestCase):
    # 逐题编译允许来源的监督记录，并同时积累排除原因。
    # 返回形式含：列表记录。
    def records(self):
        return [example(d,'validation',f'{d}-{i}-{q}',str(i),['v'],[[f'{d}-{i}']],f'q{q}',q)
                for d in ['wikitq','tabfact'] for i in range(12) for q in range(2)]
    # 回归用例：test_whole_table_independence_previous_exclusion_and_determinism
    # 应相等：groups，accuracy_cohorts(list(reversed(rows)), prior, 3)
    # 应相等：len(a)，6
    # 应相等：len(b)，6
    def test_whole_table_independence_previous_exclusion_and_determinism(self):
        rows=self.records();prior=[rows[0]];groups=accuracy_cohorts(rows,prior,3)
        self.assertEqual(groups,accuracy_cohorts(list(reversed(rows)),prior,3))
        a={table_fingerprint(r['table']) for r in groups['tune']}
        b={table_fingerprint(r['table']) for r in groups['confirmation']}
        self.assertEqual(len(a),6);self.assertEqual(len(b),6);self.assertFalse(a&b)
        self.assertNotIn(table_fingerprint(prior[0]['table']),a|b)
    # 回归用例：test_labels_do_not_choose_inputs_and_insufficient_data_fails
    # 应相等：[r['id'] for r in a['tune']]，[r['id'] for r in b['tune']]
    # 应抛出预期异常：ValueError
    def test_labels_do_not_choose_inputs_and_insufficient_data_fails(self):
        rows=self.records();changed=[{**r,'reference':'different'} for r in rows]
        a=accuracy_cohorts(rows,[],3);b=accuracy_cohorts(changed,[],3)
        self.assertEqual([r['id'] for r in a['tune']],[r['id'] for r in b['tune']])
        with self.assertRaises(ValueError):accuracy_cohorts(rows,[],7)
        with self.assertRaises(ValueError):accuracy_cohorts([{**rows[0],'split':'test'}],[],1)
