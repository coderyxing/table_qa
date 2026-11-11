# 测试范围：验证对应模块的数据隔离、输入约束与历史回归行为。
# 运行方式：python -m unittest discover -s tests -p test_freeform_metrics.py -v
# 阅读顺序：先看样本或替身模型的构造，再看每个用例前的预期约束。
import importlib.util
import math
import unittest
from tableqa.reporting import freeform_texts,freeform_metrics,compare_freeform


AVAILABLE=importlib.util.find_spec('sacrebleu') is not None and importlib.util.find_spec('rouge_score') is not None


# 构造用于自动指标对齐和失败计分的参考样本集合。
# 返回形式含：列表记录。
def data():return [{'dataset':'fetaqa','id':uid,'reference':'the red team won six games'} for uid in ['a','b']]
# 逐题生成并产出预测，使调用层可以流式保存。
# 返回形式含：列表记录。
def predictions():return [{'dataset':'fetaqa','id':uid,'status':'ok','answer':'the red team won six games'} for uid in ['b','a']]


# FreeformAlignmentTests：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
class FreeformAlignmentTests(unittest.TestCase):
    # 回归用例：test_missing_duplicate_and_unknown_outputs_cannot_inflate_scores
    # 应相等：aligned['keys']，[('fetaqa', 'a'), ('fetaqa', 'b')]
    # 应抛出预期异常：ValueError
    def test_missing_duplicate_and_unknown_outputs_cannot_inflate_scores(self):
        for rows in [predictions()[:1],predictions()+predictions()[:1],[{**p,'id':'x'} for p in predictions()]]:
            with self.assertRaises(ValueError):freeform_texts(rows,data())
        aligned=freeform_texts(predictions(),data())
        self.assertEqual(aligned['keys'],[('fetaqa','a'),('fetaqa','b')])

    # 回归用例：test_failures_are_empty_even_when_answer_field_contains_gold
    # 应相等：result['hypotheses'][1]，''
    # 应相等：result['failures']，1
    # 应相等：freeform_texts(rows, data())['invalid_answer_text']，1
    def test_failures_are_empty_even_when_answer_field_contains_gold(self):
        rows=predictions();rows[0]['status']='failed'
        result=freeform_texts(rows,data())
        self.assertEqual(result['hypotheses'][1],'');self.assertEqual(result['failures'],1)
        rows=predictions();rows[0]['answer']={'text':'the red team won six games'}
        self.assertEqual(freeform_texts(rows,data())['invalid_answer_text'],1)

    # 回归用例：test_transfer_selection_reads_no_gold_or_highlights_and_rejects_test
    # 应相等：[r['id'] for r in first]，[r['id'] for r in select_samples(changed)]
    # 应抛出预期异常：ValueError
    def test_transfer_selection_reads_no_gold_or_highlights_and_rejects_test(self):
        from copy import deepcopy
        from scripts.fetaqa_transfer_pilot import select_samples
        rows=[{'dataset':'fetaqa','split':'validation','id':str(i),'reference':'PRIVATE',
               'table':{'headers':['x'],'rows':[[str(i)]]},'highlighted_cells':[[0,0]]} for i in range(16)]
        first=select_samples(rows);changed=deepcopy(rows)
        for r in changed:r['reference']='changed';r['highlighted_cells']=[]
        self.assertEqual([r['id'] for r in first],[r['id'] for r in select_samples(changed)])
        changed[0]['split']='test'
        with self.assertRaises(ValueError):select_samples(changed)


# FreeformActualMetricTests：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
@unittest.skipUnless(AVAILABLE,'Optional automatic-metric dependencies are not installed')
class FreeformActualMetricTests(unittest.TestCase):
    # 回归用例：test_four_metrics_and_failed_case_denominator
    # 应在浮点容差内相等：full['sacrebleu']，100
    # 应在浮点容差内相等：half['sacrebleu']，100 / math.e
    # 应为 None：half['human_factual_consistency']
    def test_four_metrics_and_failed_case_denominator(self):
        rows=predictions();full=freeform_metrics(rows,data())
        self.assertAlmostEqual(full['sacrebleu'],100)
        for key in ['rouge1_f1','rouge2_f1','rougeL_f1']:self.assertAlmostEqual(full[key],1)
        rows[0]['status']='failed';half=freeform_metrics(rows,data())
        self.assertAlmostEqual(half['sacrebleu'],100/math.e)
        for key in ['rouge1_f1','rouge2_f1','rougeL_f1']:self.assertAlmostEqual(half[key],.5)
        self.assertIsNone(half['human_factual_consistency']);self.assertFalse(half['semantic_correctness_proven'])

    # 回归用例：test_identical_paired_outputs_have_zero_differences_and_intervals
    # 应相等：metric['difference_b_minus_a']，0
    # 应相等：metric['ci95']，[0, 0]
    def test_identical_paired_outputs_have_zero_differences_and_intervals(self):
        result=compare_freeform(predictions(),predictions(),data(),repeats=10)
        for metric in result['metrics'].values():
            self.assertEqual(metric['difference_b_minus_a'],0);self.assertEqual(metric['ci95'],[0,0])


if __name__=='__main__':unittest.main()
