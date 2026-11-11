# 测试范围：对完整或正在写入的预测做来源绑定、官方判分导入和配对统计，输出可追溯的开发报告。
# 运行方式：python -m unittest discover -s tests -p test_analysis.py -v
# 阅读顺序：先看样本或替身模型的构造，再看每个用例前的预期约束。
import math
from pathlib import Path
import tempfile
import unittest
from tableqa.analysis import summarize_run,compare_runs,write_analysis,failure_category,record_hash,read_completed_prefix
from tableqa.data import example


# 构造当前测试使用的独立规范样本，固定题面、原表和参考结果。
# 输入：uid；dataset。
def sample(uid='x',dataset='tabfact'):
    return example(dataset,'train',uid,'t',['v'],[[1]],'value?',1 if dataset=='tabfact' else ['1'])


# 构造与样本身份对应的预测夹具，供完整覆盖及统计测试使用。
# 输入：uid；answer；status；seconds；dataset；mode（实验模式）。
# 返回形式含：字段 id、dataset、mode、status、answer、failure、metrics。
def prediction(uid='x',answer=1,status='ok',seconds=2.,dataset='tabfact',mode='B0'):
    return {'id':uid,'dataset':dataset,'mode':mode,'status':status,'answer':answer,
            'failure':None if status=='ok' else {'type':'ValueError','message':'Input exceeds configured context'},
            'metrics':{'total_seconds':seconds,'model_calls':1,'input_tokens':20,'output_tokens':2,'cache_hits':0}}


# AnalysisTests：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
class AnalysisTests(unittest.TestCase):
    # 回归用例：test_failures_are_in_denominator_and_latency
    # 应相等：rows[0]['observed_score']，0.5
    # 应相等：rows[0]['all_case_p50_seconds']，1.0
    # 应相等：rows[0]['success_only_p50_seconds']，2.0
    def test_failures_are_in_denominator_and_latency(self):
        rows=summarize_run([sample('a'),sample('b')],[prediction('a'),prediction('b',status='failed',seconds=0.)])['rows']
        self.assertEqual(rows[0]['observed_score'],.5)
        self.assertEqual(rows[0]['all_case_p50_seconds'],1.)
        self.assertEqual(rows[0]['success_only_p50_seconds'],2.)
        self.assertEqual(rows[0]['failure_categories'],{'context_limit':1})
    # 回归用例：test_response_validity_is_not_executable_chain
    # 应为 None：row['executable_chain_rate']
    # 应相等：row['response_format_rate']，1.0
    def test_response_validity_is_not_executable_chain(self):
        row=summarize_run([sample()],[prediction()])['rows'][0]
        self.assertIsNone(row['executable_chain_rate']); self.assertEqual(row['response_format_rate'],1.)
    # 回归用例：test_partial_run_requires_explicit_option
    # 应成立：out['partial']
    # 应相等：out['missing_count']，1
    # 应相等：out['rows'][0]['completed']，1
    def test_partial_run_requires_explicit_option(self):
        with self.assertRaises(ValueError):summarize_run([sample('a'),sample('b')],[prediction('a')])
        out=summarize_run([sample('a'),sample('b')],[prediction('a')],allow_partial=True)
        self.assertTrue(out['partial']); self.assertEqual(out['missing_count'],1)
        self.assertEqual(out['rows'][0]['completed'],1)
    # 回归用例：test_unknown_or_duplicate_ids_are_rejected
    # 应抛出预期异常：ValueError
    def test_unknown_or_duplicate_ids_are_rejected(self):
        with self.assertRaises(ValueError):summarize_run([sample()],[prediction('other')])
        with self.assertRaises(ValueError):summarize_run([sample()],[prediction(),prediction()])
        with self.assertRaises(ValueError):summarize_run([sample(),sample()],[prediction()])
    # 回归用例：test_official_denotation_replaces_only_wtq_diagnostic
    # 应相等：summarize_run([ref], [p])['rows'][0]['observed_score']，0
    # 应相等：row['observed_score']，1
    # 应相等：row['metric']，'official_denotation_accuracy'
    def test_official_denotation_replaces_only_wtq_diagnostic(self):
        ref=sample(dataset='wikitq'); p=prediction(answer=['1.0'],dataset='wikitq',mode='A4')
        report={'source_sha256':'source','converted_sha256':'port','scores':[{'dataset':'wikitq','id':'x','score':1}]}
        report.update(samples_structured_sha256=record_hash([ref]),predictions_structured_sha256=record_hash([p]))
        self.assertEqual(summarize_run([ref],[p])['rows'][0]['observed_score'],0)
        row=summarize_run([ref],[p],report)['rows'][0]
        self.assertEqual(row['observed_score'],1); self.assertEqual(row['metric'],'official_denotation_accuracy')
    # 回归用例：test_bad_official_coverage_is_rejected
    # 应抛出预期异常：ValueError
    def test_bad_official_coverage_is_rejected(self):
        report={'source_sha256':'source','converted_sha256':'port','scores':[]}
        with self.assertRaises(ValueError):summarize_run([sample(dataset='wikitq')],[prediction(dataset='wikitq')],report)
    # 回归用例：test_official_scores_cannot_be_reused_for_another_prediction
    # 应抛出含指定原因的异常：ValueError，'stale'
    def test_official_scores_cannot_be_reused_for_another_prediction(self):
        ref=sample(dataset='wikitq'); p=prediction(dataset='wikitq')
        report={'source_sha256':'source','converted_sha256':'port','scores':[{'dataset':'wikitq','id':'x','score':1}],
                'samples_structured_sha256':record_hash([ref]),'predictions_structured_sha256':record_hash([p])}
        changed={**p,'answer':'different'}
        with self.assertRaisesRegex(ValueError,'stale'): summarize_run([ref],[changed],report)
    # 回归用例：test_paired_direction_and_id_alignment
    # 应相等：row['difference_b_minus_a']，1
    # 应相等：row['ci95']，[1, 1]
    # 应相等：row['b_only_correct']，2
    def test_paired_direction_and_id_alignment(self):
        refs=[sample('a'),sample('b')]
        a=[prediction('a',answer=0),prediction('b',answer=0)]
        b=[prediction('b'),prediction('a')]
        row=compare_runs(refs,a,b,repeats=50)[0]
        self.assertEqual(row['difference_b_minus_a'],1)
        self.assertEqual(row['ci95'],[1,1]); self.assertEqual(row['b_only_correct'],2)
    # 回归用例：test_mixed_wtq_scoring_protocols_are_rejected
    # 应抛出预期异常：ValueError
    def test_mixed_wtq_scoring_protocols_are_rejected(self):
        report={'source_sha256':'source','converted_sha256':'port','scores':[{'dataset':'wikitq','id':'x','score':1}]}
        with self.assertRaises(ValueError):compare_runs([sample(dataset='wikitq')],[prediction(dataset='wikitq')],[prediction(dataset='wikitq')],report,None)
    # 回归用例：test_cached_latency_is_not_reported_as_acceleration
    # 应为 None：compare_runs([sample()], [p], [prediction()], repeats=20)[0]['latency_ratio_a_over_b']
    def test_cached_latency_is_not_reported_as_acceleration(self):
        p=prediction(); p['metrics']['cache_hits']=1
        self.assertIsNone(compare_runs([sample()],[p],[prediction()],repeats=20)[0]['latency_ratio_a_over_b'])
    # 回归用例：test_invalid_timing_is_rejected
    # 应抛出预期异常：ValueError
    def test_invalid_timing_is_rejected(self):
        for seconds in [float('nan'),-1,True]:
            with self.assertRaises(ValueError):summarize_run([sample()],[prediction(seconds=seconds)])
    # 回归用例：test_confusion_matrix_retains_invalid_labels
    # 应相等：row['confusion_matrix']，{'true_1_pred_invalid': 1}
    # 应相等：row['observed_score']，0
    def test_confusion_matrix_retains_invalid_labels(self):
        row=summarize_run([sample()],[prediction(answer='unknown')])['rows'][0]
        self.assertEqual(row['confusion_matrix'],{'true_1_pred_invalid':1})
        self.assertEqual(row['observed_score'],0)
    # 回归用例：test_report_artifacts
    # 应成立：(Path(directory) / 'summary.csv').exists()
    # 应包含：'开发'，(Path(directory) / 'report.md').read_text(encoding='utf-8')
    def test_report_artifacts(self):
        summary=summarize_run([sample()],[prediction()])
        with tempfile.TemporaryDirectory() as directory:
            write_analysis(directory,{'run':summary})
            self.assertTrue((Path(directory)/'summary.csv').exists())
            self.assertIn('开发', (Path(directory)/'report.md').read_text(encoding='utf-8'))
    # 回归用例：test_live_prefix_reader_keeps_only_completed_records
    # 应相等：read_completed_prefix(p)，[{'id': 'a'}]
    # 应相等：p.read_text(encoding='utf-8')，text
    def test_live_prefix_reader_keeps_only_completed_records(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'live.jsonl'; text='{"id":"a"}\n{"id":'
            p.write_text(text,encoding='utf-8')
            self.assertEqual(read_completed_prefix(p),[{'id':'a'}])
            self.assertEqual(p.read_text(encoding='utf-8'),text)
