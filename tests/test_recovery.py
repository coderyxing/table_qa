# 测试范围：把主操作模型与失败回退模型按固定规则组合，完整计入恢复开销。
# 运行方式：python -m unittest discover -s tests -p test_recovery.py -v
# 阅读顺序：先看样本或替身模型的构造，再看每个用例前的预期约束。
import unittest
from tableqa.data import example
from tableqa.models import ReplayModel
from tableqa.experiments import settings
from tableqa.inference import predict
from tableqa.recovery import combine_recovery,predict_with_recovery
from tableqa.analysis import summarize_run,compare_runs
from tableqa.evaluation import evaluate
from tableqa.quality import audit_trajectory
from tableqa.core import TableState
from tableqa.optimization import optimize_operations
from tableqa.distillation import build_records


# 构造当前测试使用的独立规范样本，固定题面、原表和参考结果。
def sample():return example('wikitq','train','x','t',['v'],[['A']],'value?',['A'])


# RecoveryTests：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
class RecoveryTests(unittest.TestCase):
    # 回归用例：test_successful_primary_never_calls_backup
    # 应相等：r['answer']，['A']
    # 应成立：r['executable_answer']
    # 应不成立：r['routing']['fallback_attempted']
    def test_successful_primary_never_calls_backup(self):
        # Never：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
        class Never:
            # 按当前适配接口生成或重放响应，同时保留调用所需的观测状态。
            # 输入：messages（模型可见消息）。
            # 异常约束：Backup must not be called
            def generate(self,messages):raise AssertionError('Backup must not be called')
        p,r=predict_with_recovery(sample(),ReplayModel([{'operations':[{'op':'answer','columns':['c0']}]}]),Never(),settings('B5'))
        self.assertEqual(r['answer'],['A']);self.assertTrue(r['executable_answer'])
        self.assertFalse(r['routing']['fallback_attempted']);self.assertEqual(r['evidence'],p['evidence'])

    # 回归用例：test_failed_primary_is_recovered_without_false_evidence_claim
    # 应相等：p['status']，'failed'
    # 应相等：r['status']，'ok'
    # 应相等：r['answer_source']，'direct_fallback'
    def test_failed_primary_is_recovered_without_false_evidence_claim(self):
        p,r=predict_with_recovery(sample(),ReplayModel(['bad']),ReplayModel([{'answer':['A']}]),settings('B5',max_repairs=0))
        self.assertEqual(p['status'],'failed');self.assertEqual(r['status'],'ok')
        self.assertEqual(r['answer_source'],'direct_fallback');self.assertFalse(r['executable_answer'])
        self.assertEqual(r['operations'],[]);self.assertEqual(r['evidence'],[])
        self.assertEqual(r['metrics']['model_calls'],2);self.assertIsNone(r['metrics']['input_tokens'])
        self.assertEqual(r['primary_attempt'],p)
        self.assertEqual(audit_trajectory(sample(),r)['decision'],'rejected')

    # 回归用例：test_both_failures_and_costs_remain_visible
    # 应相等：r['status']，'failed'
    # 应相等：r['metrics']['model_calls']，2
    # 应相等：r['primary_attempt']['status']，'failed'
    def test_both_failures_and_costs_remain_visible(self):
        p,r=predict_with_recovery(sample(),ReplayModel(['bad']),ReplayModel(['bad']),settings('B5',max_repairs=0))
        self.assertEqual(r['status'],'failed');self.assertEqual(r['metrics']['model_calls'],2)
        self.assertEqual(r['primary_attempt']['status'],'failed');self.assertEqual(r['fallback_attempt']['status'],'failed')

    # 回归用例：test_reference_never_controls_routing_or_model_inputs
    # 应相等：r['answer']，['A']
    def test_reference_never_controls_routing_or_model_inputs(self):
        # Spy：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
        class Spy(ReplayModel):
            # 按当前适配接口生成或重放响应，同时保留调用所需的观测状态。
            # 输入：messages（模型可见消息）。
            # 相关调用：super().generate。
            # 异常约束：Reference leaked
            def generate(self,messages):
                if 'REFERENCE_SENTINEL' in str(messages):raise AssertionError('Reference leaked')
                return super().generate(messages)
        s={**sample(),'reference':['REFERENCE_SENTINEL']}
        _,r=predict_with_recovery(s,Spy(['bad']),Spy([{'answer':['A']}]),settings('B5',max_repairs=0))
        self.assertEqual(r['answer'],['A'])

    # 回归用例：test_successful_primary_or_mismatched_source_cannot_be_overridden
    # 应抛出预期异常：ValueError
    def test_successful_primary_or_mismatched_source_cannot_be_overridden(self):
        s=sample();p=predict(s,ReplayModel([{'operations':[{'op':'answer','columns':['c0']}]}]),settings('B5'))
        q=predict(s,ReplayModel([{'answer':['B']}]),settings('B0'))
        with self.assertRaises(ValueError):combine_recovery(p,q)
        failed=predict(s,ReplayModel(['bad']),settings('B5',max_repairs=0))
        with self.assertRaises(ValueError):combine_recovery(failed,{**q,'id':'other'})

    # 回归用例：test_analysis_distinguishes_coverage_from_executable_chain
    # 应相等：row['observed_score']，1
    # 应相等：row['pipeline_success_rate']，1
    # 应相等：row['executable_chain_rate']，0
    def test_analysis_distinguishes_coverage_from_executable_chain(self):
        _,r=predict_with_recovery(sample(),ReplayModel(['bad']),ReplayModel([{'answer':['A']}]),settings('B5',max_repairs=0))
        row=summarize_run([sample()],[r])['rows'][0]
        self.assertEqual(row['observed_score'],1);self.assertEqual(row['pipeline_success_rate'],1)
        self.assertEqual(row['executable_chain_rate'],0);self.assertEqual(row['fallback_answers'],1)
        self.assertIsNone(compare_runs([sample()],[r],[r],repeats=20)[0]['latency_ratio_a_over_b'])
        self.assertEqual(evaluate([r],[sample()])['wikitq']['execution_success_rate'],0)

    # 回归用例：test_guarded_search_retains_task_conditions_even_when_data_equal
    # 应相等：len(loose)，1
    # 应相等：guarded，ops
    # 应成立：records
    def test_guarded_search_retains_task_conditions_even_when_data_equal(self):
        s=sample();state=TableState.from_table(s['table'])
        ops=[{'op':'filter','column':'c0','cmp':'eq','value':'A'}, {'op':'answer','columns':['c0']}]
        loose,_=optimize_operations(state,ops)
        guarded,_=optimize_operations(state,ops,preserve_kinds=('filter','sort','select','group_aggregate'))
        self.assertEqual(len(loose),1);self.assertEqual(guarded,ops)
        records=build_records(s,ops,compression_policy='guarded_search')
        self.assertTrue(records)
        with self.assertRaises(ValueError):build_records(s,ops,compression_policy='guarded_search',prepared_operations=loose)


if __name__=='__main__':unittest.main()
