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


def sample():return example('wikitq','train','x','t',['v'],[['A']],'value?',['A'])


class RecoveryTests(unittest.TestCase):
    def test_successful_primary_never_calls_backup(self):
        class Never:
            def generate(self,messages):raise AssertionError('Backup must not be called')
        p,r=predict_with_recovery(sample(),ReplayModel([{'operations':[{'op':'answer','columns':['c0']}]}]),Never(),settings('B5'))
        self.assertEqual(r['answer'],['A']);self.assertTrue(r['executable_answer'])
        self.assertFalse(r['routing']['fallback_attempted']);self.assertEqual(r['evidence'],p['evidence'])

    def test_failed_primary_is_recovered_without_false_evidence_claim(self):
        p,r=predict_with_recovery(sample(),ReplayModel(['bad']),ReplayModel([{'answer':['A']}]),settings('B5',max_repairs=0))
        self.assertEqual(p['status'],'failed');self.assertEqual(r['status'],'ok')
        self.assertEqual(r['answer_source'],'direct_fallback');self.assertFalse(r['executable_answer'])
        self.assertEqual(r['operations'],[]);self.assertEqual(r['evidence'],[])
        self.assertEqual(r['metrics']['model_calls'],2);self.assertIsNone(r['metrics']['input_tokens'])
        self.assertEqual(r['primary_attempt'],p)
        self.assertEqual(audit_trajectory(sample(),r)['decision'],'rejected')

    def test_both_failures_and_costs_remain_visible(self):
        p,r=predict_with_recovery(sample(),ReplayModel(['bad']),ReplayModel(['bad']),settings('B5',max_repairs=0))
        self.assertEqual(r['status'],'failed');self.assertEqual(r['metrics']['model_calls'],2)
        self.assertEqual(r['primary_attempt']['status'],'failed');self.assertEqual(r['fallback_attempt']['status'],'failed')

    def test_reference_never_controls_routing_or_model_inputs(self):
        class Spy(ReplayModel):
            def generate(self,messages):
                if 'REFERENCE_SENTINEL' in str(messages):raise AssertionError('Reference leaked')
                return super().generate(messages)
        s={**sample(),'reference':['REFERENCE_SENTINEL']}
        _,r=predict_with_recovery(s,Spy(['bad']),Spy([{'answer':['A']}]),settings('B5',max_repairs=0))
        self.assertEqual(r['answer'],['A'])

    def test_successful_primary_or_mismatched_source_cannot_be_overridden(self):
        s=sample();p=predict(s,ReplayModel([{'operations':[{'op':'answer','columns':['c0']}]}]),settings('B5'))
        q=predict(s,ReplayModel([{'answer':['B']}]),settings('B0'))
        with self.assertRaises(ValueError):combine_recovery(p,q)
        failed=predict(s,ReplayModel(['bad']),settings('B5',max_repairs=0))
        with self.assertRaises(ValueError):combine_recovery(failed,{**q,'id':'other'})

    def test_analysis_distinguishes_coverage_from_executable_chain(self):
        _,r=predict_with_recovery(sample(),ReplayModel(['bad']),ReplayModel([{'answer':['A']}]),settings('B5',max_repairs=0))
        row=summarize_run([sample()],[r])['rows'][0]
        self.assertEqual(row['observed_score'],1);self.assertEqual(row['pipeline_success_rate'],1)
        self.assertEqual(row['executable_chain_rate'],0);self.assertEqual(row['fallback_answers'],1)
        self.assertIsNone(compare_runs([sample()],[r],[r],repeats=20)[0]['latency_ratio_a_over_b'])
        self.assertEqual(evaluate([r],[sample()])['wikitq']['execution_success_rate'],0)

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
