import unittest
from copy import deepcopy
from tableqa.data import example
from tableqa.experiments import settings
from tableqa.inference import predict
from tableqa.models import ReplayModel
from tableqa.mechanisms import analyze_mechanisms, error_kind
from tableqa.analysis import record_hash


class MechanismTests(unittest.TestCase):
    def test_repaired_execution_is_not_necessarily_correct(self):
        sample = example('tabfact','train','x','t',['v'],[[1]],'value is zero?',0)
        p = predict(sample,ReplayModel([{'operations':[{'op':'bad'}]},
            {'operations':[{'op':'answer','assert':{'kind':'row_count','cmp':'gt','value':0}}]}]),settings('A4'))
        report,events,_ = analyze_mechanisms([sample],[p]); row=report['rows'][0]
        self.assertEqual(row['repair_attempted'],1); self.assertEqual(row['execution_recovery_given_repair'],1)
        self.assertEqual(row['counts'].get('correct_after_repair',0),0)
        self.assertEqual(events[0]['task_score'],0)

    def test_context_before_generation_is_separate_from_after_generation(self):
        sample = example('tabfact','train','x','t',['v'],[[1]],'value?',1)
        p = predict(sample,ReplayModel(['bad']),settings('A4',max_repairs=0))
        p['failure']['message']='Input exceeds configured context; no silent truncation'; p['responses']=[]
        report,_,_=analyze_mechanisms([sample],[p])
        self.assertEqual(report['rows'][0]['counts']['context_before_any_generation'],1)
        p['responses']=[{'text':'bad'}]
        report,_,_=analyze_mechanisms([sample],[p])
        self.assertEqual(report['rows'][0]['counts']['context_after_generation'],1)

    def test_official_correct_weak_filter_rejection_is_a_funnel_not_training_acceptance(self):
        sample=example('wikitq','train','x','t',['v'],[['1.0']],'value?',['1'])
        p=predict(sample,ReplayModel([{'operations':[{'op':'answer','columns':['c0']}]}]),settings('A4'))
        official={'scores':[{'dataset':'wikitq','id':'x','score':1}],'source_sha256':'s','converted_sha256':'p',
                  'samples_structured_sha256':record_hash([sample]),'predictions_structured_sha256':record_hash([p])}
        report,_,_=analyze_mechanisms([sample],[p],official)
        self.assertEqual(report['rows'][0]['counts']['official_correct_but_weak_filter_rejected'],1)
        self.assertEqual(report['rows'][0]['training_quality_decisions'],{'rejected':1})

    def test_cases_are_deterministic_and_never_approved(self):
        samples=[example('tabfact','train',str(i),'t',['v'],[[1]],'value?',1) for i in range(8)]
        ps=[predict(s,ReplayModel(['bad']),settings('A4',max_repairs=0)) for s in samples]
        a=analyze_mechanisms(samples,ps); b=analyze_mechanisms(list(reversed(samples)),list(reversed(ps)))
        self.assertEqual(a,b); self.assertEqual(len(a[2]),3)
        self.assertTrue(all(e['diagnostic_only'] for e in a[2]))

    def test_missing_predictions_are_not_treated_as_execution_failures(self):
        sample=example('tabfact','train','x','t',['v'],[[1]],'value?',1)
        with self.assertRaises(ValueError):analyze_mechanisms([sample],[])

    def test_distinguishes_terminal_and_expression_contracts(self):
        self.assertEqual(error_kind('Answer must terminate a block'),'terminal_answer_contract')
        self.assertEqual(error_kind('Expression must be a binary whitelist arithmetic tree'),'expression_schema')
