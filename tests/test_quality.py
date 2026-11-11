import unittest
from tableqa.data import example
from tableqa.inference import predict
from tableqa.models import ReplayModel
from tableqa.experiments import settings
from tableqa.quality import audit_trajectory

class TrajectoryQualityTests(unittest.TestCase):
    def test_correct_replayed_short_is_only_weak_acceptance(self):
        r=example('wikitq','train','x','t',['v'],[['A']],'value?',['A'])
        p=predict(r,ReplayModel([{'operations':[{'op':'answer','columns':['c0']}]}]),settings('A4'))
        self.assertEqual(audit_trajectory(r,p)['decision'],'accepted_weak')
        p['answer']=['tampered']
        self.assertIn('replay_answer_mismatch',audit_trajectory(r,p)['reasons'])
    def test_freeform_cannot_auto_approve(self):
        r=example('fetaqa','train','x','t',['v'],[['A']],'value?','A')
        p=predict(r,ReplayModel([{'operations':[{'op':'answer','columns':['c0'],'value':'A'}]}]),settings('A4'))
        self.assertEqual(audit_trajectory(r,p)['decision'],'review_required')
    def test_correct_literal_short_is_not_a_grounded_chain(self):
        r=example('wikitq','train','x','t',['v'],[['A']],'value?',['A'])
        p=predict(r,ReplayModel([{'operations':[{'op':'answer','value':['A']}]}]),settings('A4'))
        self.assertIn('short_answer_not_executed',audit_trajectory(r,p)['reasons'])
