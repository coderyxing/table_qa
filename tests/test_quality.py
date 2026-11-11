# 测试范围：重放已生成的操作链，审核答案、证据和监督入池条件，记录逐条拒绝原因。
# 运行方式：python -m unittest discover -s tests -p test_quality.py -v
# 阅读顺序：先看样本或替身模型的构造，再看每个用例前的预期约束。
import unittest
from tableqa.data import example
from tableqa.inference import predict
from tableqa.models import ReplayModel
from tableqa.experiments import settings
from tableqa.quality import audit_trajectory

# TrajectoryQualityTests：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
class TrajectoryQualityTests(unittest.TestCase):
    # 回归用例：test_correct_replayed_short_is_only_weak_acceptance
    # 应相等：audit_trajectory(r, p)['decision']，'accepted_weak'
    # 应包含：'replay_answer_mismatch'，audit_trajectory(r, p)['reasons']
    def test_correct_replayed_short_is_only_weak_acceptance(self):
        r=example('wikitq','train','x','t',['v'],[['A']],'value?',['A'])
        p=predict(r,ReplayModel([{'operations':[{'op':'answer','columns':['c0']}]}]),settings('A4'))
        self.assertEqual(audit_trajectory(r,p)['decision'],'accepted_weak')
        p['answer']=['tampered']
        self.assertIn('replay_answer_mismatch',audit_trajectory(r,p)['reasons'])
    # 回归用例：test_freeform_cannot_auto_approve
    # 应相等：audit_trajectory(r, p)['decision']，'review_required'
    def test_freeform_cannot_auto_approve(self):
        r=example('fetaqa','train','x','t',['v'],[['A']],'value?','A')
        p=predict(r,ReplayModel([{'operations':[{'op':'answer','columns':['c0'],'value':'A'}]}]),settings('A4'))
        self.assertEqual(audit_trajectory(r,p)['decision'],'review_required')
    # 回归用例：test_correct_literal_short_is_not_a_grounded_chain
    # 应包含：'short_answer_not_executed'，audit_trajectory(r, p)['reasons']
    def test_correct_literal_short_is_not_a_grounded_chain(self):
        r=example('wikitq','train','x','t',['v'],[['A']],'value?',['A'])
        p=predict(r,ReplayModel([{'operations':[{'op':'answer','value':['A']}]}]),settings('A4'))
        self.assertIn('short_answer_not_executed',audit_trajectory(r,p)['reasons'])
