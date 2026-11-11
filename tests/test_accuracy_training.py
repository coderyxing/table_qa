# 测试范围：验证对应模块的数据隔离、输入约束与历史回归行为。
# 运行方式：python -m unittest discover -s tests -p test_accuracy_training.py -v
# 阅读顺序：先看样本或替身模型的构造，再看每个用例前的预期约束。
import unittest
from collections import defaultdict
from tableqa.training import attach_loss_weights
from tableqa.data import example
from tableqa.experiments import settings
from tableqa.inference import predict
from tableqa.models import ReplayModel
from tableqa.quality import audit_run_trajectories,audit_trajectory
from tableqa.analysis import record_hash


# AccuracyTrainingTests：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
class AccuracyTrainingTests(unittest.TestCase):
    # 回归用例：test_task_and_label_weights_balance_mass_without_losing_records
    # 应相等：len(actual)，len(rows)
    # 应在浮点容差内相等：sum(mass.values())，len(rows)
    # 应在浮点容差内相等：mass['wikitq']，mass['tabfact']
    def test_task_and_label_weights_balance_mass_without_losing_records(self):
        rows=[{'source_key':('wikitq',str(i)),'kind_id':k} for k in [0,1] for i in range(2)]
        rows += [{'source_key':('tabfact',str(i)),'kind_id':k,'source_label':int(i<6)} for k in [0,1] for i in range(8)]
        actual,report=attach_loss_weights(rows,'task_label');mass=defaultdict(float)
        for r in actual:mass[r['source_key'][0]]+=r['sample_weight']
        self.assertEqual(len(actual),len(rows));self.assertAlmostEqual(sum(mass.values()),len(rows))
        self.assertAlmostEqual(mass['wikitq'],mass['tabfact'])
        tf=[r for r in report if r['dataset']=='tabfact']
        self.assertEqual(len({round(r['weighted_mass'],8) for r in tf}),1)
        self.assertTrue(all('sample_weight' not in r for r in rows))
    # 回归用例：test_default_weights_preserve_legacy_and_bad_labels_fail
    # 应相等：attach_loss_weights(rows)，(rows, [])
    # 应抛出预期异常：ValueError
    def test_default_weights_preserve_legacy_and_bad_labels_fail(self):
        rows=[{'source_key':('tabfact','x'),'kind_id':0}]
        self.assertEqual(attach_loss_weights(rows),(rows,[]))
        with self.assertRaises(ValueError):attach_loss_weights(rows,'task_label')
        with self.assertRaises(ValueError):attach_loss_weights(rows,'invalid')
    # 回归用例：test_official_acceptance_is_bound_and_still_not_semantic_proof
    # 应相等：audit_trajectory(s, p)['decision']，'rejected'
    # 应相等：result['decision']，'accepted_official'
    # 应不成立：result['semantic_correctness_proven']
    def test_official_acceptance_is_bound_and_still_not_semantic_proof(self):
        s=example('wikitq','train','x','t',['v'],[['1.0']],'value?',['1'])
        p=predict(s,ReplayModel([{'operations':[{'op':'answer','columns':['c0']}]}]),settings('A4'))
        report={'source_sha256':'s','converted_sha256':'c','scores':[{'dataset':'wikitq','id':'x','score':1}],
                'samples_structured_sha256':record_hash([s]),'predictions_structured_sha256':record_hash([p])}
        self.assertEqual(audit_trajectory(s,p)['decision'],'rejected')
        result=audit_run_trajectories([s],[p],report)[0]
        self.assertEqual(result['decision'],'accepted_official');self.assertFalse(result['semantic_correctness_proven'])
        with self.assertRaises(ValueError):audit_run_trajectories([s],[{**p,'answer':['changed']}],report)
    # 回归用例：test_official_score_cannot_bypass_execution_checks
    # 应相等：audit_trajectory(s, p, 1)['decision']，'rejected'
    # 应抛出预期异常：ValueError
    def test_official_score_cannot_bypass_execution_checks(self):
        s=example('wikitq','train','x','t',['v'],[['A']],'value?',['A'])
        p=predict(s,ReplayModel([{'operations':[{'op':'answer','value':['A']}]}]),settings('A4'))
        self.assertEqual(audit_trajectory(s,p,1)['decision'],'rejected')
        with self.assertRaises(ValueError):audit_trajectory(s,p,True)
