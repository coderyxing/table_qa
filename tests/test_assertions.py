# 测试范围：验证对应模块的数据隔离、输入约束与历史回归行为。
# 运行方式：python -m unittest discover -s tests -p test_assertions.py -v
# 阅读顺序：先看样本或替身模型的构造，再看每个用例前的预期约束。
import unittest
from tableqa.core import TableState,run_plan,ExecutionError
from tableqa.inference import predict
from tableqa.models import ReplayModel
from tableqa.experiments import settings
from tableqa.data import example

# AssertionTests：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
class AssertionTests(unittest.TestCase):
    # 回归用例：test_empty_filter_cannot_make_claim_true
    # 应相等：result.answer，0
    # 应相等：result.evidence，{'r0:c0', 'r1:c0'}
    def test_empty_filter_cannot_make_claim_true(self):
        state=TableState.from_table({'headers':['n'],'rows':[[5],[9]]})
        ops=[{'op':'filter','column':'c0','cmp':'gt','value':10,'type':'number'},
             {'op':'answer','assert':{'kind':'row_count','cmp':'gt','value':0}}]
        result,_=run_plan(state,ops)
        self.assertEqual(result.answer,0)
        self.assertEqual(result.evidence,{'r0:c0','r1:c0'})
    # 回归用例：test_count_assertion_is_computed
    # 应相等：result.answer，0
    def test_count_assertion_is_computed(self):
        state=TableState.from_table({'headers':['n'],'rows':[[5],[9]]})
        result,_=run_plan(state,[{'op':'group_aggregate','by':[],'aggregations':[{'fn':'count','as':'n'}]},
             {'op':'answer','assert':{'kind':'cell','column':'n','cmp':'eq','value':1,'type':'number'}}])
        self.assertEqual(result.answer,0)
    # 回归用例：test_last_is_position_not_numeric_max
    # 应相等：result.answer，['B']
    def test_last_is_position_not_numeric_max(self):
        state=TableState.from_table({'headers':['state','n'],'rows':[['A',9],['B',1]]})
        result,_=run_plan(state,[{'op':'select','mode':'last','n':1},{'op':'answer','columns':['c0']}])
        self.assertEqual(result.answer,['B'])
    # 回归用例：test_literal_verification_is_rejected
    # 应相等：result['status']，'failed'
    def test_literal_verification_is_rejected(self):
        sample=example('tabfact','train','x','t',['n'],[[1]],'n is 1',1)
        result=predict(sample,ReplayModel([{'operations':[{'op':'answer','value':1}]}]),settings('A4',max_repairs=0))
        self.assertEqual(result['status'],'failed')
    # 回归用例：test_assertion_column_is_a_dependency
    # 应抛出预期异常：ExecutionError
    def test_assertion_column_is_a_dependency(self):
        state=TableState.from_table({'headers':['n'],'rows':[[1]]})
        with self.assertRaises(ExecutionError):
            run_plan(state,[{'op':'answer','assert':{'kind':'cell','column':'missing','cmp':'eq','value':1}}])
