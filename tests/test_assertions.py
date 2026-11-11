import unittest
from tableqa.core import TableState,run_plan,ExecutionError
from tableqa.inference import predict
from tableqa.models import ReplayModel
from tableqa.experiments import settings
from tableqa.data import example

class AssertionTests(unittest.TestCase):
    def test_empty_filter_cannot_make_claim_true(self):
        state=TableState.from_table({'headers':['n'],'rows':[[5],[9]]})
        ops=[{'op':'filter','column':'c0','cmp':'gt','value':10,'type':'number'},
             {'op':'answer','assert':{'kind':'row_count','cmp':'gt','value':0}}]
        result,_=run_plan(state,ops)
        self.assertEqual(result.answer,0)
        self.assertEqual(result.evidence,{'r0:c0','r1:c0'})
    def test_count_assertion_is_computed(self):
        state=TableState.from_table({'headers':['n'],'rows':[[5],[9]]})
        result,_=run_plan(state,[{'op':'group_aggregate','by':[],'aggregations':[{'fn':'count','as':'n'}]},
             {'op':'answer','assert':{'kind':'cell','column':'n','cmp':'eq','value':1,'type':'number'}}])
        self.assertEqual(result.answer,0)
    def test_last_is_position_not_numeric_max(self):
        state=TableState.from_table({'headers':['state','n'],'rows':[['A',9],['B',1]]})
        result,_=run_plan(state,[{'op':'select','mode':'last','n':1},{'op':'answer','columns':['c0']}])
        self.assertEqual(result.answer,['B'])
    def test_literal_verification_is_rejected(self):
        sample=example('tabfact','train','x','t',['n'],[[1]],'n is 1',1)
        result=predict(sample,ReplayModel([{'operations':[{'op':'answer','value':1}]}]),settings('A4',max_repairs=0))
        self.assertEqual(result['status'],'failed')
    def test_assertion_column_is_a_dependency(self):
        state=TableState.from_table({'headers':['n'],'rows':[[1]]})
        with self.assertRaises(ExecutionError):
            run_plan(state,[{'op':'answer','assert':{'kind':'cell','column':'missing','cmp':'eq','value':1}}])
