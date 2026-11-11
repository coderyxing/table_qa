import unittest
from tableqa.core import TableState,run_plan,evaluate_expression,ExecutionError
from tableqa.models import first_object_end

class ExpressionTests(unittest.TestCase):
    def test_numeric_membership_and_header_removal(self):
        state=TableState.from_table({'headers':['n'],'rows':[['n'],[1],[2],[3]]})
        result,_=run_plan(state,[{'op':'filter','column':'c0','cmp':'is_number'},
             {'op':'filter','column':'c0','cmp':'in','value':[1,3],'type':'number'},
             {'op':'answer','columns':['c0']}])
        self.assertEqual(result.answer,[1,3])
    def test_if_comparison_is_safe_and_lazy(self):
        expression={'op':'if','args':[{'op':'gt','args':[{'col':'n'},1]},'yes',{'op':'div','args':[1,0]}]}
        self.assertEqual(evaluate_expression(expression,{'n':2}),'yes')
        with self.assertRaises(ExecutionError):evaluate_expression({'op':'exec','args':['x',1]}, {})
    def test_two_column_assertion_tracks_both_sources(self):
        state=TableState.from_table({'headers':['a','b'],'rows':[[3,2]]})
        result,_=run_plan(state,[{'op':'answer','assert':{'kind':'expression',
              'expression':{'op':'gt','args':[{'col':'c0'},{'col':'c1'}]}}}])
        self.assertEqual(result.answer,1)
        self.assertEqual(result.evidence,{'r0:c0','r0:c1'})
    def test_json_token_overhang_is_separate_from_object(self):
        raw='  {"answer":"}"}"'
        end=first_object_end(raw)
        self.assertEqual(raw[:end],'  {"answer":"}"}')
        self.assertEqual(raw[end:],'"')
