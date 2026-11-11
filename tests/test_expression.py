# 测试范围：验证对应模块的数据隔离、输入约束与历史回归行为。
# 运行方式：python -m unittest discover -s tests -p test_expression.py -v
# 阅读顺序：先看样本或替身模型的构造，再看每个用例前的预期约束。
import unittest
from tableqa.core import TableState,run_plan,evaluate_expression,ExecutionError
from tableqa.models import first_object_end

# ExpressionTests：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
class ExpressionTests(unittest.TestCase):
    # 回归用例：test_numeric_membership_and_header_removal
    # 应相等：result.answer，[1, 3]
    def test_numeric_membership_and_header_removal(self):
        state=TableState.from_table({'headers':['n'],'rows':[['n'],[1],[2],[3]]})
        result,_=run_plan(state,[{'op':'filter','column':'c0','cmp':'is_number'},
             {'op':'filter','column':'c0','cmp':'in','value':[1,3],'type':'number'},
             {'op':'answer','columns':['c0']}])
        self.assertEqual(result.answer,[1,3])
    # 回归用例：test_if_comparison_is_safe_and_lazy
    # 应相等：evaluate_expression(expression, {'n': 2})，'yes'
    # 应抛出预期异常：ExecutionError
    def test_if_comparison_is_safe_and_lazy(self):
        expression={'op':'if','args':[{'op':'gt','args':[{'col':'n'},1]},'yes',{'op':'div','args':[1,0]}]}
        self.assertEqual(evaluate_expression(expression,{'n':2}),'yes')
        with self.assertRaises(ExecutionError):evaluate_expression({'op':'exec','args':['x',1]}, {})
    # 回归用例：test_two_column_assertion_tracks_both_sources
    # 应相等：result.answer，1
    # 应相等：result.evidence，{'r0:c0', 'r0:c1'}
    def test_two_column_assertion_tracks_both_sources(self):
        state=TableState.from_table({'headers':['a','b'],'rows':[[3,2]]})
        result,_=run_plan(state,[{'op':'answer','assert':{'kind':'expression',
              'expression':{'op':'gt','args':[{'col':'c0'},{'col':'c1'}]}}}])
        self.assertEqual(result.answer,1)
        self.assertEqual(result.evidence,{'r0:c0','r0:c1'})
    # 回归用例：test_json_token_overhang_is_separate_from_object
    # 应相等：raw[:end]，'  {"answer":"}"}'
    # 应相等：raw[end:]，'"'
    def test_json_token_overhang_is_separate_from_object(self):
        raw='  {"answer":"}"}"'
        end=first_object_end(raw)
        self.assertEqual(raw[:end],'  {"answer":"}"}')
        self.assertEqual(raw[end:],'"')
