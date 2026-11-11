# 测试范围：严格解析完整单元格中的数值和单位，处理可追溯的数量转换及工具提示。
# 运行方式：python -m unittest discover -s tests -p test_quantities.py -v
# 阅读顺序：先看样本或替身模型的构造，再看每个用例前的预期约束。
from decimal import Decimal
import unittest

from tableqa.core import TableState, run_plan, ExecutionError, evaluate_expression
from tableqa.data import example
from tableqa.distillation import build_records
from tableqa.experiments import settings
from tableqa.inference import SYSTEM, predict, system_prompt
from tableqa.models import ReplayModel
from tableqa.quantities import parse_quantity, quantity_text, QuantityError


# 构造数量转换与读出链，验证单位规则及来源保留。
# 返回形式含：列表记录。
def conversion_plan():
    return [
        {'op': 'derive', 'as': 'km', 'expression':
         {'op': 'quantity', 'args': [{'col': 'c1'}], 'unit': 'km'}},
        {'op': 'select', 'mode': 'extreme', 'column': 'km', 'which': 'max'},
        {'op': 'answer', 'columns': ['c0']},
    ]


# 构造当前测试使用的独立规范样本，固定题面、原表和参考结果。
def sample():
    return example('wikitq', 'train', 'quantity-fixture', 'quantity-fixture',
                   ['name', 'distance'], [['A', '500 m'], ['B', '2 km']],
                   'Which has the greatest distance?', ['B'])


# QuantityTests：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
class QuantityTests(unittest.TestCase):
    # 回归用例：test_explicit_scale_units_and_percent
    # 应相等：parse_quantity('1,200 km').value，Decimal('1200000')
    # 应相等：quantity_text('500 m', 'km')，'0.5'
    # 应相等：quantity_text('1.2 million USD', 'USD')，'1200000.0'
    def test_explicit_scale_units_and_percent(self):
        self.assertEqual(parse_quantity('1,200 km').value, Decimal('1200000'))
        self.assertEqual(quantity_text('500 m', 'km'), '0.5')
        self.assertEqual(quantity_text('1.2 million USD', 'USD'), '1200000.0')
        self.assertEqual(quantity_text('12.5%', 'ratio'), '0.125')
        self.assertEqual(quantity_text('0.125', 'percent'), '12.5')
        self.assertEqual(quantity_text('90 s', 'min'), '1.5')

    # 回归用例：test_frequency_aliases_keep_dimensions_and_case
    # 应相等：Decimal(quantity_text('3200 MHz', 'GHz'))，Decimal('3.2')
    # 应相等：quantity_text('2.67 ghz', 'MHz')，'2670.00'
    # 应抛出预期异常：QuantityError
    def test_frequency_aliases_keep_dimensions_and_case(self):
        self.assertEqual(Decimal(quantity_text('3200 MHz', 'GHz')), Decimal('3.2'))
        self.assertEqual(quantity_text('2.67 ghz', 'MHz'), '2670.00')
        for value in ['1 mHz', '2014/15', 'Ret', '1 × 4.8 gt / s qpi i / o']:
            with self.subTest(value=value):
                with self.assertRaises(QuantityError):
                    parse_quantity(value)
        with self.assertRaises(QuantityError):
            quantity_text('2 GHz', 's')

    # 回归用例：test_units_are_never_guessed_or_crossed
    # 应相等：quantity_text('5', 'km', source_unit='m')，'0.005'
    # 应不成立：parsed.assumed_unit_used
    # 应成立：parse_quantity('2', source_unit='m').assumed_unit_used
    def test_units_are_never_guessed_or_crossed(self):
        for value, target in [('5', 'km'), ('2 kg', 'km'), ('1 USD', 'EUR')]:
            with self.subTest(value=value, target=target):
                with self.assertRaises(QuantityError): quantity_text(value, target)
        self.assertEqual(quantity_text('5', 'km', source_unit='m'), '0.005')
        parsed = parse_quantity('2 km', source_unit='m')
        self.assertFalse(parsed.assumed_unit_used)
        self.assertTrue(parse_quantity('2', source_unit='m').assumed_unit_used)

    # 回归用例：test_full_cell_and_numeric_bounds_reject_ambiguous_inputs
    # 应抛出预期异常：QuantityError
    def test_full_cell_and_numeric_bounds_reject_ambiguous_inputs(self):
        values = ['1999-2004', '3/4', '1,20 m', '1.2.3 km', '1 km [2]',
                  '$2 million', '12 million people', 'N/A', '-', 'NaN',
                  '1e999999', '1e-101', '5 M', '2 G', None, True, float('inf'), {'x': 1}]
        for value in values:
            with self.subTest(value=value):
                with self.assertRaises(QuantityError): parse_quantity(value)
        with self.assertRaises(QuantityError): quantity_text('1 s', 'min')

    # 回归用例：test_conversion_keeps_decimal_precision
    # 应相等：quantity_text(text + ' m', 'm')，text
    # 应相等：quantity_text('-2e2 cm', 'm')，'-2'
    def test_conversion_keeps_decimal_precision(self):
        text = '123456789012345678901234567890.125'
        self.assertEqual(quantity_text(text + ' m', 'm'), text)
        self.assertEqual(quantity_text('-2e2 cm', 'm'), '-2')

    # 回归用例：test_original_cells_and_lineage_survive_conversion
    # 应相等：final.answer，['B']
    # 应相等：state.signature()，before
    # 应相等：final.rows[0]['c1']，'2 km'
    def test_original_cells_and_lineage_survive_conversion(self):
        state = TableState.from_table(sample()['table']); before = state.signature()
        final, _ = run_plan(state, conversion_plan())
        self.assertEqual(final.answer, ['B'])
        self.assertEqual(state.signature(), before)
        self.assertEqual(final.rows[0]['c1'], '2 km')
        self.assertEqual(final.lineage['r1']['km'], {'r1:c1'})
        self.assertTrue({'r0:c1', 'r1:c1', 'r1:c0'} <= final.evidence)

    # 回归用例：test_conflicting_units_fail_atomically_and_are_not_dropped
    # 应相等：state.signature()，before
    # 应抛出预期异常：ExecutionError
    def test_conflicting_units_fail_atomically_and_are_not_dropped(self):
        table = {'headers': ['name', 'distance'], 'rows': [['A', '500 m'], ['B', '2 kg']]}
        state = TableState.from_table(table); before = state.signature()
        with self.assertRaises(ExecutionError): run_plan(state, conversion_plan())
        self.assertEqual(state.signature(), before)
        for expr in [{'op': 'quantity', 'args': [1]},
                     {'op': 'quantity', 'args': [1, 2], 'unit': 'km'}]:
            with self.assertRaises(ExecutionError): evaluate_expression(expr, {})

    # 回归用例：test_experimental_prompt_is_opt_in_and_gold_stays_private
    # 应相等：system_prompt('B5')，SYSTEM
    # 应排除：'quantity-v2'，system_prompt('B0', True)
    # 应相等：result['answer']，['B']
    def test_experimental_prompt_is_opt_in_and_gold_stays_private(self):
        self.assertEqual(system_prompt('B5'), SYSTEM)
        self.assertNotIn('quantity-v2', system_prompt('B0', True))
        item = {**sample(), 'reference': ['PRIVATE_GOLD_SENTINEL']}
        # Spy：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
        class Spy(ReplayModel):
            # 按当前适配接口生成或重放响应，同时保留调用所需的观测状态。
            # 输入：messages（模型可见消息）。
            # 相关调用：super().generate。
            def generate(self, messages):
                assert 'PRIVATE_GOLD_SENTINEL' not in str(messages)
                assert 'quantity-v2' in messages[0]['content']
                return super().generate(messages)
        result = predict(item, Spy([{'operations': conversion_plan()}]),
                         settings('B5', typed_quantities=True, max_repairs=0))
        self.assertEqual(result['answer'], ['B'])
        disabled = predict(item, ReplayModel([{'operations': conversion_plan()}]),
                           settings('B5', max_repairs=0))
        self.assertEqual(disabled['status'], 'failed')

    # 回归用例：test_distillation_requires_matching_experimental_prompt
    # 应成立：all(('quantity-v2' in r['messages'][0]['content'] for r in records))
    # 应排除：'PRIVATE_GOLD_SENTINEL'，str(records)
    # 应排除：'quantity-v2'，str(direct[0]['messages'])
    def test_distillation_requires_matching_experimental_prompt(self):
        with self.assertRaises(ValueError): build_records(sample(), conversion_plan())
        records = build_records(sample(), conversion_plan(), typed_quantities=True)
        self.assertTrue(all('quantity-v2' in r['messages'][0]['content'] for r in records))
        self.assertNotIn('PRIVATE_GOLD_SENTINEL', str(records))
        # Answer-only controls can reuse the same accepted source pool without
        # acquiring an operation tool or requiring an irrelevant tool flag.
        direct = build_records(sample(), conversion_plan(), 'B0')
        self.assertNotIn('quantity-v2', str(direct[0]['messages']))

    # 回归用例：test_pilot_selection_is_train_only_label_independent_and_per_table
    # 应相等：[r['id'] for r in first]，[r['id'] for r in select_samples(changed, 1)]
    # 应相等：len(first)，2
    # 应抛出预期异常：ValueError
    def test_pilot_selection_is_train_only_label_independent_and_per_table(self):
        from scripts.quantity_pilot import select_samples
        from copy import deepcopy
        rows = [{**sample(), 'dataset': d, 'id': f'{d}-{i}'}
                for d in ['wikitq', 'tabfact'] for i in range(2)]
        rows[-1]['table'] = {'headers': ['x'], 'rows': [['3 kg']]}
        first = select_samples(rows, 1)
        changed = deepcopy(rows)
        for r in changed:
            r['reference'] = ['ARBITRARY']; r['question'] = 'Changed without changing selection'
        self.assertEqual([r['id'] for r in first], [r['id'] for r in select_samples(changed, 1)])
        self.assertEqual(len(first), 2)
        changed[0]['split'] = 'test'
        with self.assertRaises(ValueError): select_samples(changed, 1)


if __name__ == '__main__':
    unittest.main()
