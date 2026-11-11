from decimal import Decimal
import unittest

from tableqa.core import TableState, run_plan, ExecutionError, evaluate_expression
from tableqa.data import example
from tableqa.distillation import build_records
from tableqa.experiments import settings
from tableqa.inference import SYSTEM, predict, system_prompt
from tableqa.models import ReplayModel
from tableqa.quantities import parse_quantity, quantity_text, QuantityError


def conversion_plan():
    return [
        {'op': 'derive', 'as': 'km', 'expression':
         {'op': 'quantity', 'args': [{'col': 'c1'}], 'unit': 'km'}},
        {'op': 'select', 'mode': 'extreme', 'column': 'km', 'which': 'max'},
        {'op': 'answer', 'columns': ['c0']},
    ]


def sample():
    return example('wikitq', 'train', 'quantity-fixture', 'quantity-fixture',
                   ['name', 'distance'], [['A', '500 m'], ['B', '2 km']],
                   'Which has the greatest distance?', ['B'])


class QuantityTests(unittest.TestCase):
    def test_explicit_scale_units_and_percent(self):
        self.assertEqual(parse_quantity('1,200 km').value, Decimal('1200000'))
        self.assertEqual(quantity_text('500 m', 'km'), '0.5')
        self.assertEqual(quantity_text('1.2 million USD', 'USD'), '1200000.0')
        self.assertEqual(quantity_text('12.5%', 'ratio'), '0.125')
        self.assertEqual(quantity_text('0.125', 'percent'), '12.5')
        self.assertEqual(quantity_text('90 s', 'min'), '1.5')

    def test_frequency_aliases_keep_dimensions_and_case(self):
        self.assertEqual(Decimal(quantity_text('3200 MHz', 'GHz')), Decimal('3.2'))
        self.assertEqual(quantity_text('2.67 ghz', 'MHz'), '2670.00')
        for value in ['1 mHz', '2014/15', 'Ret', '1 × 4.8 gt / s qpi i / o']:
            with self.subTest(value=value):
                with self.assertRaises(QuantityError):
                    parse_quantity(value)
        with self.assertRaises(QuantityError):
            quantity_text('2 GHz', 's')

    def test_units_are_never_guessed_or_crossed(self):
        for value, target in [('5', 'km'), ('2 kg', 'km'), ('1 USD', 'EUR')]:
            with self.subTest(value=value, target=target):
                with self.assertRaises(QuantityError): quantity_text(value, target)
        self.assertEqual(quantity_text('5', 'km', source_unit='m'), '0.005')
        parsed = parse_quantity('2 km', source_unit='m')
        self.assertFalse(parsed.assumed_unit_used)
        self.assertTrue(parse_quantity('2', source_unit='m').assumed_unit_used)

    def test_full_cell_and_numeric_bounds_reject_ambiguous_inputs(self):
        values = ['1999-2004', '3/4', '1,20 m', '1.2.3 km', '1 km [2]',
                  '$2 million', '12 million people', 'N/A', '-', 'NaN',
                  '1e999999', '1e-101', '5 M', '2 G', None, True, float('inf'), {'x': 1}]
        for value in values:
            with self.subTest(value=value):
                with self.assertRaises(QuantityError): parse_quantity(value)
        with self.assertRaises(QuantityError): quantity_text('1 s', 'min')

    def test_conversion_keeps_decimal_precision(self):
        text = '123456789012345678901234567890.125'
        self.assertEqual(quantity_text(text + ' m', 'm'), text)
        self.assertEqual(quantity_text('-2e2 cm', 'm'), '-2')

    def test_original_cells_and_lineage_survive_conversion(self):
        state = TableState.from_table(sample()['table']); before = state.signature()
        final, _ = run_plan(state, conversion_plan())
        self.assertEqual(final.answer, ['B'])
        self.assertEqual(state.signature(), before)
        self.assertEqual(final.rows[0]['c1'], '2 km')
        self.assertEqual(final.lineage['r1']['km'], {'r1:c1'})
        self.assertTrue({'r0:c1', 'r1:c1', 'r1:c0'} <= final.evidence)

    def test_conflicting_units_fail_atomically_and_are_not_dropped(self):
        table = {'headers': ['name', 'distance'], 'rows': [['A', '500 m'], ['B', '2 kg']]}
        state = TableState.from_table(table); before = state.signature()
        with self.assertRaises(ExecutionError): run_plan(state, conversion_plan())
        self.assertEqual(state.signature(), before)
        for expr in [{'op': 'quantity', 'args': [1]},
                     {'op': 'quantity', 'args': [1, 2], 'unit': 'km'}]:
            with self.assertRaises(ExecutionError): evaluate_expression(expr, {})

    def test_experimental_prompt_is_opt_in_and_gold_stays_private(self):
        self.assertEqual(system_prompt('B5'), SYSTEM)
        self.assertNotIn('quantity-v2', system_prompt('B0', True))
        item = {**sample(), 'reference': ['PRIVATE_GOLD_SENTINEL']}
        class Spy(ReplayModel):
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

    def test_distillation_requires_matching_experimental_prompt(self):
        with self.assertRaises(ValueError): build_records(sample(), conversion_plan())
        records = build_records(sample(), conversion_plan(), typed_quantities=True)
        self.assertTrue(all('quantity-v2' in r['messages'][0]['content'] for r in records))
        self.assertNotIn('PRIVATE_GOLD_SENTINEL', str(records))
        # Answer-only controls can reuse the same accepted source pool without
        # acquiring an operation tool or requiring an irrelevant tool flag.
        direct = build_records(sample(), conversion_plan(), 'B0')
        self.assertNotIn('quantity-v2', str(direct[0]['messages']))

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
