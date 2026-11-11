import json
import unittest
from copy import deepcopy

from tableqa.core import TableState, ExecutionError, run_plan, evaluate_expression
from tableqa.data import example
from tableqa.distillation import build_records
from tableqa.experiments import settings
from tableqa.inference import predict, system_prompt, SYSTEM
from tableqa.models import ReplayModel
from tableqa.optimization import state_cost_partition
from tableqa.semantic_columns import extract_span, ExtractionError


def sample():
    return example('wikitq', 'train', 'extract-fixture', 'extract-fixture',
                   ['Athlete', 'Rank'], [['Rina (RED)', '12th'], ['Kaito (BLU)', '3rd'],
                                         ['Noor (RED)', 'T3']],
                   'Which affiliation code has most athletes?', ['RED'])


def plan():
    return [{'op': 'derive', 'as': 'affiliation', 'expression':
             {'op': 'extract', 'args': [{'col': 'c0'}], 'rule': 'parenthesized'}},
            {'op': 'group_aggregate', 'by': ['affiliation'], 'aggregations': [{'fn': 'count', 'as': 'n'}]},
            {'op': 'select', 'mode': 'extreme', 'column': 'n', 'which': 'max'},
            {'op': 'answer', 'columns': ['affiliation']}]


class ExtractionTests(unittest.TestCase):
    def test_offsets_copy_unicode_and_trim_without_inventing_text(self):
        raw = '林 ( 红队 ) (reserve)'
        first = extract_span(raw, 'parenthesized')
        self.assertEqual(first['value'], '红队')
        self.assertEqual(raw[first['start']:first['end']], first['value'])
        self.assertEqual(extract_span(raw, 'parenthesized', -1)['value'], 'reserve')
        self.assertEqual(first['semantic_correctness'], 'not_proven')

    def test_unbalanced_nested_missing_and_empty_fields_fail(self):
        for text in ['A (X', 'A X)', 'A ((X))', 'A', 'A ()', 'A ( )']:
            with self.subTest(text=text), self.assertRaises(ExtractionError):
                extract_span(text, 'parenthesized')
        for index in [True, 1.0, 999]:
            with self.assertRaises(ExtractionError): extract_span('A (B)', 'parenthesized', index)

    def test_literal_split_preserves_exact_span_and_requires_delimiter(self):
        raw = 'City,  Region, Country '
        result = extract_span(raw, 'split', -1, ',')
        self.assertEqual(result['value'], 'Country')
        self.assertEqual(raw[result['start']:result['end']], 'Country')
        for delimiter in ['', None, ';', 'x' * 33]:
            with self.assertRaises(ExtractionError): extract_span(raw, 'split', delimiter=delimiter)
        with self.assertRaises(ExtractionError): extract_span('a,,b', 'split', 1, ',')

    def test_complete_rank_parsing_preserves_ties_and_validates_suffix(self):
        for value, expected in [('T3', '3'), ('14th (q)', '14'), (' 21st ', '21'), ('=3', '3')]:
            self.assertEqual(extract_span(value, 'ordinal')['value'], expected)
        for value in ['11st', '3th', '0th', '3-4', 'won 3 races', '3/4', '3rd and 4th']:
            with self.subTest(value=value), self.assertRaises(ExtractionError): extract_span(value, 'ordinal')
        ops = [{'op': 'derive', 'as': 'rank', 'expression':
                {'op': 'extract', 'args': [{'col': 'c1'}], 'rule': 'ordinal'}},
               {'op': 'select', 'mode': 'extreme', 'column': 'rank', 'which': 'min'},
               {'op': 'answer', 'columns': ['c0']}]
        final, _ = run_plan(TableState.from_table(sample()['table']), ops)
        self.assertEqual(final.answer, ['Kaito (BLU)', 'Noor (RED)'])

    def test_replay_retains_raw_cells_lineage_and_character_evidence(self):
        initial = TableState.from_table(sample()['table']); signature = initial.signature()
        result, trace = run_plan(initial, plan())
        self.assertEqual(result.answer, ['RED'])
        self.assertEqual(initial.signature(), signature)
        spans = trace[0]['extractions']
        self.assertEqual(len(spans), 3)
        self.assertEqual(spans[0]['source_cells'], ['r0:c0'])
        self.assertTrue(all(s['copy_verified'] for s in spans))
        self.assertTrue({'r0:c0', 'r1:c0', 'r2:c0'} <= result.evidence)
        self.assertEqual(trace[0]['after']['rows'][0]['c0'], 'Rina (RED)')

    def test_missing_structure_does_not_silently_discard_rows(self):
        data = sample(); data['table']['rows'][1][0] = 'Kaito'
        state = TableState.from_table(data['table']); original = state.signature()
        with self.assertRaises(ExecutionError): run_plan(state, plan())
        self.assertEqual(original, state.signature())

    def test_only_direct_cell_arguments_are_accepted(self):
        for args in [[], ['forged (X)'], [{'literal': 'forged (X)'}],
                     [{'op': 'add', 'args': [1, 2]}], [{'col': 'c0'}, {'col': 'c1'}]]:
            expr = {'op': 'extract', 'args': args, 'rule': 'parenthesized'}
            with self.assertRaises(ExecutionError): evaluate_expression(expr, {'c0': 'Rina (RED)'})
        for rule in ['regex', 'eval', '__import__']:
            with self.assertRaises(ExtractionError): extract_span('x', rule)

    def test_lazy_branch_audit_does_not_execute_unused_extraction(self):
        state = TableState.from_table({'headers': ['x'], 'rows': [['Rina (RED)'], ['No parentheses']]})
        expression = {'op': 'if', 'args': [
            {'op': 'eq', 'type': 'text', 'args': [{'col': 'c0'}, 'Rina (RED)']},
            {'op': 'extract', 'args': [{'col': 'c0'}], 'rule': 'parenthesized'}, 'unknown']}
        final, trace = run_plan(state, [{'op': 'derive', 'as': 'derived', 'expression': expression}])
        self.assertEqual([r['derived'] for r in final.rows], ['RED', 'unknown'])
        self.assertEqual(len(trace[0]['extractions']), 1)

    def test_spans_of_derived_cells_are_explicitly_scoped(self):
        state = TableState.from_table({'headers': ['x'], 'rows': [['City, Rina (RED)']]})
        ops = [{'op': 'derive', 'as': 'person', 'expression':
                {'op': 'extract', 'args': [{'col': 'c0'}], 'rule': 'split', 'delimiter': ',', 'index': 1}},
               {'op': 'derive', 'as': 'code', 'expression':
                {'op': 'extract', 'args': [{'col': 'person'}], 'rule': 'parenthesized'}}]
        _, trace = run_plan(state, ops)
        record = trace[1]['extractions'][0]
        self.assertEqual(record['span_scope'], 'current_source_cell')
        self.assertEqual(record['source_cells'], ['r0:c0'])
        self.assertEqual(record['start'], 6)

    def test_opt_in_prompts_and_private_reference_isolation(self):
        item = sample(); item['reference'] = ['PRIVATE_SENTINEL']
        class Spy(ReplayModel):
            def generate(self, messages):
                assert 'PRIVATE_SENTINEL' not in str(messages)
                assert 'extract-v1' in messages[0]['content']
                return super().generate(messages)
        result = predict(item, Spy([{'operations': plan()}]), settings('A4', semantic_extract=True))
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(system_prompt('B5'), SYSTEM)
        self.assertEqual(system_prompt('B0', True, True, True), system_prompt('B0'))
        disabled = predict(item, ReplayModel([{'operations': plan()}]), settings('A4', max_repairs=0))
        self.assertEqual(disabled['status'], 'failed')

    def test_training_flags_match_inference_and_answer_controls_reuse_pool(self):
        with self.assertRaises(ValueError): build_records(sample(), plan())
        rows = build_records(sample(), plan(), semantic_extract=True, tool_demonstrations=True)
        self.assertEqual(rows[0]['messages'][0]['content'], system_prompt('B5', False, True, True))
        self.assertIn('Synthetic tool demonstration', rows[0]['messages'][0]['content'])
        self.assertNotIn('extract-v1', str(build_records(sample(), plan(), 'B0')[0]['messages']))

    def test_partition_charges_enabled_tool_and_demonstration_tokens(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs): return list(json.dumps(messages))
            def encode(self, value, **kwargs): return list(value)
        with self.assertRaises(ValueError):
            state_cost_partition(sample(), plan(), Tokenizer(), max_length=100000, max_output_tokens=10000)
        _, bare = state_cost_partition(sample(), plan(), Tokenizer(), max_length=100000,
            max_output_tokens=10000, semantic_extract=True)
        _, enabled = state_cost_partition(sample(), plan(), Tokenizer(), max_length=100000,
            max_output_tokens=10000, semantic_extract=True, tool_demonstrations=True)
        self.assertGreater(enabled['cost'], bare['cost'])

    def test_pilot_selection_does_not_read_gold_and_rejects_test_data(self):
        from scripts.semantic_capability_pilot import select_samples
        rows = []
        for dataset in ['wikitq', 'tabfact']:
            item = sample(); item['dataset'] = dataset
            # Distinct input tables across tasks and categories.
            item['table']['rows'][0][0] = f'Person{dataset} (RED)'
            item['question'] = 'Which country has most athletes?'
            item['id'] = dataset + '-country'; rows.append(item)
            rank = deepcopy(item); rank['table']['rows'][0][0] += '-rank'
            rank['question'] = 'Who has the lowest rank?'
            rank['id'] = dataset + '-rank'; rows.append(rank)
        first, _ = select_samples(rows, 1)
        changed = deepcopy(rows)
        for item in changed: item['reference'] = ['ARBITRARY']
        second, _ = select_samples(changed, 1)
        self.assertEqual([r['id'] for r in first], [r['id'] for r in second])
        changed[0]['split'] = 'test'
        with self.assertRaises(ValueError): select_samples(changed, 1)


if __name__ == '__main__': unittest.main()
