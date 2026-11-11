# 测试范围：用精确字符跨度提取派生列，保留当前单元格与原表来源之间的审核关系。
# 运行方式：python -m unittest discover -s tests -p test_semantic_columns.py -v
# 阅读顺序：先看样本或替身模型的构造，再看每个用例前的预期约束。
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


# 构造当前测试使用的独立规范样本，固定题面、原表和参考结果。
def sample():
    return example('wikitq', 'train', 'extract-fixture', 'extract-fixture',
                   ['Athlete', 'Rank'], [['Rina (RED)', '12th'], ['Kaito (BLU)', '3rd'],
                                         ['Noor (RED)', 'T3']],
                   'Which affiliation code has most athletes?', ['RED'])


# 构造测试使用的白名单计划，固定列和预期执行关系。
# 返回形式含：列表记录。
def plan():
    return [{'op': 'derive', 'as': 'affiliation', 'expression':
             {'op': 'extract', 'args': [{'col': 'c0'}], 'rule': 'parenthesized'}},
            {'op': 'group_aggregate', 'by': ['affiliation'], 'aggregations': [{'fn': 'count', 'as': 'n'}]},
            {'op': 'select', 'mode': 'extreme', 'column': 'n', 'which': 'max'},
            {'op': 'answer', 'columns': ['affiliation']}]


# ExtractionTests：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
class ExtractionTests(unittest.TestCase):
    # 回归用例：test_offsets_copy_unicode_and_trim_without_inventing_text
    # 应相等：first['value']，'红队'
    # 应相等：raw[first['start']:first['end']]，first['value']
    # 应相等：extract_span(raw, 'parenthesized', -1)['value']，'reserve'
    def test_offsets_copy_unicode_and_trim_without_inventing_text(self):
        raw = '林 ( 红队 ) (reserve)'
        first = extract_span(raw, 'parenthesized')
        self.assertEqual(first['value'], '红队')
        self.assertEqual(raw[first['start']:first['end']], first['value'])
        self.assertEqual(extract_span(raw, 'parenthesized', -1)['value'], 'reserve')
        self.assertEqual(first['semantic_correctness'], 'not_proven')

    # 回归用例：test_unbalanced_nested_missing_and_empty_fields_fail
    # 应抛出预期异常：ExtractionError
    def test_unbalanced_nested_missing_and_empty_fields_fail(self):
        for text in ['A (X', 'A X)', 'A ((X))', 'A', 'A ()', 'A ( )']:
            with self.subTest(text=text), self.assertRaises(ExtractionError):
                extract_span(text, 'parenthesized')
        for index in [True, 1.0, 999]:
            with self.assertRaises(ExtractionError): extract_span('A (B)', 'parenthesized', index)

    # 回归用例：test_literal_split_preserves_exact_span_and_requires_delimiter
    # 应相等：result['value']，'Country'
    # 应相等：raw[result['start']:result['end']]，'Country'
    # 应抛出预期异常：ExtractionError
    def test_literal_split_preserves_exact_span_and_requires_delimiter(self):
        raw = 'City,  Region, Country '
        result = extract_span(raw, 'split', -1, ',')
        self.assertEqual(result['value'], 'Country')
        self.assertEqual(raw[result['start']:result['end']], 'Country')
        for delimiter in ['', None, ';', 'x' * 33]:
            with self.assertRaises(ExtractionError): extract_span(raw, 'split', delimiter=delimiter)
        with self.assertRaises(ExtractionError): extract_span('a,,b', 'split', 1, ',')

    # 回归用例：test_complete_rank_parsing_preserves_ties_and_validates_suffix
    # 应相等：final.answer，['Kaito (BLU)', 'Noor (RED)']
    # 应相等：extract_span(value, 'ordinal')['value']，expected
    # 应抛出预期异常：ExtractionError
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

    # 回归用例：test_replay_retains_raw_cells_lineage_and_character_evidence
    # 应相等：result.answer，['RED']
    # 应相等：initial.signature()，signature
    # 应相等：len(spans)，3
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

    # 回归用例：test_missing_structure_does_not_silently_discard_rows
    # 应相等：original，state.signature()
    # 应抛出预期异常：ExecutionError
    def test_missing_structure_does_not_silently_discard_rows(self):
        data = sample(); data['table']['rows'][1][0] = 'Kaito'
        state = TableState.from_table(data['table']); original = state.signature()
        with self.assertRaises(ExecutionError): run_plan(state, plan())
        self.assertEqual(original, state.signature())

    # 回归用例：test_only_direct_cell_arguments_are_accepted
    # 应抛出预期异常：ExecutionError
    # 应抛出预期异常：ExtractionError
    def test_only_direct_cell_arguments_are_accepted(self):
        for args in [[], ['forged (X)'], [{'literal': 'forged (X)'}],
                     [{'op': 'add', 'args': [1, 2]}], [{'col': 'c0'}, {'col': 'c1'}]]:
            expr = {'op': 'extract', 'args': args, 'rule': 'parenthesized'}
            with self.assertRaises(ExecutionError): evaluate_expression(expr, {'c0': 'Rina (RED)'})
        for rule in ['regex', 'eval', '__import__']:
            with self.assertRaises(ExtractionError): extract_span('x', rule)

    # 回归用例：test_lazy_branch_audit_does_not_execute_unused_extraction
    # 应相等：[r['derived'] for r in final.rows]，['RED', 'unknown']
    # 应相等：len(trace[0]['extractions'])，1
    def test_lazy_branch_audit_does_not_execute_unused_extraction(self):
        state = TableState.from_table({'headers': ['x'], 'rows': [['Rina (RED)'], ['No parentheses']]})
        expression = {'op': 'if', 'args': [
            {'op': 'eq', 'type': 'text', 'args': [{'col': 'c0'}, 'Rina (RED)']},
            {'op': 'extract', 'args': [{'col': 'c0'}], 'rule': 'parenthesized'}, 'unknown']}
        final, trace = run_plan(state, [{'op': 'derive', 'as': 'derived', 'expression': expression}])
        self.assertEqual([r['derived'] for r in final.rows], ['RED', 'unknown'])
        self.assertEqual(len(trace[0]['extractions']), 1)

    # 回归用例：test_spans_of_derived_cells_are_explicitly_scoped
    # 应相等：record['span_scope']，'current_source_cell'
    # 应相等：record['source_cells']，['r0:c0']
    # 应相等：record['start']，6
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

    # 回归用例：test_opt_in_prompts_and_private_reference_isolation
    # 应相等：result['status']，'ok'
    # 应相等：system_prompt('B5')，SYSTEM
    # 应相等：system_prompt('B0', True, True, True)，system_prompt('B0')
    def test_opt_in_prompts_and_private_reference_isolation(self):
        item = sample(); item['reference'] = ['PRIVATE_SENTINEL']
        # Spy：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
        class Spy(ReplayModel):
            # 按当前适配接口生成或重放响应，同时保留调用所需的观测状态。
            # 输入：messages（模型可见消息）。
            # 相关调用：super().generate。
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

    # 回归用例：test_training_flags_match_inference_and_answer_controls_reuse_pool
    # 应相等：rows[0]['messages'][0]['content']，system_prompt('B5', False, True, True)
    # 应包含：'Synthetic tool demonstration'，rows[0]['messages'][0]['content']
    # 应排除：'extract-v1'，str(build_records(sample(), plan(), 'B0')[0]['messages'])
    def test_training_flags_match_inference_and_answer_controls_reuse_pool(self):
        with self.assertRaises(ValueError): build_records(sample(), plan())
        rows = build_records(sample(), plan(), semantic_extract=True, tool_demonstrations=True)
        self.assertEqual(rows[0]['messages'][0]['content'], system_prompt('B5', False, True, True))
        self.assertIn('Synthetic tool demonstration', rows[0]['messages'][0]['content'])
        self.assertNotIn('extract-v1', str(build_records(sample(), plan(), 'B0')[0]['messages']))

    # 回归用例：test_partition_charges_enabled_tool_and_demonstration_tokens
    # 左值应更大：enabled['cost']，bare['cost']
    # 应抛出预期异常：ValueError
    def test_partition_charges_enabled_tool_and_demonstration_tokens(self):
        # Tokenizer：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
        class Tokenizer:
            # 用确定性的测试聊天模板表示消息，便于核对提示边界和成本。
            # 输入：messages（模型可见消息）；**kwargs。
            def apply_chat_template(self, messages, **kwargs): return list(json.dumps(messages))
            # 用确定性的测试编码规则返回 token 序列，避免依赖实际模型权重。
            # 输入：value；**kwargs。
            def encode(self, value, **kwargs): return list(value)
        with self.assertRaises(ValueError):
            state_cost_partition(sample(), plan(), Tokenizer(), max_length=100000, max_output_tokens=10000)
        _, bare = state_cost_partition(sample(), plan(), Tokenizer(), max_length=100000,
            max_output_tokens=10000, semantic_extract=True)
        _, enabled = state_cost_partition(sample(), plan(), Tokenizer(), max_length=100000,
            max_output_tokens=10000, semantic_extract=True, tool_demonstrations=True)
        self.assertGreater(enabled['cost'], bare['cost'])

    # 回归用例：test_pilot_selection_does_not_read_gold_and_rejects_test_data
    # 应相等：[r['id'] for r in first]，[r['id'] for r in second]
    # 应抛出预期异常：ValueError
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
