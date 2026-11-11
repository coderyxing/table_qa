# 测试范围：在独立进程内加载已校验的上游 Chain-of-Table 源码，适配本地文本模型及 TabFact 输出。
# 运行方式：python -m unittest discover -s tests -p test_chain_of_table.py -v
# 阅读顺序：先看样本或替身模型的构造，再看每个用例前的预期约束。
import math
import unittest
from copy import deepcopy
from tableqa.chain_of_table import (UpstreamTextAdapter, public_tabfact,
                                   parse_tabfact_text, safe_source_eval)
from tableqa.data import example


# TextBackend：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
class TextBackend:
    # 初始化 TextBackend 的依赖及持久状态；具体预算和身份字段见赋值。
    def __init__(self): self.calls = []
    # 使用上游文本生成协议返回候选文本和生成使用量。
    # 输入：messages（模型可见消息）；**kwargs。
    # 返回形式含：字段 text、raw_text、input_tokens、output_tokens、model_seconds、cache_hit。
    def generate_text(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {'text': f'text response {len(self.calls)}', 'raw_text': 'public text',
                'input_tokens': 20, 'output_tokens': 5, 'model_seconds': .1, 'cache_hit': False}


# ChainOfTableAdapterTests：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
class ChainOfTableAdapterTests(unittest.TestCase):
    # 回归用例：test_all_eight_candidates_keep_original_order_proxy_and_costs
    # 应相等：len(results)，8
    # 应相等：len(backend.calls)，8
    # 应相等：len(adapter.calls)，8
    def test_all_eight_candidates_keep_original_order_proxy_and_costs(self):
        backend = TextBackend(); adapter = UpstreamTextAdapter(backend)
        options = adapter.get_model_options(temperature=.5, n_sample=8)
        results = adapter.generate_plus_with_score('table and statement', options, end_str='\n\n')
        self.assertEqual(len(results), 8)
        self.assertEqual(len(backend.calls), 8)
        self.assertEqual(len(adapter.calls), 8)
        self.assertEqual(adapter.requests, 1)
        self.assertEqual([round(math.exp(score), 5) for _, score in results],
                         [1, .875, .75, .625, .5, .375, .25, .125])
        self.assertEqual(sum(r['input_tokens'] for r in adapter.calls), 160)
        self.assertEqual(backend.calls[0][1]['stop'], '\n\n')
        self.assertEqual(backend.calls[0][1]['temperature'], .5)

    # 回归用例：test_generate_does_not_mutate_options_or_keep_eight_candidates
    # 应相等：options，original
    # 应相等：len(backend.calls)，1
    def test_generate_does_not_mutate_options_or_keep_eight_candidates(self):
        backend = TextBackend(); adapter = UpstreamTextAdapter(backend)
        options = adapter.get_model_options(n_sample=8); original = deepcopy(options)
        adapter.generate('prompt', options)
        self.assertEqual(options, original)
        self.assertEqual(len(backend.calls), 1)

    # 回归用例：test_cap_is_checked_before_partial_candidate_request
    # 应相等：adapter.calls，[]
    # 应相等：backend.calls，[]
    # 应抛出预期异常：ValueError
    def test_cap_is_checked_before_partial_candidate_request(self):
        backend = TextBackend(); adapter = UpstreamTextAdapter(backend, max_generations=2)
        with self.assertRaises(ValueError):
            adapter.generate_plus_with_score('prompt', adapter.get_model_options(n_sample=8))
        self.assertEqual(adapter.calls, [])
        self.assertEqual(backend.calls, [])
        for options in [{'n': True}, {'n': 0}, {'n': 9}, {'max_tokens': 0}, {'temperature': float('nan')}]:
            with self.assertRaises(ValueError): adapter.generate_plus_with_score('p', options)

    # 回归用例：test_backend_failure_is_recorded_and_not_hidden_by_retry
    # 应相等：len(adapter.calls)，1
    # 应相等：adapter.calls[0]['status']，'failed'
    # 应相等：adapter.calls[0]['error']['message']，'context exceeded'
    def test_backend_failure_is_recorded_and_not_hidden_by_retry(self):
        # Failing：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
        class Failing:
            # 使用上游文本生成协议返回候选文本和生成使用量。
            # 输入：*args；**kwargs。
            # 异常约束：context exceeded
            def generate_text(self, *args, **kwargs): raise ValueError('context exceeded')
        adapter = UpstreamTextAdapter(Failing())
        with self.assertRaises(ValueError): adapter.generate('p')
        self.assertEqual(len(adapter.calls), 1)
        self.assertEqual(adapter.calls[0]['status'], 'failed')
        self.assertEqual(adapter.calls[0]['error']['message'], 'context exceeded')

    # 回归用例：test_public_input_excludes_labels_and_gold
    # 应排除：'PRIVATE'，str(public)
    # 应排除：'label'，public
    # 应相等：public['table_text']，[['a'], ['2']]
    def test_public_input_excludes_labels_and_gold(self):
        sample = example('tabfact', 'train', 'x', 't', ['a'], [[2]], 'The value is 2.', 1)
        sample.update(label='PRIVATE_LABEL', highlights='PRIVATE_HIGHLIGHT', reference='PRIVATE_REFERENCE')
        public = public_tabfact(sample)
        self.assertNotIn('PRIVATE', str(public))
        self.assertNotIn('label', public)
        self.assertEqual(public['table_text'], [['a'], ['2']])
        sample['dataset'] = 'wikitq'
        with self.assertRaises(ValueError): public_tabfact(sample)

    # 回归用例：test_only_exact_upstream_labels_are_accepted
    # 应相等：parse_tabfact_text(text)，expected
    # 应为 None：parse_tabfact_text(text)
    def test_only_exact_upstream_labels_are_accepted(self):
        for text, expected in [('YES', 1), ('true', 1), (' NO ', 0), ('False', 0)]:
            self.assertEqual(parse_tabfact_text(text), expected)
        for text in ['yes because', 'NO.', '0', '1', '{"answer":1}']:
            self.assertIsNone(parse_tabfact_text(text))

    # 回归用例：test_safe_literal_and_static_lookup_do_not_execute_candidate_code
    # 核验约束：safe_source_eval('select_row_act', {'select_row_act': value})，value
    # 应相等：safe_source_eval("['1','2']", {})，['1', '2']
    # 应相等：safe_source_eval("('x', ['y'])", {})，('x', ['y'])
    def test_safe_literal_and_static_lookup_do_not_execute_candidate_code(self):
        value = object()
        self.assertIs(safe_source_eval('select_row_act', {'select_row_act': value}), value)
        self.assertEqual(safe_source_eval("['1','2']", {}), ['1', '2'])
        self.assertEqual(safe_source_eval("('x', ['y'])", {}), ('x', ['y']))
        for text in ['__import__("os")', '1+2', 'globals()', 'unknown_act', 'x' * 131073]:
            with self.assertRaises((ValueError, SyntaxError)): safe_source_eval(text, {})


if __name__ == '__main__': unittest.main()
