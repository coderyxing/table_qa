"""Run a synthetic text replay through the verified official planning/operators."""
import argparse
import json
from tableqa.chain_of_table import load_upstream, predict_tabfact
from tableqa.data import example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_audit'); args = parser.parse_args()
    chain, _ = load_upstream(args.source_audit)
    responses = ['f_select_column(name, score)'] + ["f_col(['name', 'score'])"] * 8 + ['<END>', 'YES']
    class Backend:
        def __init__(self, outputs):
            self.outputs = iter(outputs)
            self.prompts = []
        def generate_text(self, messages, **kwargs):
            if 'PRIVATE_GOLD_SENTINEL' in str(messages):
                raise ValueError('Reference leaked into an upstream prompt')
            self.prompts.append(messages[-1]['content'])
            text = next(self.outputs)
            return {'text': text, 'input_tokens': 10, 'output_tokens': 5,
                    'model_seconds': 0, 'cache_hit': False}
    sample = example('tabfact', 'train', 'synthetic-cot', 'synthetic-cot',
                     ['name', 'score'], [['A', '1'], ['B', '2'], ['C', '3']],
                     'B has score 2.', 'PRIVATE_GOLD_SENTINEL')
    result = predict_tabfact(sample, Backend(responses), chain)
    if result['status'] != 'ok' or result['answer'] != 1:
        raise ValueError(f'Official source replay failed: {result["failure"]}')
    if result['metrics']['model_calls'] != 11:
        raise ValueError('Original eight-candidate selection was not preserved')
    raw = 'RAW_STATEMENT has year 2024.'
    cleaned = 'DATER_STATEMENT has year 481.'
    add_sample = example('tabfact', 'train', 'synthetic-dater', 'synthetic-dater',
                         ['name and country'], [['A (US)'], ['B (FR)'], ['C (JP)']],
                         raw, 'PRIVATE_GOLD_SENTINEL')
    add_outputs = ['f_add_column(country)',
                   'f_add_column(country)\nWe extract the value from name and country.\nThe value: US | FR | JP',
                   '<END>', 'YES']
    for mapping, hit, expected in [({raw: cleaned}, True, 'DATER_STATEMENT has year _.'),
                                   ({'other statement': cleaned}, False, 'RAW_STATEMENT has year _.')]:
        backend = Backend(add_outputs)
        mapped = predict_tabfact(add_sample, backend, chain, dater_mapping=mapping)
        if mapped['status'] != 'ok' or mapped['dater_map_hit'] != hit or mapped['dater_text_changed'] != hit:
            raise ValueError(f'Dater replay failed: {mapped["failure"]}')
        if len(backend.prompts) != 4 or expected not in backend.prompts[1]:
            raise ValueError('The add-column prompt must receive the mapped or fallback statement with digits masked')
        for index in [0, 2, 3]:
            if raw not in backend.prompts[index] or 'DATER_STATEMENT' in backend.prompts[index]:
                raise ValueError('Planning and final query must retain the original statement')
    print(json.dumps({'fixture_only': True, 'upstream_source_replay': 'passed',
                      'candidate_generations': result['metrics']['model_calls'],
                      'logical_requests': result['metrics']['upstream_api_requests'],
                      'answer': result['answer'], 'dater_hit_and_fallback_prompt_replays': 'passed',
                      'not_a_model_accuracy_result': True}))


if __name__ == '__main__': main()
