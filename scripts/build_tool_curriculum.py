"""Create labelled SYNTHETIC tool-use training templates, not benchmark scores.

Answers are independently computed from generator variables, then checked
against execution. No benchmark question, answer, validation or test is read.
This pool is a training candidate; it is not automatically merged or adopted.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

from tableqa.core import TableState, run_plan
from tableqa.data import example, write_jsonl, digest
from tableqa.distillation import build_records
from tableqa.evaluation import diagnostic_score


def col(name): return {'col': name}
def binary(op, a, b): return {'op': op, 'args': [a, b]}
def derive(name, expr): return {'op': 'derive', 'as': name, 'expression': expr}
def extract(column, rule, **kwargs): return {'op': 'extract', 'args': [col(column)], 'rule': rule, **kwargs}
def equal(column, value): return {'op': 'filter', 'column': column, 'cmp': 'eq', 'value': value, 'type': 'text'}
def answer(column): return {'op': 'answer', 'columns': [column]}


def make_pool(per_family=16, seed=2033):
    if type(per_family) is not int or per_family < 1: raise ValueError('Positive family size required')
    rng = random.Random(seed); samples = []; trajectories = []
    for family in ['affiliation_count', 'ordinal_scope', 'round_mean', 'unit_extreme', 'crossrow_difference', 'universal_scope']:
        for i in range(per_family):
            tag = f'{family}-{seed}-{i}'
            if family == 'affiliation_count':
                codes = [rng.choice(['RED', 'BLU', 'GRN']) for _ in range(7)]
                counts = Counter(codes); maximum = max(counts.values())
                headers = ['Athlete']; rows = [[f'Person{tag}_{j} ({code})'] for j, code in enumerate(codes)]
                question = 'Which affiliation code has the most athletes? Include all tied codes.'
                reference = [c for c in dict.fromkeys(codes) if counts[c] == maximum]; dataset = 'wikitq'
                ops = [derive('code', extract('c0', 'parenthesized')),
                       {'op': 'group_aggregate', 'by': ['code'], 'aggregations': [{'fn': 'count', 'as': 'n'}]},
                       {'op': 'select', 'mode': 'extreme', 'column': 'n', 'which': 'max'}, answer('code')]
            elif family == 'ordinal_scope':
                ranks = [rng.randint(1, 20) for _ in range(5)]; ranks[-1] = ranks[0]
                def ordinal(n):
                    suffix = 'th' if 10 < n % 100 < 14 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
                    return str(n) + suffix
                headers = ['Competitor', 'Recorded position']
                rows = [[f'Unranked{tag}', '\u2014']] + [[f'Person{tag}_{j}', ordinal(n)] for j, n in enumerate(ranks)]
                question = 'Among competitors with a recorded finishing position, who has the greatest numeric position? Include all ties.'
                reference = [f'Person{tag}_{j}' for j, n in enumerate(ranks) if n == max(ranks)]; dataset = 'wikitq'
                ops = [{'op': 'filter', 'column': 'c1', 'cmp': 'ne', 'value': '\u2014', 'type': 'text'},
                       derive('position', extract('c1', 'ordinal')),
                       {'op': 'select', 'mode': 'extreme', 'column': 'position', 'which': 'max'}, answer('c0')]
            elif family == 'round_mean':
                a, b = rng.randint(50, 90), rng.randint(50, 90)
                correct = (a + b) / 2; claimed = correct if i % 2 == 0 else correct + 1
                headers = ['Player', 'Two round scores']
                rows = [[f'Player{tag}', f'{a} + {b} = {a+b}'], ['Other', '60 + 64 = 124']]
                question = f'The average of Player{tag}\'s first two round scores equals {claimed}.'
                reference = int(claimed == correct); dataset = 'tabfact'
                ops = [equal('c0', f'Player{tag}'),
                       derive('rounds', extract('c1', 'split', delimiter='=', index=0)),
                       derive('first', extract('rounds', 'split', delimiter='+', index=0)),
                       derive('second', extract('rounds', 'split', delimiter='+', index=1)),
                       derive('mean', binary('div', binary('add', col('first'), col('second')), 2)),
                       {'op': 'answer', 'assert': {'kind': 'cell', 'column': 'mean', 'cmp': 'eq', 'value': claimed, 'type': 'number'}}]
            elif family == 'unit_extreme':
                distances = [rng.randint(1, 15) * 100 for _ in range(5)]
                headers = ['Entry', 'Distance']
                rows = [[f'Entry{tag}_{j}', f'{distance} m' if j % 2 == 0 else f'{distance/1000:g} km']
                        for j, distance in enumerate(distances)]
                question = 'Which entries have the longest distance? Include all ties.'
                reference = [f'Entry{tag}_{j}' for j, n in enumerate(distances) if n == max(distances)]; dataset = 'wikitq'
                ops = [derive('distance_m', {'op': 'quantity', 'args': [col('c1')], 'unit': 'm'}),
                       {'op': 'select', 'mode': 'extreme', 'column': 'distance_m', 'which': 'max'}, answer('c0')]
            elif family == 'crossrow_difference':
                totals = [rng.randint(1, 30) for _ in range(4)]
                headers = ['Rank', 'Team', 'Total']
                rows = [[str(j+1), f'Team{tag}_{j}', str(n)] for j, n in enumerate(totals)]
                question = 'What is the total for rank 1 minus the total for rank 2?'
                reference = [totals[0] - totals[1]]; dataset = 'wikitq'
                def scoped_value(rank):
                    return {'op': 'if', 'args': [{'op': 'eq', 'type': 'text', 'args': [col('c0'), rank]}, col('c2'), 0]}
                ops = [derive('first_total', scoped_value('1')), derive('second_total', scoped_value('2')),
                       {'op': 'group_aggregate', 'by': [], 'aggregations': [
                           {'fn': 'sum', 'column': 'first_total', 'as': 'a'}, {'fn': 'sum', 'column': 'second_total', 'as': 'b'}]},
                       derive('difference', binary('sub', col('a'), col('b'))), answer('difference')]
            else:
                threshold = rng.randint(3, 10)
                blue_scores = [threshold + 1, threshold + 2] if i % 2 == 0 else [threshold, threshold + 2]
                headers = ['Entry', 'Region', 'Score']
                rows = [[f'Entry{tag}_0', 'BLUE', str(blue_scores[0])],
                        [f'Entry{tag}_1', 'BLUE', str(blue_scores[1])], ['Other', 'RED', '0']]
                question = f'Every entry from region BLUE has a score greater than {threshold}.'
                reference = int(all(n > threshold for n in blue_scores)); dataset = 'tabfact'
                ops = [equal('c1', 'BLUE'), {'op': 'filter', 'column': 'c2', 'cmp': 'le', 'value': threshold, 'type': 'number'},
                       {'op': 'answer', 'assert': {'kind': 'row_count', 'cmp': 'eq', 'value': 0}}]
            sample = example(dataset, 'train', 'synthetic-' + tag, 'synthetic-' + tag,
                             headers, rows, question, reference,
                             synthetic=True, template_family=family,
                             reference_basis='independent template variables; not generated by model execution')
            final, trace = run_plan(TableState.from_table(sample['table']), ops)
            if len(ops) > 8 or diagnostic_score({'status': 'ok', 'answer': final.answer}, sample) != 1:
                raise ValueError('Synthetic template/program disagreement')
            samples.append(sample); trajectories.append({'id': sample['id'], 'dataset': dataset,
                'operations': ops, 'trace': trace, 'answer': final.answer,
                'synthetic': True, 'not_benchmark_evidence': True})
    return samples, trajectories


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output'); p.add_argument('--per-family', type=int, default=16)
    p.add_argument('--seed', type=int, default=2033); args = p.parse_args(); root = Path(args.output)
    if root.exists(): raise ValueError('Output exists; preserve this candidate pool')
    samples, traces = make_pool(args.per_family, args.seed); records = []
    for sample, trace in zip(samples, traces):
        records += build_records(sample, trace['operations'], typed_quantities=True,
                                 semantic_extract=True, tool_demonstrations=True)
    root.mkdir(parents=True); write_jsonl(root/'samples.jsonl', samples)
    write_jsonl(root/'trajectories.jsonl', traces); write_jsonl(root/'records.jsonl', records)
    report = {'synthetic_only': True, 'not_a_benchmark_result': True, 'seed': args.seed,
              'per_family': args.per_family, 'sources': len(samples), 'records': len(records),
              'families': dict(Counter(s['template_family'] for s in samples)),
              'tasks': dict(Counter(s['dataset'] for s in samples)),
              'tabfact_labels': dict(Counter(str(s['reference']) for s in samples if s['dataset']=='tabfact')),
              'all_programs_checked_against_independent_template_answers': True,
              'samples_sha256': digest(samples), 'records_sha256': digest(records),
              'automatically_adopted_into_real_pool': False,
              'limits': 'Authored templates demonstrate tool syntax and scopes, not open-domain reasoning or benchmark improvement.'}
    (root/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__': main()
