"""Bounded, train-only paired capability diagnostic, not a benchmark result.

Inputs are selected without questions' labels or prediction scores. Both
conditions share a single teacher instance. The only intervention is exposing
the experimental quantity expression. Original tables are never rewritten.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

from tableqa.cli import metadata
from tableqa.data import read_jsonl, write_jsonl, digest, table_fingerprint
from tableqa.evaluation import evaluate
from tableqa.experiments import settings
from tableqa.inference import predict
from tableqa.quantities import parse_quantity, QuantityError, uses_quantities


def select_samples(rows, n_per_task=4, seed=2031):
    if type(n_per_task) is not int or n_per_task < 1:
        raise ValueError('A positive sample cap is required')
    candidates = list(rows)
    if any(r['split'] != 'train' for r in candidates):
        raise ValueError('This diagnostic accepts training inputs only')
    random.Random(seed).shuffle(candidates)
    counts = {'wikitq': 0, 'tabfact': 0}; selected = []; seen = set()
    for row in candidates:
        dataset = row['dataset']
        if dataset not in counts or counts[dataset] >= n_per_task:
            continue
        fingerprint = table_fingerprint(row['table'])
        if fingerprint in seen:
            continue
        has_quantity = False
        for cells in row['table']['rows']:
            for value in cells:
                try:
                    q = parse_quantity(value)
                    has_quantity |= q.source_unit != '1' and not q.assumed_unit_used
                except QuantityError:
                    pass
        if not has_quantity:
            continue
        selected.append(row); seen.add(fingerprint); counts[dataset] += 1
    if any(counts[d] != n_per_task for d in counts):
        raise ValueError(f'Insufficient eligible training tables: {counts}')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data'); parser.add_argument('model'); parser.add_argument('output')
    parser.add_argument('--n-per-task', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2031)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise ValueError('Output already exists; preserve the frozen diagnostic')
    samples = select_samples(read_jsonl(args.data), args.n_per_task, args.seed)
    out.mkdir(parents=True)
    model_config = {'backend': 'local', 'model': str(Path(args.model).resolve()),
                    'device_map': 'balanced', 'max_input_tokens': 8192,
                    'max_new_tokens': 512, 'seed': 42}
    configs = {name: settings('A4', typed_quantities=enabled, max_repairs=1)
               for name, enabled in [('plain', False), ('quantity', True)]}
    protocol = {'train_only': True, 'diagnostic_subset_not_general_accuracy': True,
                'selection': 'seeded training tables containing at least one full-cell explicit quantity',
                'selection_uses_labels_questions_or_predictions': False,
                'one_question_per_table': True, 'samples_sha256': digest(samples),
                'n_per_task': args.n_per_task, 'selection_seed': args.seed,
                'model_config': model_config,
                'conditions': {k: asdict(v) for k, v in configs.items()},
                'ordering': 'alternate first condition per sample',
                'formal_latency_benchmark': False,
                'changes': 'quantity prompt and whitelisted expression availability only',
                'limits': 'Selected tables may contain quantities irrelevant to the question. No overall benchmark or novelty claim.'}
    write_jsonl(out / 'samples.jsonl', samples)
    (out / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'run.json').write_text(json.dumps(metadata(protocol), ensure_ascii=False, indent=2), encoding='utf-8')
    if args.prepare_only:
        print('PREPARED', len(samples), flush=True)
        return
    from tableqa.models import make_model
    model = make_model(model_config)
    outputs = {name: [] for name in configs}
    for i, sample in enumerate(samples):
        order = ['plain', 'quantity'] if i % 2 == 0 else ['quantity', 'plain']
        for name in order:
            result = predict(sample, model, configs[name])
            result['metrics']['latency_comparable'] = False
            result['quantity_expression_used'] = uses_quantities(result['operations'])
            outputs[name].append(result)
            write_jsonl(out / f'{name}-predictions.jsonl', outputs[name])
            progress = {k: len(v) for k, v in outputs.items()}
            (out / 'progress.json').write_text(json.dumps(progress), encoding='utf-8')
            print(json.dumps({'id': sample['id'], 'condition': name, 'status': result['status'],
                              'quantity_used': result['quantity_expression_used'], 'progress': progress}), flush=True)
    report = {'diagnostic_only': True, 'official_wikitq_scores_still_required': True,
              'conditions': {k: evaluate(v, samples) for k, v in outputs.items()}}
    (out / 'diagnostic.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('QUANTITY_DIAGNOSTIC_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
