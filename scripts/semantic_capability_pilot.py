"""Predeclared train-only diagnostic of extraction tools and synthetic demos."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import re

from tableqa.cli import metadata
from tableqa.data import read_jsonl, write_jsonl, digest, table_fingerprint
from tableqa.evaluation import evaluate
from tableqa.experiments import settings
from tableqa.inference import predict
from tableqa.semantic_columns import extract_span, ExtractionError, uses_extraction


def category(sample):
    """Public-input heuristic, NOT a human label of tool necessity or correctness."""
    question = sample['question'].casefold()
    if re.search(r'\b(countr(?:y|ies)|nationality|nation)\b', question):
        for col in range(len(sample['table']['headers'])):
            codes = 0
            for row in sample['table']['rows']:
                try:
                    value = extract_span(row[col], 'parenthesized')['value']
                    codes += bool(re.fullmatch(r'[A-Za-z]{2,3}', value))
                except ExtractionError:
                    pass
            if codes >= 2:
                return 'parenthesized_code'
    if re.search(r'\b(place|position|rank|finish|finisher|placed|highest|lowest|first|last)\w*\b', question):
        for col, header in enumerate(sample['table']['headers']):
            if not re.search(r'\b(rank|position|place)\w*\b', header.casefold()):
                continue
            ranks = 0
            for row in sample['table']['rows']:
                value = row[col]
                if not isinstance(value, str) or re.fullmatch(r'\s*[0-9]+\s*', value):
                    continue
                try:
                    extract_span(value, 'ordinal'); ranks += 1
                except ExtractionError:
                    pass
            if ranks >= 2:
                return 'ordinal_rank'
    return None


def select_samples(rows, n_per_category=2, seed=2032):
    if type(n_per_category) is not int or n_per_category < 1:
        raise ValueError('Positive per-category cap required')
    if any(r['split'] != 'train' for r in rows):
        raise ValueError('This capability diagnostic only accepts training data')
    items = list(rows); random.Random(seed).shuffle(items)
    counts = {(d, c): 0 for d in ['wikitq', 'tabfact']
              for c in ['parenthesized_code', 'ordinal_rank']}
    selected = []; annotations = []; tables = set()
    for item in items:
        if item['dataset'] not in {'wikitq', 'tabfact'}:
            continue
        key = (item['dataset'], category(item))
        if key not in counts or counts[key] >= n_per_category:
            continue
        fingerprint = table_fingerprint(item['table'])
        if fingerprint in tables:
            continue
        tables.add(fingerprint); counts[key] += 1; selected.append(item)
        annotations.append({'dataset': item['dataset'], 'id': item['id'],
                            'heuristic_category': key[1], 'tool_necessity_human_verified': False})
        if all(n == n_per_category for n in counts.values()):
            break
    if any(n != n_per_category for n in counts.values()):
        raise ValueError(f'Insufficient training input coverage: {counts}')
    return selected, annotations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('data'); p.add_argument('model'); p.add_argument('output')
    p.add_argument('--n-per-category', type=int, default=2)
    p.add_argument('--seed', type=int, default=2032)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args(); root = Path(args.output)
    if root.exists():
        raise ValueError('Output exists; do not alter or rerun this frozen diagnostic')
    samples, annotations = select_samples(read_jsonl(args.data), args.n_per_category, args.seed)
    configs = {name: settings('A4', semantic_extract=tools, tool_demonstrations=demos, max_repairs=1)
               for name, tools, demos in [('plain', False, False), ('tools', True, False),
                                          ('tools_demo', True, True)]}
    model_config = {'backend': 'local', 'model': str(Path(args.model).resolve()),
                    'device_map': 'balanced', 'max_input_tokens': 8192,
                    'max_new_tokens': 1536, 'seed': 42}
    protocol = {'train_only': True, 'selection_seed': args.seed, 'n_per_category': args.n_per_category,
                'selection_uses_labels_predictions_or_answers': False,
                'selection_uses_public_question_and_table': True,
                'diagnostic_not_overall_accuracy': True, 'one_question_per_table': True,
                'categories_not_human_verified': True, 'samples_sha256': digest(samples),
                'conditions': {k: asdict(v) for k, v in configs.items()}, 'model_config': model_config,
                'order': 'rotate condition order by sample index', 'formal_latency_benchmark': False,
                'hypotheses': ['Does the teacher use a source-copy operation?',
                               'Do synthetic tool demonstrations change tool adoption?',
                               'Are process and answer failures reduced on these inputs?'],
                'limits': 'Small heuristic training subset; copied text does not prove semantic correctness; no faithful Chain-of-Table comparison in this diagnostic.'}
    root.mkdir(parents=True)
    write_jsonl(root / 'samples.jsonl', samples); write_jsonl(root / 'categories.jsonl', annotations)
    (root / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    (root / 'run.json').write_text(json.dumps(metadata(protocol), ensure_ascii=False, indent=2), encoding='utf-8')
    if args.prepare_only:
        print('SEMANTIC_DIAGNOSTIC_PREPARED', len(samples), flush=True); return
    from tableqa.models import make_model
    model = make_model(model_config); outputs = {name: [] for name in configs}; names = list(configs)
    for i, item in enumerate(samples):
        offset = i % len(names)
        for name in names[offset:] + names[:offset]:
            result = predict(item, model, configs[name])
            result['metrics']['latency_comparable'] = False
            result['extraction_used_in_committed_chain'] = uses_extraction(result['operations'])
            result['extraction_mentioned_in_response'] = any('"extract"' in r['text'] for r in result['responses'])
            outputs[name].append(result)
            write_jsonl(root / f'{name}-predictions.jsonl', outputs[name])
            (root / 'progress.json').write_text(json.dumps({k: len(v) for k, v in outputs.items()}), encoding='utf-8')
            print(json.dumps({'condition': name, 'id': item['id'], 'status': result['status'],
                              'extract_used': result['extraction_used_in_committed_chain']}), flush=True)
    report = {'diagnostic_only': True, 'official_wikitq_scores_required': True,
              'conditions': {k: evaluate(v, samples) for k, v in outputs.items()},
              'adoption': {k: {'committed': sum(r['extraction_used_in_committed_chain'] for r in v),
                                'attempted': sum(r['extraction_mentioned_in_response'] for r in v)}
                           for k, v in outputs.items()}}
    (root / 'diagnostic.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('SEMANTIC_DIAGNOSTIC_COMPLETE', flush=True)


if __name__ == '__main__': main()
