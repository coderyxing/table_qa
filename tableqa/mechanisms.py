"""Failure/repair diagnostics; recovery of execution is not semantic correctness."""
from collections import Counter
import random
from .analysis import validate_run, official_lookup, primary_score, metric_name, record_hash
from .quality import audit_trajectory


def error_kind(message):
    text = message.casefold()
    for fragments, kind in [
        (('configured context',), 'context_limit'),
        (('enclosed in double quotes', 'expecting', 'extra data', 'expected json'), 'json_format'),
        (('projection removes future dependency',), 'dependency_violation'),
        (('unavailable columns', 'missing column'), 'invalid_reference'),
        (('ambiguous numeric', 'not a number', 'non-finite'), 'numeric_interpretation'),
        (('expression must', 'expression comparison type', "'expression'", 'unhashable'), 'expression_schema'),
        (('answer must terminate', 'complete plan must terminate'), 'terminal_answer_contract'),
        (('assertion', 'assert',), 'assertion_contract'),
        (('feedback-dependent',), 'feedback_boundary'),
        (('budget',), 'operation_budget'),
        (('unknown operation', 'unknown comparator', 'unsupported selection'), 'unsupported_operation')]:
        if any(fragment in text for fragment in fragments): return kind
    return 'other'


def analyze_mechanisms(samples, predictions, official_report=None, seed=42, examples_per_category=3):
    refs, preds, _ = validate_run(samples, predictions)
    wtq = {k for k in refs if k[0] == 'wikitq'}
    official = official_lookup(official_report, wtq, samples, predictions)
    rows = []; example_groups = {}; events = []
    for dataset in sorted({k[0] for k in refs}):
        keys = sorted(k for k in refs if k[0] == dataset)
        repairs = []; counts = Counter(); first_errors = Counter(); final_errors = Counter(); accepted = Counter()
        for key in keys:
            p, sample = preds[key], refs[key]
            repaired = p['metrics'].get('repairs', 0) > 0
            errors = p.get('errors', [])
            score = primary_score(p, sample, official)
            if p['status'] == 'ok':
                counts['execution_ok'] += 1
                counts['execution_ok_after_repair' if repaired else 'execution_ok_first_attempt'] += 1
                if score == 1 and dataset != 'fetaqa':
                    counts['correct_after_repair' if repaired else 'correct_first_attempt'] += 1
            else:
                message = (p.get('failure') or {}).get('message', '')
                kind = error_kind(message); final_errors[kind] += 1
                if kind == 'context_limit':
                    counts['context_before_any_generation' if not p['responses'] else 'context_after_generation'] += 1
                example_groups.setdefault((dataset, kind), []).append({
                    'dataset': dataset, 'id': sample['id'], 'question': sample['question'],
                    'reference': sample['reference'], 'table': sample['table'],
                    'failure': p['failure'], 'errors': errors,
                    'generated_responses': [r['text'] for r in p['responses']],
                    'selection_seed': seed, 'diagnostic_only': True})
            if errors: first_errors[error_kind(errors[0]['message'])] += 1
            if repaired: repairs.append(p)
            events.append({'dataset': dataset, 'id': sample['id'], 'repaired': repaired,
                           'execution_ok': p['status'] == 'ok', 'task_score': score,
                           'first_observed_error': errors[0]['message'] if errors else None,
                           'final_failure': p.get('failure'), 'diagnostic_only': True})
            if sample['split'] == 'train':
                audit = audit_trajectory(sample, p); accepted[audit['decision']] += 1
                if dataset == 'wikitq' and score == 1 and audit['decision'] == 'rejected':
                    counts['official_correct_but_weak_filter_rejected'] += 1
        rows.append({'dataset': dataset, 'n': len(keys), 'metric': metric_name(dataset, official),
                     'counts': dict(counts), 'repair_attempted': len(repairs),
                     'execution_recovery_given_repair': sum(p['status'] == 'ok' for p in repairs)/len(repairs) if repairs else None,
                     'first_observed_error_categories': dict(first_errors), 'final_failure_categories': dict(final_errors),
                     'training_quality_decisions': dict(accepted),
                     'limitations': 'Observed errors are format/execution errors. No first semantic-error oracle; repairs are not a causal ablation.'})
    examples = []
    for group, pool in sorted(example_groups.items()):
        random.Random(seed).shuffle(pool)
        for record in pool[:examples_per_category]: examples.append({**record, 'category': group[1]})
    return {'development_only': all(s['split'] == 'train' for s in samples),
            'samples_sha256': record_hash(samples), 'predictions_sha256': record_hash(predictions), 'rows': rows}, events, examples
