"""Frozen record schedules for a target-token/step matched curriculum pilot.

Exact target budgets do NOT imply equal input tokens, GPU time, source diversity
or real/synthetic distributions. Those quantities are reported independently.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import random
from .data import digest


def equal_kind_coverage(records, tokenizer, multiple=16, seed=42):
    """Keep every input record, then balance kinds and pad using paired replays."""
    if type(multiple) is not int or multiple < 2 or multiple % 2:
        raise ValueError('An even positive effective optimizer batch is required')
    pools = {kind: [r for r in records if r['kind'] == kind] for kind in ['block', 'answer']}
    if sum(map(len, pools.values())) != len(records) or not all(pools.values()):
        raise ValueError('Both valid supervision kinds are required')
    rng = random.Random(seed)
    n = max(map(len, pools.values()))
    n += (-2 * n) % multiple // 2
    result = []
    for kind in ['block', 'answer']:
        selected = list(pools[kind])
        while len(selected) < n:
            cycle = list(pools[kind]); rng.shuffle(cycle)
            selected += cycle[:n - len(selected)]
        result += selected
    rng.shuffle(result)
    return deepcopy(result)


def match_record_costs(pool, desired_costs, tokenizer, seed=42):
    """Match N complete target records and their total tokens exactly, with replay.

    Bitset DP enumerates reachable sums for exactly j records. Backtracking
    prefers costs close to each corresponding treatment slot, never cuts text.
    Pool costs only use training targets; no evaluation inputs or scores enter.
    """
    n = len(desired_costs); target = sum(desired_costs)
    if not pool or not 1 <= n <= 1024 or not 1 <= target <= 200000:
        raise ValueError('Bounded nonempty record/cost pools required')
    by_cost = defaultdict(list)
    for record in pool:
        by_cost[target_cost(record, tokenizer)].append(record)
    costs = sorted(by_cost)
    if n * min(costs) > target or n * max(costs) < target:
        raise ValueError('Requested target-token mean is outside the control support')
    mask = (1 << (target + 1)) - 1; reachable = [1]
    for _ in range(n):
        bits = 0
        for cost in costs:
            if cost <= target:
                bits |= reachable[-1] << cost
        reachable.append(bits & mask)
    if not ((reachable[-1] >> target) & 1):
        raise ValueError('Exact complete-record target budget is not reachable')
    rng = random.Random(seed); selected = []; remaining = target
    for j in range(n, 0, -1):
        preferred = desired_costs[j - 1]
        options = [cost for cost in costs if cost <= remaining and ((reachable[j - 1] >> (remaining - cost)) & 1)]
        rng.shuffle(options); options.sort(key=lambda c: abs(c - preferred))
        cost = options[0]; selected.append(deepcopy(rng.choice(by_cost[cost]))); remaining -= cost
    selected.reverse()
    if remaining or sum(target_cost(r, tokenizer) for r in selected) != target:
        raise ValueError('Exact matching invariant violated')
    return selected


def target_cost(record, tokenizer):
    return len(tokenizer.encode(record['target'], add_special_tokens=False)) + 1


def stratum(record):
    label = record.get('sampling_label') if record['dataset'] == 'tabfact' else None
    if record['dataset'] == 'tabfact' and (type(label) is not int or label not in (0, 1)):
        raise ValueError('Training-only TabFact labels are required for schedule auditing')
    return record['dataset'], record['kind'], label


def matched_schedules(real_records, synthetic_records, tokenizer, multiple=16, seed=42):
    common = equal_kind_coverage(real_records, tokenizer, multiple, seed)
    treatment_extra = equal_kind_coverage(synthetic_records, tokenizer, multiple, seed + 1)
    by_stratum = defaultdict(list)
    for record in real_records:
        by_stratum[stratum(record)].append(record)
    treatment_groups = defaultdict(list)
    for i, record in enumerate(treatment_extra):
        treatment_groups[stratum(record)].append((i, record))
    control_extra = [None] * len(treatment_extra); failures = []
    # Prefer stronger task/kind/label matching. The protocol allows a disclosed
    # kind-only fallback if real target lengths cannot support a stratum budget.
    for offset, (key, entries) in enumerate(sorted(treatment_groups.items())):
        try:
            selected = match_record_costs(by_stratum[key], [target_cost(r, tokenizer) for _, r in entries],
                                         tokenizer, seed + 10 + offset)
            for (i, _), record in zip(entries, selected): control_extra[i] = record
        except ValueError as exc:
            failures.append({'stratum': list(key), 'reason': str(exc)})
    if failures:
        control_extra = [None] * len(treatment_extra)
        for offset, kind in enumerate(['block', 'answer']):
            entries = [(i, r) for i, r in enumerate(treatment_extra) if r['kind'] == kind]
            selected = match_record_costs([r for r in real_records if r['kind'] == kind],
                [target_cost(r, tokenizer) for _, r in entries], tokenizer, seed + 50 + offset)
            for (i, _), record in zip(entries, selected): control_extra[i] = record
    pairs = [(deepcopy(r), deepcopy(r), 'common') for r in common]
    pairs += [(a, deepcopy(b), 'augmentation') for a, b in zip(control_extra, treatment_extra)]
    random.Random(seed + 2).shuffle(pairs)
    control = []; treatment = []
    for i, (a, b, origin) in enumerate(pairs):
        for record, condition in [(a, 'real_replay'), (b, 'synthetic_curriculum')]:
            record['frozen_slot'] = i; record['schedule_part'] = origin; record['condition'] = condition
        control.append(a); treatment.append(b)
    def summary(rows):
        return {'records': len(rows), 'target_tokens': sum(target_cost(r, tokenizer) for r in rows),
                'target_by_kind': {k: sum(target_cost(r, tokenizer) for r in rows if r['kind'] == k) for k in ['block', 'answer']},
                'source_ids': len({(r['dataset'], r['id']) for r in rows}),
                'strata': [{'dataset': k[0], 'kind': k[1], 'label': k[2], 'records': n}
                           for k, n in sorted(Counter(stratum(r) for r in rows).items())],
                'records_sha256': digest(rows)}
    a, b = summary(control), summary(treatment)
    if a['records'] != b['records'] or a['target_by_kind'] != b['target_by_kind']:
        raise ValueError('Matched schedule invariants violated')
    report = {'control': a, 'treatment': b, 'common_balanced_records': len(common),
              'augmentation_records': len(treatment_extra), 'effective_batch': multiple,
              'matched_target_tokens_and_record_count_and_kind': True,
              'matching_level': 'kind_only' if failures else 'task_kind_label',
              'stratum_matching_failures': failures, 'all_common_raw_records_preserved': True,
              'all_synthetic_raw_records_preserved': True,
              'equal_input_tokens_or_gpu_compute_claimed': False,
              'control_selection': 'length-matched real replay with replacement, not uniform real sampling'}
    return control, treatment, report
