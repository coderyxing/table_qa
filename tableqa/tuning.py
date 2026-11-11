"""Validation-only cohorts, disjoint by whole-table content and selected without labels."""
import random
from .data import table_fingerprint


def accuracy_cohorts(records,previous,per_task=100,seed=2030):
    if type(per_task) is not int or per_task<1:raise ValueError('Positive cohort size required')
    if any(r['split']!='validation' for r in records):raise ValueError('Accuracy tuning must use validation only')
    excluded={table_fingerprint(r['table']) for r in previous}
    pools={}
    for d in ['wikitq','tabfact']:
        grouped={}
        for r in records:
            if r['dataset']!=d:continue
            fp=table_fingerprint(r['table'])
            if fp not in excluded:grouped.setdefault(fp,[]).append(r)
        keys=sorted(grouped);rng=random.Random(seed);rng.shuffle(keys)
        pools[d]=[]
        for fp in keys:
            candidates=sorted(grouped[fp],key=lambda r:r['id'])
            pools[d].append((fp,candidates[rng.randrange(len(candidates))]))
    used=set(excluded);out={}
    for name in ['tune','confirmation']:
        chosen={}
        for d in ['wikitq','tabfact']:
            selected=[]
            for fp,r in pools[d]:
                if fp in used:continue
                selected.append(r);used.add(fp)
                if len(selected)==per_task:break
            if len(selected)!=per_task:raise ValueError(f'Insufficient disjoint validation tables for {d}/{name}: {len(selected)}')
            chosen[d]=selected
        out[name]=[chosen[d][i] for i in range(per_task) for d in ['wikitq','tabfact']]
    return out
