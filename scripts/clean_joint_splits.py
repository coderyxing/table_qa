"""Protect test/validation tables across task boundaries; preserve official files."""
import json
import random
from pathlib import Path
from tableqa.data import read_jsonl,write_jsonl,table_fingerprint
files=sorted(Path('data/normalized').glob('*_*.jsonl'))
records=[]; fingerprints={}
for path in files:
    if 'pilot' in path.name or path.name.startswith('joint_'): continue
    records+=read_jsonl(path)
tabfact_train=sorted([r for r in records if r['dataset']=='tabfact' and r['split']=='train'],key=lambda r:r['id'])
random.Random(42).shuffle(tabfact_train)
selected_tabfact_ids={r['id'] for r in tabfact_train[:10000]}
records=[r for r in records if not (r['dataset']=='tabfact' and r['split']=='train') or r['id'] in selected_tabfact_ids]
def fp(r):
    key=(r['dataset'],r['table_id'])
    if key not in fingerprints: fingerprints[key]=table_fingerprint(r['table'])
    return fingerprints[key]
test={fp(r) for r in records if r['split']=='test'}
valid={fp(r) for r in records if r['split']=='validation'}
kept={'train':[],'validation':[],'test':[]}; removed=[]
for r in records:
    excluded=(r['split']=='train' and fp(r) in test|valid) or (r['split']=='validation' and fp(r) in test)
    if excluded: removed.append({'dataset':r['dataset'],'id':r['id'],'split':r['split']})
    else: kept[r['split']].append(r)
for split,rows in kept.items():write_jsonl(f'data/normalized/joint_{split}_clean.jsonl',rows)
report={'counts':{s:len(v) for s,v in kept.items()},'removed':removed,
        'tabfact_train_cap':10000,'tabfact_selection_seed':42,
        'policy':'Official input files preserved; test > validation > training at table level.'}
Path('runs/setup/joint-split-cleaning.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'counts':report['counts'],'removed':len(removed)}),flush=True)
