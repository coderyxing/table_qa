"""One frozen accuracy condition, with resumable atomic outputs and no reference routing."""
import argparse
import json
import os
from pathlib import Path
import torch
from tableqa.cli import metadata
from tableqa.data import read_jsonl,write_jsonl,digest
from tableqa.analysis import index_records
from tableqa.models import make_model
from tableqa.inference import predict
from tableqa.experiments import settings
from tableqa.recovery import predict_with_recovery

p=argparse.ArgumentParser(description=__doc__);p.add_argument('condition',choices=['A','B','C']);args=p.parse_args()
root=Path('runs/accuracy-development-v1');name=args.condition
cfg=json.loads((root/f'{name}.json').read_text());samples=read_jsonl(root/'samples.jsonl')
if any(s['split']!='validation' for s in samples):raise ValueError('Validation tuning only')
run=metadata(cfg);identity=digest({'source':run['source_hash'],'config':cfg,'samples':samples})
meta=root/f'{name}-run.json'
if meta.exists() and json.loads(meta.read_text())['identity']!=identity:raise ValueError('Cannot resume changed accuracy condition')
meta.write_text(json.dumps({**run,'identity':identity,'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES')},indent=2))
canonical=root/(f'{name}-pairs.jsonl' if name=='C' else f'{name}-predictions.jsonl')
records=read_jsonl(canonical) if canonical.exists() else []
previous=[r['primary'] for r in records] if name=='C' else records
done=set(index_records(previous,'previous prediction'));expected={(r['dataset'],r['id']) for r in samples}
if done-expected:raise ValueError('Unknown prior accuracy outputs')
model=make_model(cfg['model_config']);backup=make_model(cfg['fallback_model_config']) if name=='C' else None
options=settings('B5',**cfg['settings']);torch.cuda.reset_peak_memory_stats()

def atomic_write(path,rows):
    temporary=path.with_suffix('.tmp');write_jsonl(temporary,rows);temporary.replace(path)

for sample in samples:
    if (sample['dataset'],sample['id']) in done:continue
    if name=='C':
        primary,routed=predict_with_recovery(sample,model,backup,options)
        primary['metrics']['latency_comparable']=False
        result={'primary':primary,'routed':routed}
    else:
        result=predict(sample,model,options);result['metrics']['latency_comparable']=False
    records.append(result);atomic_write(canonical,records)
    (root/f'{name}-progress.json').write_text(json.dumps({'completed':len(records),'expected':len(samples),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'condition':name}))
    print('ACCURACY_PROGRESS',name,len(records),len(samples),flush=True)
if name=='C':
    atomic_write(root/'C-predictions.jsonl',[r['primary'] for r in records])
    atomic_write(root/'D-predictions.jsonl',[r['routed'] for r in records])
print('ACCURACY_CONDITION_COMPLETE',name,flush=True)
