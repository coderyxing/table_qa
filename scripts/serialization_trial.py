"""Small, preregistered training-only format trial after the main batch finishes."""
import json
from pathlib import Path
import random
import tableqa
from tableqa.cli import metadata
from tableqa.data import read_jsonl,write_jsonl,digest
from tableqa.models import LocalModel
from tableqa.serialization import ArrayInputModel
from tableqa.inference import predict
from tableqa.experiments import settings

root=Path('runs/serialization-followup'); root.mkdir(parents=True,exist_ok=True)
implementation=Path(tableqa.__file__).parent
baseline_code=Path('runs/medium-development/frozen-code/tableqa')
for name in ['core.py','inference.py','models.py','experiments.py']:
    if (implementation/name).read_bytes()!=(baseline_code/name).read_bytes():
        raise ValueError(f'Baseline/candidate implementation differs beyond serialization: {name}')
source=read_jsonl('runs/medium-development/samples.jsonl'); selected=[]
for dataset in ['wikitq','tabfact','fetaqa']:
    pool=sorted([r for r in source if r['dataset']==dataset],key=lambda r:r['id'])
    random.Random(2027).shuffle(pool); selected+=pool[:10]
keys={(r['dataset'],r['id']) for r in selected}
baseline=[r for r in read_jsonl('runs/medium-development/A4-predictions.jsonl') if (r['dataset'],r['id']) in keys]
if len(baseline)!=len(selected): raise ValueError('Prerequisite baseline coverage incomplete')
write_jsonl(root/'samples.jsonl',selected); write_jsonl(root/'dictionary-predictions.jsonl',baseline)
config={'model':str(Path('models/Qwen2.5-32B-Instruct').resolve()),'device_map':'balanced',
        'max_input_tokens':4096,'max_new_tokens':1024,'seed':2026}
(root/'protocol.json').write_text(json.dumps({'development_only':True,'selection_seed':2027,
    'selection_uses_labels':False,'n_per_dataset':10,'source_sha256':digest(source),
    'candidate':'lossless array state serialization','baseline':'frozen medium-development A4',
    'latency_scope':'Observed across separate runs, not a matched warm-up timing benchmark'},indent=2))
(root/'run.json').write_text(json.dumps(metadata(config),indent=2))
backend=ArrayInputModel(LocalModel(config['model'],device_map='balanced',seed=2026))
predictions=[]
for sample in selected:
    p=predict(sample,backend,settings('A4')); p['input_representation']='array'
    predictions.append(p); write_jsonl(root/'array-predictions.jsonl',predictions)
print('SERIALIZATION_TRIAL_COMPLETE',len(predictions),flush=True)
