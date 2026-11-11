"""Validation-split smoke evaluation of the tiny answer-supervised adapter."""
import json
from pathlib import Path
import random
import argparse
from tableqa.cli import metadata
from tableqa.data import read_jsonl,write_jsonl
from tableqa.models import LocalModel
from tableqa.inference import predict
from tableqa.experiments import settings
from tableqa.evaluation import evaluate

parser=argparse.ArgumentParser(); parser.add_argument('--directory',default='runs/supervised-pilot'); args=parser.parse_args()
root=Path(args.directory)
if not (root/'statistics.json').exists(): raise RuntimeError('Training has not completed')
if (root/'validation-predictions.jsonl').exists(): raise ValueError('Validation output exists')
data=read_jsonl('data/normalized/joint_validation_clean.jsonl'); samples=[]
for dataset in ['wikitq','tabfact','fetaqa']:
    pool=sorted([r for r in data if r['dataset']==dataset],key=lambda r:r['id'])
    random.Random(42).shuffle(pool); samples+=pool[:3]
write_jsonl(root/'validation-samples.jsonl',samples)
model_path=str(Path('models/Qwen2.5-3B-Instruct').resolve())
config={'model':model_path,'adapter':str(root/'adapter'),'mode':'B0','split':'validation','seed':42}
(root/'validation-run.json').write_text(json.dumps(metadata(config),indent=2))
model=LocalModel(model_path,adapter=str(root/'adapter'),max_new_tokens=256)
predictions=[]
for r in samples:
    predictions.append(predict(r,model,settings('B0')))
    write_jsonl(root/'validation-predictions.jsonl',predictions)
report={'pilot_only':True,'n_training_samples':24,'n_validation_samples':len(samples),
        'metrics':evaluate(predictions,samples),'official_test_evaluation':False}
(root/'validation-report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
