"""Six source-training examples to inspect 32B executable trajectory quality."""
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
parser=argparse.ArgumentParser()
parser.add_argument('--model',default='models/Qwen2.5-32B-Instruct')
parser.add_argument('--output',default='runs/teacher-pilot')
parser.add_argument('--per-dataset',type=int,default=2)
parser.add_argument('--max-input-tokens',type=int,default=4096)
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--exclude-samples')
parser.add_argument('--samples-file')
parser.add_argument('--mode',default='A4',choices=['A0','A1','A2','A3','A4'])
args=parser.parse_args()
out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
if (out/'predictions.jsonl').exists(): raise ValueError('Pilot output exists; choose a fresh output directory')
data=read_jsonl('data/normalized/joint_train_clean.jsonl'); selected=[]
excluded={(r['dataset'],r['id']) for r in read_jsonl(args.exclude_samples)} if args.exclude_samples else set()
for dataset in ['wikitq','tabfact','fetaqa']:
    candidates=sorted([r for r in data if r['dataset']==dataset and len(r['table']['rows'])<=30 and (r['dataset'],r['id']) not in excluded],key=lambda r:r['id'])
    random.Random(args.seed).shuffle(candidates); selected+=candidates[:args.per_dataset]
write_jsonl(out/'samples.jsonl',selected)
if args.samples_file:
    selected=read_jsonl(args.samples_file)
    if any(r['split']!='train' for r in selected): raise ValueError('This pilot uses source training samples only')
    write_jsonl(out/'samples.jsonl',selected)
config={'model':str(Path(args.model).resolve()),'max_new_tokens':1024,'max_input_tokens':args.max_input_tokens,'device_map':'balanced','seed':args.seed,'mode':args.mode}
(out/'run.json').write_text(json.dumps(metadata(config),indent=2))
model=LocalModel(config['model'],max_new_tokens=1024,max_input_tokens=args.max_input_tokens,device_map='balanced')
predictions=[]
for r in selected:
    predictions.append(predict(r,model,settings(args.mode)))
    write_jsonl(out/'predictions.jsonl',predictions)
report={'pilot_only':True,'split':'train','metrics':evaluate(predictions,selected)}
(out/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
