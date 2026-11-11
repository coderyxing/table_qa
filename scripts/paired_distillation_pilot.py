"""B2/B5 engineering pilot from the SAME replay-checked source sample IDs.

One epoch and equal original examples, NOT equal token budgets or thesis results.
Freeform trajectories remain pending human review and are not auto-approved.
"""
import gc
import json
from pathlib import Path
import random
import argparse
import torch
from transformers import AutoTokenizer
from tableqa.cli import metadata
from tableqa.data import read_jsonl,write_jsonl
from tableqa.distillation import build_records
from tableqa.quality import audit_trajectory
from tableqa.training import encode_record,train
from tableqa.models import LocalModel
from tableqa.inference import predict
from tableqa.experiments import settings
from tableqa.evaluation import evaluate

parser=argparse.ArgumentParser(); parser.add_argument('--output',default='runs/paired-distillation-pilot')
parser.add_argument('--source',default='runs/teacher-pilot-fresh30')
parser.add_argument('--epochs',type=int,default=1); parser.add_argument('--rank',type=int,default=8)
parser.add_argument('--eval-per-dataset',type=int,default=3); args=parser.parse_args()
source=Path(args.source); root=Path(args.output)
if not (source/'report.json').exists(): raise RuntimeError('Fresh teacher run has not completed')
if root.exists(): raise ValueError('Output directory exists; inspect it before rerunning')
root.mkdir()
samples={(r['dataset'],r['id']):r for r in read_jsonl(source/'samples.jsonl')}
model_path=str(Path('models/Qwen2.5-3B-Instruct').resolve())
tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
training={'B2':[],'B5':[]}; accepted=[]; audits=[]
for prediction in read_jsonl(source/'predictions.jsonl'):
    sample=samples[(prediction['dataset'],prediction['id'])]
    result=audit_trajectory(sample,prediction)
    audit={'dataset':sample['dataset'],'id':sample['id'],**result}; audits.append(audit)
    if not result['decision'].startswith('accepted_'): continue
    try:
        records={mode:build_records(sample,prediction['operations'],mode) for mode in training}
        for rows in records.values():
            for row in rows: encode_record(row,tokenizer,4096)
    except ValueError as exc:
        audit['decision']='rejected'; audit['reasons']=['paired_training_context_limit']; continue
    accepted.append({'dataset':sample['dataset'],'id':sample['id']})
    for mode in training: training[mode]+=records[mode]
write_jsonl(root/'audit.jsonl',audits); write_jsonl(root/'accepted-samples.jsonl',accepted)
if len(accepted)<2: raise RuntimeError('Too few valid original samples; no model is trained')
for mode,records in training.items():write_jsonl(root/f'{mode}-records.jsonl',records)
# Draw source validation samples, preserve failures and never inspect official test answers.
validation=read_jsonl('data/normalized/joint_validation_clean.jsonl'); evaluation=[]
for dataset in sorted({r['dataset'] for r in accepted}):
    candidates=sorted([r for r in validation if r['dataset']==dataset],key=lambda r:r['id'])
    random.Random(123).shuffle(candidates); evaluation+=candidates[:args.eval_per_dataset]
write_jsonl(root/'validation-samples.jsonl',evaluation)
report={'pilot_only':True,'budget':f'equal_original_examples_{args.epochs}_epochs_NOT_equal_tokens',
        'accepted_original_samples':len(accepted),'datasets_covered':sorted({r['dataset'] for r in accepted}),
        'freeform_auto_approved':False,'models':{}}
for mode in training:
    config={'model':model_path,'seed':42,'max_seq_length':4096,'lora_rank':args.rank,'lora_alpha':2*args.rank,
            'learning_rate':2e-4,'answer_weight':1,'micro_batch':1,'gradient_accumulation_steps':4,'max_epochs':args.epochs}
    directory=root/mode; directory.mkdir()
    (directory/'run.json').write_text(json.dumps(metadata(config),indent=2))
    torch.cuda.reset_peak_memory_stats()
    statistics=train(config,training[mode],directory/'adapter')
    gc.collect(); torch.cuda.empty_cache()
    backend=LocalModel(model_path,adapter=str(directory/'adapter'),max_new_tokens=512)
    predictions=[]
    for sample in evaluation:
        predictions.append(predict(sample,backend,settings(mode)))
        write_jsonl(directory/'validation-predictions.jsonl',predictions)
    report['models'][mode]={'training':statistics,'validation':evaluate(predictions,evaluation)}
    del backend; gc.collect(); torch.cuda.empty_cache()
    (root/'report.json').write_text(json.dumps(report,indent=2))
print('PAIRED_PILOT_COMPLETE',json.dumps(report),flush=True)
