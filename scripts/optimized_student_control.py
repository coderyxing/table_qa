"""One-factor compression control: search vs legacy, both fixed-block B5."""
import gc
import json
from pathlib import Path
import subprocess
import sys
import torch
from transformers import AutoTokenizer
from tableqa.analysis import index_records
from tableqa.cli import metadata
from tableqa.data import read_jsonl,digest,write_jsonl
from tableqa.training import prepare_training_data,train
from tableqa.models import LocalModel
from tableqa.experiments import settings
from tableqa.inference import predict

root=Path('runs/optimized-student-control/results')
if root.exists():raise ValueError('Output exists; preserve it before rerunning')
audit=Path('runs/optimization-audit');control=Path('runs/coverage-student-control/results')
summary=json.loads((audit/'report.json').read_text())
records=read_jsonl(audit/'search_fixed-records.jsonl')
if digest(records)!=summary['variants']['search_fixed']['record_sha256']:raise ValueError('Audited records changed')
if summary['variants']['search_fixed']['record_sha256']==summary['variants']['legacy_fixed']['record_sha256']:
    raise ValueError('Candidate is identical; reuse the existing control rather than retraining')
cfg=json.loads((control/'B5/run.json').read_text())['config']
validation=read_jsonl(control/'validation-samples.jsonl')
if any(r['split']!='validation' for r in validation):raise ValueError('Source validation only')
tokenizer=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True)
_,prepared=prepare_training_data(cfg,records,tokenizer)
previous=json.loads((control/'report.json').read_text())['models']['B5']['training']
if prepared['retained_source_ids_sha256']!=previous['retained_source_ids_sha256']:
    raise ValueError('Actual training source IDs differ from the corrected control')
root.mkdir(parents=True);write_jsonl(root/'records.jsonl',records);write_jsonl(root/'validation-samples.jsonl',validation)
(root/'preflight.json').write_text(json.dumps(prepared,indent=2))
(root/'run.json').write_text(json.dumps(metadata(cfg),indent=2))
(root/'protocol.json').write_text(json.dumps({'development_only':True,'factor':'legacy vs replay-subsequence compression',
    'partition':'fixed max4 with separate answer supervision in both conditions',
    'training_control':'same actual source IDs, seed, rank and epochs; NOT equal target-token budgets',
    'baseline_records_sha256':summary['variants']['legacy_fixed']['record_sha256'],
    'candidate_records_sha256':digest(records),'n_original_samples':prepared['source_samples_after_alignment'],
    'state_dp_not_tested_by_this_training':True,
    'timing':'Separate runs, no matched warm-up; observed ratios are not formal acceleration results'},indent=2))
torch.cuda.reset_peak_memory_stats();statistics=train(cfg,records,root/'adapter')
if statistics['retained_source_ids_sha256']!=previous['retained_source_ids_sha256']:raise ValueError('Training lost source coverage')
(root/'training-statistics.json').write_text(json.dumps(statistics,indent=2))
gc.collect();torch.cuda.empty_cache()
model=LocalModel(cfg['model'],adapter=str(root/'adapter'),max_new_tokens=512,max_input_tokens=4096,seed=42)
predictions=[]
for sample in validation:
    predictions.append(predict(sample,model,settings('B5')))
    write_jsonl(root/'predictions.jsonl',predictions)
del model;gc.collect();torch.cuda.empty_cache()
scripts=Path(__file__).resolve().parent
paths={'legacy':control/'B5/validation-predictions.jsonl','search':root/'predictions.jsonl'}
for name,path in paths.items():
    subprocess.run([sys.executable,str(scripts/'evaluate_wtq_official.py'),str(root/'validation-samples.jsonl'),str(path),
        str(root/f'wtq-{name}'),'--evaluator','data/raw/wikitq-official/WikiTableQuestions-master/evaluator.py',
        '--tagged','data/raw/wikitq-official/WikiTableQuestions-master/tagged/data'],check=True)
subprocess.run([sys.executable,str(scripts/'analyze_runs.py'),str(root/'validation-samples.jsonl'),str(paths['legacy']),
    str(root/'comparison'),'--predictions-b',str(paths['search']),'--official-a',str(root/'wtq-legacy/scores.json'),
    '--official-b',str(root/'wtq-search/scores.json')],check=True)
print('OPTIMIZED_STUDENT_CONTROL_COMPLETE',flush=True)
