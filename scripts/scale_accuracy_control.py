"""7B capacity control on the identical 149 sources and frozen tuning cohort."""
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import torch
from transformers import AutoTokenizer
from tableqa.cli import metadata
from tableqa.data import read_jsonl,write_jsonl,digest
from tableqa.training import prepare_training_data,train
from tableqa.models import LocalModel,make_model
from tableqa.experiments import settings
from tableqa.inference import predict
from tableqa.recovery import predict_with_recovery

root=Path('runs/scale-7b-accuracy/results');source=Path('runs/coverage-student-control/results')
if root.exists():raise ValueError('Output exists; preserve prior results before rerunning')
model_path=Path('models/Qwen2.5-7B-Instruct').resolve()
if not (model_path/'DOWNLOAD_COMPLETE').exists():raise ValueError('Verified weights required')
if shutil.disk_usage('.').free<8*1024**3:raise ValueError('Keep at least 8 GiB free before training')
cfg=json.loads((source/'B5/run.json').read_text())['config'];cfg={**cfg,'model':str(model_path)}
records=read_jsonl('runs/medium-development/students/B5-records.jsonl')
samples=read_jsonl('runs/accuracy-development-v1/samples.jsonl')
if any(s['split']!='validation' for s in samples):raise ValueError('Only frozen validation tuning samples allowed')
context_limit=json.loads((model_path/'config.json').read_text())['max_position_embeddings']
if context_limit<8192+512:raise ValueError('Actual 7B context configuration is too small')
tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
_,preflight=prepare_training_data(cfg,records,tokenizer)
old=json.loads((source/'report.json').read_text())['models']['B5']['training']
if preflight['retained_source_ids_sha256']!=old['retained_source_ids_sha256']:
    raise ValueError('7B actual source coverage differs from 3B control')
root.mkdir(parents=True);write_jsonl(root/'samples.jsonl',samples)
(root/'preflight.json').write_text(json.dumps(preflight,indent=2))
(root/'run.json').write_text(json.dumps(metadata(cfg),indent=2))
(root/'protocol.json').write_text(json.dumps({'development_only':True,'factor':'3B vs 7B student capacity',
    'same_raw_record_sha256':digest(records),'actual_source_ids_sha256':preflight['retained_source_ids_sha256'],
    'E':'7B input4096 repairs1, compared with 3B A','F':'7B input8192 repairs2, compared with 3B C',
    'G':'F plus same 3B B0 fallback as 3B D','native_tokenizer_audited':True,
    'not_equal_parameter_or_compute_budget':True,'confirmation_not_evaluated':True,
    'timing':'Accuracy control only; no matched latency benchmark or algorithm-only attribution'},indent=2))
torch.cuda.reset_peak_memory_stats();statistics=train(cfg,records,root/'adapter')
if statistics['retained_source_ids_sha256']!=old['retained_source_ids_sha256']:raise ValueError('Actual training coverage changed')
(root/'training-statistics.json').write_text(json.dumps(statistics,indent=2))
gc.collect();torch.cuda.empty_cache()
primary=LocalModel(str(model_path),adapter=str(root/'adapter'),max_new_tokens=512,max_input_tokens=4096,seed=42)
outputs={'E':[],'F':[],'G':[]}
for sample in samples:
    pred=predict(sample,primary,settings('B5',max_repairs=1));pred['metrics']['latency_comparable']=False
    outputs['E'].append(pred);write_jsonl(root/'E-predictions.jsonl',outputs['E'])
    (root/'progress.json').write_text(json.dumps({'condition':'E','completed':len(outputs['E']),'expected':len(samples)}))
primary.max_input_tokens=8192
fallback_config=json.loads(Path('runs/accuracy-development-v1/C.json').read_text())['fallback_model_config']
backup=make_model(fallback_config)
for sample in samples:
    pred,routed=predict_with_recovery(sample,primary,backup,settings('B5',max_repairs=2))
    pred['metrics']['latency_comparable']=False
    outputs['F'].append(pred);outputs['G'].append(routed)
    write_jsonl(root/'F-predictions.jsonl',outputs['F']);write_jsonl(root/'G-predictions.jsonl',outputs['G'])
    (root/'progress.json').write_text(json.dumps({'condition':'F/G','completed':len(outputs['F']),'expected':len(samples)}))
del primary,backup;gc.collect();torch.cuda.empty_cache()
scripts=Path(__file__).resolve().parent;base=Path('runs/accuracy-development-v1')
for name in ['E','F','G']:
    subprocess.run([sys.executable,str(scripts/'evaluate_wtq_official.py'),str(root/'samples.jsonl'),str(root/f'{name}-predictions.jsonl'),
        str(root/f'wtq-{name}'),'--evaluator','data/raw/wikitq-official/WikiTableQuestions-master/evaluator.py',
        '--tagged','data/raw/wikitq-official/WikiTableQuestions-master/tagged/data'],check=True)
for a,b in [('A','E'),('C','F'),('D','G')]:
    subprocess.run([sys.executable,str(scripts/'analyze_runs.py'),str(root/'samples.jsonl'),str(base/f'{a}-predictions.jsonl'),
        str(root/f'comparison-{a}-vs-{b}'),'--predictions-b',str(root/f'{b}-predictions.jsonl'),
        '--official-a',str(base/f'wtq-{a}/scores.json'),'--official-b',str(root/f'wtq-{b}/scores.json')],check=True)
print('SCALE_7B_ACCURACY_COMPLETE',flush=True)
