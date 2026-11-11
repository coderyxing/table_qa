"""Development-only B2/B5 rerun retaining every original training record."""
import gc
import json
from pathlib import Path
import subprocess
import sys
import torch
from transformers import AutoTokenizer
from tableqa.cli import metadata
from tableqa.data import read_jsonl,write_jsonl,digest
from tableqa.analysis import index_records
from tableqa.training import train,prepare_training_data
from tableqa.models import LocalModel
from tableqa.inference import predict
from tableqa.experiments import settings

source=Path('runs/medium-development/students');root=Path('runs/coverage-student-control/results')
if root.exists():raise ValueError('Output exists; preserve it and inspect before rerunning')
original=read_jsonl(source/'accepted-samples.jsonl');keys=set(index_records(original,'source'))
validation=read_jsonl(source/'validation-samples.jsonl')
if any(s['split']!='validation' for s in validation):raise ValueError('Only the existing source validation protocol is allowed')
cfg=json.loads((source/'B2/run.json').read_text())['config']
if cfg!=json.loads((source/'B5/run.json').read_text())['config']:raise ValueError('Legacy configs differ')
cfg={**cfg,'sampling_policy':'coverage'}
tokenizer=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True)
prepared={};records={}
for mode in ['B2','B5']:
    records[mode]=read_jsonl(source/f'{mode}-records.jsonl')
    if {tuple((r['dataset'],r['id'])) for r in records[mode]}!=keys:raise ValueError('Raw source pools differ')
    encoded,stats=prepare_training_data(cfg,records[mode],tokenizer)
    if {r['source_key'] for r in encoded}!=keys:raise ValueError('Coverage policy lost a source sample')
    prepared[mode]=stats
root.mkdir(parents=True)
write_jsonl(root/'validation-samples.jsonl',validation)
(root/'preflight.json').write_text(json.dumps(prepared,indent=2))
(root/'protocol.json').write_text(json.dumps({'development_only':True,'change':'coverage-preserving kind balancing',
    'n_original_samples':len(keys),'source_ids_sha256':digest(sorted(keys)),
    'control':'same complete source ID coverage, same seed/rank/epochs; NOT equal target-token budgets',
    'legacy_results':'retained; raw-pool-only controls are not strict equal-sample comparisons',
    'reuse_B0':'Single-kind B0 sampler sequence is unchanged by the coverage policy; reuse the answer control.'},indent=2))
report={'development_only':True,'models':{}}
for mode in ['B2','B5']:
    directory=root/mode;directory.mkdir()
    (directory/'run.json').write_text(json.dumps(metadata(cfg),indent=2))
    torch.cuda.reset_peak_memory_stats();statistics=train(cfg,records[mode],directory/'adapter')
    if statistics['source_samples_after_alignment']!=len(keys):raise ValueError('Actual source coverage differs')
    gc.collect();torch.cuda.empty_cache()
    backend=LocalModel(cfg['model'],adapter=str(directory/'adapter'),max_new_tokens=512,max_input_tokens=4096,seed=42)
    predictions=[]
    for sample in validation:
        predictions.append(predict(sample,backend,settings(mode)))
        write_jsonl(directory/'validation-predictions.jsonl',predictions)
    report['models'][mode]={'training':statistics}
    (root/'report.json').write_text(json.dumps(report,indent=2))
    del backend;gc.collect();torch.cuda.empty_cache()
scripts=Path(__file__).resolve().parent
paths={'B0':Path('runs/student-answer-control/results/trained-B0-predictions.jsonl'),
       'B2':root/'B2/validation-predictions.jsonl','B5':root/'B5/validation-predictions.jsonl'}
for name,path in paths.items():
    subprocess.run([sys.executable,str(scripts/'evaluate_wtq_official.py'),str(root/'validation-samples.jsonl'),str(path),
        str(root/f'wtq-{name}'),'--evaluator','data/raw/wikitq-official/WikiTableQuestions-master/evaluator.py',
        '--tagged','data/raw/wikitq-official/WikiTableQuestions-master/tagged/data'],check=True)
for a,b in [('B0','B2'),('B0','B5'),('B2','B5')]:
    subprocess.run([sys.executable,str(scripts/'analyze_runs.py'),str(root/'validation-samples.jsonl'),str(paths[a]),
        str(root/f'comparison-{a}-vs-{b}'),'--predictions-b',str(paths[b]),'--official-a',str(root/f'wtq-{a}/scores.json'),
        '--official-b',str(root/f'wtq-{b}/scores.json')],check=True)
print('COVERAGE_CONTROL_COMPLETE',flush=True)
