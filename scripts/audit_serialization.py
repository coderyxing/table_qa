"""Training-only token audit. This does NOT change live prompts or claim QA gains."""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer
from tableqa.data import read_jsonl,write_jsonl,digest
from tableqa.core import TableState
from tableqa.inference import context,SYSTEM
from tableqa.serialization import array_view,restore_view
from tableqa.evaluation import quantile
parser=argparse.ArgumentParser(); parser.add_argument('samples'); parser.add_argument('output'); args=parser.parse_args()
data=read_jsonl(args.samples)
if any(r['split']!='train' for r in data): raise ValueError('Design audit must not use validation/test inputs')
tokenizer=AutoTokenizer.from_pretrained('models/Qwen2.5-3B-Instruct',local_files_only=True)
rows=[]
for sample in data:
    payload=json.loads(context(sample,TableState.from_table(sample['table']),[],mode='A4'))
    original=payload['state']; alternative=array_view(original)
    if restore_view(alternative)!=original: raise ValueError('Lossless round trip failed')
    counts={}
    for name,view in [('dictionary',original),('array',alternative)]:
        payload['state']=view
        messages=[{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}]
        counts[name]=len(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))
    rows.append({'dataset':sample['dataset'],'id':sample['id'],'tokens':counts,'round_trip_equal':True})
root=Path(args.output); root.mkdir(parents=True,exist_ok=True); write_jsonl(root/'per-example.jsonl',rows)
report={'candidate_only':True,'qa_quality_not_tested':True,'samples_sha256':digest(data),'n':len(rows),'datasets':{}}
for dataset in sorted({r['dataset'] for r in rows}):
    selected=[r for r in rows if r['dataset']==dataset]; d={}
    for name in ['dictionary','array']:
        values=[r['tokens'][name] for r in selected]
        d[name]={'p50':quantile(values,.5),'p95':quantile(values,.95),
                 'over_4096':sum(v>4096 for v in values),'over_8192':sum(v>8192 for v in values),'total':sum(values)}
    d['token_reduction']=1-d['array']['total']/d['dictionary']['total']; report['datasets'][dataset]=d
(root/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
