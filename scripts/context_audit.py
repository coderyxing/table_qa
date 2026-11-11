"""Training-only input-length audit; no model weights or reference in prompts."""
import json
from pathlib import Path
import random
from transformers import AutoTokenizer
from tableqa.data import read_jsonl
from tableqa.core import TableState
from tableqa.inference import SYSTEM,context
from tableqa.evaluation import quantile
tokenizer=AutoTokenizer.from_pretrained('models/Qwen2.5-3B-Instruct',local_files_only=True)
records=read_jsonl('data/normalized/joint_train_clean.jsonl'); report={}
for dataset in ['wikitq','tabfact','fetaqa']:
    candidates=sorted([r for r in records if r['dataset']==dataset],key=lambda r:r['id'])
    random.Random(42).shuffle(candidates); candidates=candidates[:200]; lengths=[]
    for r in candidates:
        messages=[{'role':'system','content':SYSTEM},{'role':'user','content':context(r,TableState.from_table(r['table']),[],mode='A4')}]
        lengths.append(len(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)))
    report[dataset]={'n':len(lengths),'p50':quantile(lengths,.5),'p95':quantile(lengths,.95),'max':max(lengths),
                     'over_limit':{str(n):sum(v>n for v in lengths) for n in [4096,8192,16384]}}
Path('runs/setup/context-audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
