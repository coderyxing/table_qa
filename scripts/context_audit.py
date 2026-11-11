# 脚本用途：只加载 tokenizer，核算训练输入的长度分布和超限情况。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Training-only input-length audit; no model weights or reference in prompts."""
import json
from pathlib import Path
import random
from transformers import AutoTokenizer
from tableqa.data import read_jsonl
from tableqa.core import TableState
from tableqa.inference import SYSTEM,context
from tableqa.evaluation import quantile
# 使用本阶段指定的 tokenizer；预算、提示模板与监督掩码应沿用同一版本。
tokenizer=AutoTokenizer.from_pretrained('models/Qwen2.5-3B-Instruct',local_files_only=True)
# 读取结构化输入：'data/normalized/joint_train_clean.jsonl'，保留原有样本身份字段。
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
