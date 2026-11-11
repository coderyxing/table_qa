# 脚本用途：按与答案正确性无关的规则准备各任务的新训练开发样本。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Prepare 200 new source-training examples/task; never selects by correctness."""
import json
from pathlib import Path
import random
from tableqa.data import read_jsonl,write_jsonl
from tableqa.inference import PROMPT_VERSION
if PROMPT_VERSION!='tableqa-v5': raise ValueError('Freeze/review the prompt version before preparing this stage')
root=Path('runs/medium-development'); root.mkdir(parents=True,exist_ok=True)
excluded=set()
for path in ['runs/teacher-pilot/samples.jsonl','runs/teacher-pilot-fresh30/samples.jsonl']:
    # 读取结构化输入：path，保留原有样本身份字段。
    excluded.update((r['dataset'],r['id']) for r in read_jsonl(path))
# 读取结构化输入：'data/normalized/joint_train_clean.jsonl'，保留原有样本身份字段。
source=read_jsonl('data/normalized/joint_train_clean.jsonl'); selected=[]
for dataset in ['wikitq','tabfact','fetaqa']:
    pool=sorted([r for r in source if r['dataset']==dataset and (r['dataset'],r['id']) not in excluded],key=lambda r:r['id'])
    random.Random(2026).shuffle(pool)
    if len(pool)<200: raise ValueError('Insufficient source training examples')
    selected+=pool[:200]
# 保存逐条结构化产物到 root / 'samples.jsonl'，供后续重放或审核读取。
write_jsonl(root/'samples.jsonl',selected)
for mode in ['A4','A0']:
    config={'mode':mode,'model_config':{'backend':'local','model':str(Path('models/Qwen2.5-32B-Instruct').resolve()),
            'device_map':'balanced','seed':2026,'max_input_tokens':4096,'max_new_tokens':1024},
            'settings':{'max_repairs':1 if mode=='A4' else 0}}
    (root/f'{mode}.json').write_text(json.dumps(config,indent=2))
(root/'protocol.json').write_text(json.dumps({'development_only':True,'n_per_dataset':200,'seed':2026,
    'split':'train','excludes_prior_development_examples':True,'no_length_or_label_based_selection':True,
    'student_training_tasks':'Only quality-accepted tasks; freeform requires human review.'},indent=2))
print('MEDIUM_STAGE_PREPARED',len(selected),flush=True)
