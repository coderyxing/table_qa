# 脚本用途：在两组教师推理前冻结相同的新训练题队列。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Freeze a fresh training-only cohort before either teacher condition runs."""
import json
from pathlib import Path
import random
from tableqa.data import read_jsonl,write_jsonl,digest
root=Path('runs/interactive-teacher');root.mkdir(parents=True,exist_ok=True)
if (root/'protocol.json').exists():raise ValueError('Protocol exists; reuse it with the frozen runner')
excluded=set();sources=[]
for path in sorted(Path('runs').glob('**/samples.jsonl')):
    if root in path.parents:continue
    # 读取结构化输入：path，保留原有样本身份字段。
    rows=read_jsonl(path);excluded.update((r['dataset'],r['id']) for r in rows)
    sources.append(str(path))
# 读取结构化输入：'data/normalized/joint_train_clean.jsonl'，保留原有样本身份字段。
source=read_jsonl('data/normalized/joint_train_clean.jsonl');pools={}
for dataset in ['wikitq','tabfact','fetaqa']:
    candidates=sorted([r for r in source if r['dataset']==dataset and (r['dataset'],r['id']) not in excluded],key=lambda r:r['id'])
    random.Random(2029).shuffle(candidates)
    if len(candidates)<20:raise ValueError('Insufficient fresh training examples')
    pools[dataset]=candidates[:20]
samples=[pools[d][i] for i in range(20) for d in ['wikitq','tabfact','fetaqa']]
if any(s['split']!='train' for s in samples):raise ValueError('Only source training samples allowed')
# 保存逐条结构化产物到 root / 'samples.jsonl'，供后续重放或审核读取。
write_jsonl(root/'samples.jsonl',samples)
model={'backend':'local','model':str(Path('models/Qwen2.5-32B-Instruct').resolve()),
       'device_map':'balanced','max_input_tokens':4096,'max_new_tokens':1024,'seed':2029}
configs={'whole':{'mode':'A4','model_config':model,'settings':{'max_operations':8,'max_repairs':1}},
         'interactive':{'mode':'B5','model_config':model,'settings':{'max_operations':8,'max_repairs':1,'max_block':2}}}
for name,cfg in configs.items():(root/f'{name}.json').write_text(json.dumps(cfg,indent=2))
(root/'protocol.json').write_text(json.dumps({'development_only':True,'seed':2029,'n_per_task':20,
    'samples_sha256':digest(samples),'source_split':'train','selection_uses_labels_or_lengths':False,
    'excluded_source_files':sources,'excluded_ids_sha256':digest(sorted(excluded)),
    'whole':'32B A4 full executable plan','interactive':'same 32B, B5 interface, at most two operations per call',
    'no_student_adapter':True,'common_limits':{'max_operations':8,'max_repairs':1,'input_tokens':4096,'new_tokens_per_call':1024},
    'cost_limit':'Interaction changes the number of calls and total tokens; all costs/failures are reported, not assumed equal.',
    'timing':'Separate runs; no matched warm-up acceleration claim','freeform_auto_approved':False},indent=2))
print('INTERACTIVE_TEACHER_PREPARED',len(samples),flush=True)
