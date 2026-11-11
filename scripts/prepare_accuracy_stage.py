# 脚本用途：冻结上下文、修复、回退对照及独立确认样本。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Preregister input/repair/fallback accuracy controls and a reserved confirmation set."""
import json
from pathlib import Path
from tableqa.data import read_jsonl,write_jsonl,digest,table_fingerprint
from tableqa.tuning import accuracy_cohorts
root=Path('runs/accuracy-development-v1');root.mkdir(parents=True,exist_ok=True)
if (root/'protocol.json').exists():raise ValueError('Protocol exists; use the frozen runner')
previous=[]
for path in sorted(Path('runs').glob('**/validation-samples.jsonl')):
    # 读取结构化输入：path，保留原有样本身份字段。
    previous += [r for r in read_jsonl(path) if r['split']=='validation']
# 读取结构化输入：'data/normalized/joint_validation_clean.jsonl'，保留原有样本身份字段。
validation=read_jsonl('data/normalized/joint_validation_clean.jsonl')
cohorts=accuracy_cohorts(validation,previous,100,2030)
# 读取结构化输入：'data/normalized/joint_train_clean.jsonl'，保留原有样本身份字段。
training=read_jsonl('data/normalized/joint_train_clean.jsonl');train_fps=set();seen=set()
for r in training:
    key=(r['dataset'],r['table_id'])
    if key not in seen:train_fps.add(table_fingerprint(r['table']));seen.add(key)
if any(table_fingerprint(r['table']) in train_fps for rows in cohorts.values() for r in rows):
    raise ValueError('Chosen validation table overlaps training content')
# 保存逐条结构化产物到 root / 'samples.jsonl'，供后续重放或审核读取。
write_jsonl(root/'samples.jsonl',cohorts['tune'])
# 保存逐条结构化产物到 root / 'confirmation-samples.jsonl'，供后续重放或审核读取。
write_jsonl(root/'confirmation-samples.jsonl',cohorts['confirmation'])
model_path=str(Path('models/Qwen2.5-3B-Instruct').resolve())
max_positions=json.loads(Path(model_path,'config.json').read_text())['max_position_embeddings']
if 8192+512>max_positions:raise ValueError('8K condition exceeds actual model context configuration')
for name,limit,repairs in [('A',4096,1),('B',8192,1),('C',8192,2)]:
    cfg={'mode':'B5','model_config':{'backend':'local','model':model_path,
         'adapter':str(Path('runs/coverage-student-control/results/B5/adapter').resolve()),
         'max_input_tokens':limit,'max_new_tokens':512,'seed':42},
         'settings':{'max_operations':8,'max_block':4,'max_repairs':repairs}}
    if name=='C':cfg['fallback_model_config']={'backend':'local','model':model_path,
        'adapter':str(Path('runs/student-answer-control/results/adapter').resolve()),
        'max_input_tokens':8192,'max_new_tokens':512,'seed':42}
    (root/f'{name}.json').write_text(json.dumps(cfg,indent=2))
(root/'protocol.json').write_text(json.dumps({'development_only':True,'selection_seed':2030,'n_per_task':100,
    'datasets':['wikitq','tabfact'],'split':'validation','one_question_per_table':True,
    'previous_validation_tables_excluded':True,'confirmation_tables_disjoint':True,
    'selection_uses_labels_or_lengths':False,'samples_sha256':digest(cohorts['tune']),
    'confirmation_sha256':digest(cohorts['confirmation']),'confirmation_not_evaluated_by_this_stage':True,
    'A':'3B coverage B5, input4096, repairs1','B':'same checkpoint, input8192, repairs1',
    'C':'same checkpoint, input8192, repairs2','D':'C plus failure-only 3B B0 answer fallback',
    'fallback_not_executable_evidence':True,'feTaQA':'Operation training and factual assessment still await human review.',
    'timing':'Accuracy tuning on two A40s, C measured with backup resident; not a matched latency benchmark.'},indent=2))
print('ACCURACY_STAGE_PREPARED',len(cohorts['tune']),flush=True)
