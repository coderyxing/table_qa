# 脚本用途：在同一扩充记录上比较任务与标签损失平衡，隔离采样和目标权重因素。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""U/W isolate task/label loss balancing on the exact same expanded records."""
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
from tableqa.models import LocalModel
from tableqa.inference import predict
from tableqa.experiments import settings

root=Path('runs/balanced-accuracy/results');pool=Path('runs/accuracy-training-pool');tuning=Path('runs/accuracy-development-v1')
if root.exists():raise ValueError('Output exists; preserve it before rerunning')
if shutil.disk_usage('.').free<8*1024**3:raise ValueError('Keep at least 8 GiB free')
# 读取结构化输入：pool / 'records.jsonl'，保留原有样本身份字段。
records=read_jsonl(pool/'records.jsonl');report=json.loads((pool/'report.json').read_text())
if digest(records)!=report['records_sha256']:raise ValueError('Audited training records changed')
# 读取结构化输入：tuning / 'samples.jsonl'，保留原有样本身份字段。
samples=read_jsonl(tuning/'samples.jsonl')
if any(s['split']!='validation' for s in samples):raise ValueError('Validation tuning only')
base=json.loads(Path('runs/coverage-student-control/results/B5/run.json').read_text())['config']
# 使用本阶段指定的 tokenizer；预算、提示模板与监督掩码应沿用同一版本。
base={**base,'max_seq_length':8192};tokenizer=AutoTokenizer.from_pretrained(base['model'],local_files_only=True)
prepared={};configs={}
for name,policy in [('U','none'),('W','task_label')]:
    cfg={**base,'loss_balance':policy};configs[name]=cfg
    _,prepared[name]=prepare_training_data(cfg,records,tokenizer)
if prepared['U']['sampling_sequence_sha256']!=prepared['W']['sampling_sequence_sha256']:
    raise ValueError('Loss balancing changed actual input/target sampling')
# 保存逐条结构化产物到 root / 'samples.jsonl'，供后续重放或审核读取。
root.mkdir(parents=True);write_jsonl(root/'samples.jsonl',samples)
(root/'preflight.json').write_text(json.dumps(prepared,indent=2))
(root/'protocol.json').write_text(json.dumps({'development_only':True,'U':'expanded sources, unweighted loss',
    'W':'same expanded sources, task/kind/label weighted loss','same_raw_records_sha256':digest(records),
    'same_actual_sampling_sequence':True,'same_epochs_seed_rank':True,'same_actual_target_tokens':True,
    'C_vs_U':'Data expansion plus official screening and training length allowance; not a single-module ablation',
    'confirmation_not_evaluated':True,'no_fetaqa_operation_auto_approval':True},indent=2))
for name in ['U','W']:
    cfg=configs[name];out=root/name;out.mkdir();(out/'run.json').write_text(json.dumps(metadata(cfg),indent=2))
    # 执行当前配置的训练阶段，输出检查点与实际训练记录统计。
    torch.cuda.reset_peak_memory_stats();statistics=train(cfg,records,out/'adapter')
    (out/'training-statistics.json').write_text(json.dumps(statistics,indent=2));gc.collect();torch.cuda.empty_cache()
    backend=LocalModel(cfg['model'],adapter=str(out/'adapter'),max_new_tokens=512,max_input_tokens=8192,seed=42)
    predictions=[]
    for sample in samples:
        # 对当前样本运行完整推理入口，返回值包含答案、状态、错误和调用成本。
        pred=predict(sample,backend,settings('B5',max_repairs=2));pred['metrics']['latency_comparable']=False
        # 保存逐条结构化产物到 out / 'predictions.jsonl'，供后续重放或审核读取。
        predictions.append(pred);write_jsonl(out/'predictions.jsonl',predictions)
        (root/'progress.json').write_text(json.dumps({'condition':name,'completed':len(predictions),'expected':len(samples)}))
    del backend;gc.collect();torch.cuda.empty_cache()
scripts=Path(__file__).resolve().parent
for name in ['U','W']:
    subprocess.run([sys.executable,str(scripts/'evaluate_wtq_official.py'),str(root/'samples.jsonl'),str(root/name/'predictions.jsonl'),
        str(root/f'wtq-{name}'),'--evaluator','data/raw/wikitq-official/WikiTableQuestions-master/evaluator.py',
        '--tagged','data/raw/wikitq-official/WikiTableQuestions-master/tagged/data'],check=True)
subprocess.run([sys.executable,str(scripts/'analyze_runs.py'),str(root/'samples.jsonl'),str(root/'U/predictions.jsonl'),
    str(root/'comparison-U-vs-W'),'--predictions-b',str(root/'W/predictions.jsonl'),
    '--official-a',str(root/'wtq-U/scores.json'),'--official-b',str(root/'wtq-W/scores.json')],check=True)
subprocess.run([sys.executable,str(scripts/'analyze_runs.py'),str(root/'samples.jsonl'),str(tuning/'C-predictions.jsonl'),
    str(root/'comparison-C-vs-U'),'--predictions-b',str(root/'U/predictions.jsonl'),
    '--official-a',str(tuning/'wtq-C/scores.json'),'--official-b',str(root/'wtq-U/scores.json')],check=True)
print('BALANCED_ACCURACY_COMPLETE',flush=True)
