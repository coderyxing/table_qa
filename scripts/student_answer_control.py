# 脚本用途：在已有 B2/B5 的相同来源池上补齐 B0 答案监督对照。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Complete the B0 development control on the exact existing B2/B5 source pool."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import torch
from transformers import AutoTokenizer
from tableqa.analysis import index_records,validate_run
from tableqa.cli import metadata
from tableqa.data import read_jsonl,write_jsonl,digest
from tableqa.distillation import build_records
from tableqa.experiments import settings
from tableqa.inference import predict
from tableqa.models import LocalModel
from tableqa.quality import audit_trajectory
from tableqa.training import train,encode_record

# 定义本入口的参数解析器；位置参数、默认值和允许取值由相邻 add_argument 明确声明。
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',default='runs/student-answer-control/results')
# 解析本次命令的实际参数，后续样本路径和实验开关均由此确定。
args=p.parse_args()
root=Path(args.output)
if root.exists():raise ValueError('Output exists: retain it and select a new output directory')
source=Path('runs/medium-development');students=source/'students'
previous=json.loads((students/'report.json').read_text())
if set(previous['models'])!={'B2','B5'}:raise ValueError('Both predecessor students must finish')
# 读取结构化输入：source / 'samples.jsonl'，保留原有样本身份字段。
samples=read_jsonl(source/'samples.jsonl');predictions=read_jsonl(source/'A4-predictions.jsonl')
refs,preds,_=validate_run(samples,predictions)
# 读取结构化输入：students / 'accepted-samples.jsonl'，保留原有样本身份字段。
accepted=read_jsonl(students/'accepted-samples.jsonl');keys=list(index_records(accepted,'accepted source'))
if not keys or any(k not in refs for k in keys):raise ValueError('Invalid paired source pool')
# 读取结构化输入：students / 'validation-samples.jsonl'，保留原有样本身份字段。
validation=read_jsonl(students/'validation-samples.jsonl')
if any(r['split']!='validation' for r in validation):raise ValueError('Explicit validation split required')
if {k[0] for k in keys}!={r['dataset'] for r in validation}:raise ValueError('Task coverage differs')
cfg=json.loads((students/'B2/run.json').read_text())['config']
cfg5=json.loads((students/'B5/run.json').read_text())['config']
if cfg!=cfg5:raise ValueError('B2/B5 training configurations differ')
# 使用本阶段指定的 tokenizer；预算、提示模板与监督掩码应沿用同一版本。
tokenizer=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True)
records=[]
for key in keys:
    sample=refs[key]
    if sample['split']!='train' or not audit_trajectory(sample,preds[key])['decision'].startswith('accepted_'):
        raise ValueError(f'Previously accepted source no longer passes current audit: {key}')
    addition=build_records(sample,preds[key]['operations'],'B0')
    for record in addition:encode_record(record,tokenizer,cfg['max_seq_length'])
    records+=addition
root.mkdir(parents=True)
# 保存逐条结构化产物到 root / 'accepted-samples.jsonl'，供后续重放或审核读取。
write_jsonl(root/'accepted-samples.jsonl',accepted);write_jsonl(root/'records.jsonl',records)
# 保存逐条结构化产物到 root / 'validation-samples.jsonl'，供后续重放或审核读取。
write_jsonl(root/'validation-samples.jsonl',validation)
protocol={'development_only':True,'source_pool_sha256':digest(accepted),'n_original_samples':len(keys),
          'datasets':sorted({k[0] for k in keys}),'training_comparison':'same original source pool, NOT equal target-token budgets',
          'baseline':'untrained same 3B, answer-only prompt','validation':'same fixed validation IDs as existing B2/B5',
          'timing':'separate runs, no matched warm-up; observed timing is not controlled acceleration',
          'freeform_auto_approved':False}
(root/'protocol.json').write_text(json.dumps(protocol,indent=2))
(root/'run.json').write_text(json.dumps(metadata(cfg),indent=2))

# 重新加载指定检查点并在固定队列上保存评价。
# 输入：adapter；name。
# 相关调用：write_jsonl。
# 返回形式含：output。
def evaluate_checkpoint(adapter,name):
    model=LocalModel(cfg['model'],adapter=adapter,max_new_tokens=512,max_input_tokens=4096,seed=42)
    output=[]
    for sample in validation:
        # 对当前样本运行完整推理入口，返回值包含答案、状态、错误和调用成本。
        row=predict(sample,model,settings('B0'));row['checkpoint_role']=name
        # 保存逐条结构化产物到 root / f'{name}-predictions.jsonl'，供后续重放或审核读取。
        output.append(row);write_jsonl(root/f'{name}-predictions.jsonl',output)
    del model;gc.collect();torch.cuda.empty_cache()
    return output

evaluate_checkpoint(None,'base-B0')
torch.cuda.reset_peak_memory_stats()
# 执行当前配置的训练阶段，输出检查点与实际训练记录统计。
statistics=train(cfg,records,root/'adapter')
gc.collect();torch.cuda.empty_cache()
(root/'training-statistics.json').write_text(json.dumps(statistics,indent=2))
evaluate_checkpoint(str(root/'adapter'),'trained-B0')
scripts=Path(__file__).resolve().parent
evaluator='data/raw/wikitq-official/WikiTableQuestions-master/evaluator.py'
tagged='data/raw/wikitq-official/WikiTableQuestions-master/tagged/data'
paths={'base-B0':root/'base-B0-predictions.jsonl','trained-B0':root/'trained-B0-predictions.jsonl',
       'B2':students/'B2/validation-predictions.jsonl','B5':students/'B5/validation-predictions.jsonl'}
for name,path in paths.items():
    subprocess.run([sys.executable,str(scripts/'evaluate_wtq_official.py'),str(root/'validation-samples.jsonl'),str(path),
                    str(root/f'wtq-{name}'),'--evaluator',evaluator,'--tagged',tagged],check=True)
for name_a,name_b in [('base-B0','trained-B0'),('trained-B0','B2'),('trained-B0','B5'),('B2','B5')]:
    subprocess.run([sys.executable,str(scripts/'analyze_runs.py'),str(root/'validation-samples.jsonl'),str(paths[name_a]),
                    str(root/f'comparison-{name_a}-vs-{name_b}'),'--predictions-b',str(paths[name_b]),
                    '--official-a',str(root/f'wtq-{name_a}/scores.json'),'--official-b',str(root/f'wtq-{name_b}/scores.json')],check=True)
print('STUDENT_ANSWER_CONTROL_COMPLETE',json.dumps({'n_train':len(keys),'n_validation':len(validation),
      'supervised_tokens_per_epoch':statistics['supervised_tokens_per_epoch']}),flush=True)
