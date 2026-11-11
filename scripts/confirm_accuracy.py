# 脚本用途：冻结开发选出的配置，在保留确认集上进行一次性复核。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Freeze the tuning winner, then perform one-shot reserved validation confirmation."""
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch
from tableqa.analysis import summarize_run
from tableqa.cli import metadata
from tableqa.data import read_jsonl,write_jsonl,digest
from tableqa.experiments import settings
from tableqa.inference import predict
from tableqa.models import make_model
from tableqa.recovery import predict_with_recovery

root=Path('runs/accuracy-confirmation/results');base=Path('runs/accuracy-development-v1')
if root.exists():raise ValueError('Confirmation already exists; do not repeatedly tune against it')
# 读取结构化输入：base / 'samples.jsonl'，保留原有样本身份字段。
tune=read_jsonl(base/'samples.jsonl');confirm=read_jsonl(base/'confirmation-samples.jsonl')
protocol=json.loads((base/'protocol.json').read_text())
if digest(tune)!=protocol['samples_sha256'] or digest(confirm)!=protocol['confirmation_sha256']:
    raise ValueError('Frozen validation cohorts changed')
if any(s['split']!='validation' for s in tune+confirm):raise ValueError('Validation-only selection/confirmation')
cases={}
for name in ['A','B','C','D']:
    cfg=json.loads((base/('C.json' if name=='D' else f'{name}.json')).read_text())
    cases[name]={'path':base/f'{name}-predictions.jsonl','official':base/f'wtq-{name}/scores.json',
                 'model':cfg['model_config'],'settings':cfg['settings'],'fallback':cfg.get('fallback_model_config') if name=='D' else None,'size':3}
scale=Path('runs/scale-7b-accuracy/results')
scale_cfg=json.loads((scale/'run.json').read_text())['config']
for name,limit,repairs in [('E',4096,1),('F',8192,2),('G',8192,2)]:
    cases[name]={'path':scale/f'{name}-predictions.jsonl','official':scale/f'wtq-{name}/scores.json',
        'model':{'backend':'local','model':scale_cfg['model'],'adapter':str((scale/'adapter').resolve()),
                 'max_input_tokens':limit,'max_new_tokens':512,'seed':42},
        'settings':{'max_operations':8,'max_block':4,'max_repairs':repairs},
        'fallback':cases['D']['fallback'] if name=='G' else None,'size':7}
balanced=Path('runs/balanced-accuracy/results')
for name in ['U','W']:
    cfg=json.loads((balanced/name/'run.json').read_text())['config']
    cases[name]={'path':balanced/name/'predictions.jsonl','official':balanced/f'wtq-{name}/scores.json',
        'model':{'backend':'local','model':cfg['model'],'adapter':str((balanced/name/'adapter').resolve()),
                 'max_input_tokens':8192,'max_new_tokens':512,'seed':42},
        'settings':{'max_operations':8,'max_block':4,'max_repairs':2},'fallback':None,'size':3}
ranking=[]
for name,case in cases.items():
    # 读取结构化输入：case['path']，保留原有样本身份字段。
    predictions=read_jsonl(case['path']);official=json.loads(case['official'].read_text())
    result=summarize_run(tune,predictions,official)
    rows=result['rows']
    if {r['dataset'] for r in rows}!={'wikitq','tabfact'}:raise ValueError('Selection task coverage differs')
    score=sum(r['observed_score'] for r in rows)/len(rows)
    mean_calls=sum(r['mean_calls'] for r in rows)/len(rows)
    ranking.append({'name':name,'macro_accuracy':score,'per_task':{r['dataset']:r['observed_score'] for r in rows},
                    'mean_calls':mean_calls,'model_size_B':case['size'],'fallback':case['fallback'] is not None})
# Predetermined tie-break: accuracy, calls, model size, then stable name.
ranking.sort(key=lambda r:(-r['macro_accuracy'],r['mean_calls'],r['model_size_B'],r['name']))
winner=ranking[0]['name'];selected=cases[winner]
root.mkdir(parents=True)
# 对适配器权重及配置登记摘要，确保确认阶段使用同一检查点。
# 输入：case。
# 相关调用：hashlib.sha256(path.read_bytes()).hexdigest。
def adapter_sha(case):
    path=Path(case['model']['adapter'])/'adapter_model.safetensors'
    return hashlib.sha256(path.read_bytes()).hexdigest()
selection={'development_only':True,'winner':winner,'ranking':ranking,
    'rule':'mean of two task accuracies; tie by calls, size, name',
    'selection_uses_only_tuning_outputs':True,'selected_adapter_sha256':adapter_sha(selected),
    'baseline_adapter_sha256':adapter_sha(cases['A']),
    'selected_config':{k:v for k,v in selected.items() if k not in ['path','official']},
    'confirmation_sha256':digest(confirm),'confirmation_may_not_be_reused_as_unseen_after_this_run':True}
# This file is written BEFORE any confirmation prediction or scoring.
(root/'frozen-selection.json').write_text(json.dumps(selection,indent=2))
(root/'run.json').write_text(json.dumps(metadata(selection['selected_config']),indent=2))
# 保存逐条结构化产物到 root / 'samples.jsonl'，供后续重放或审核读取。
write_jsonl(root/'samples.jsonl',confirm)

# 用同一个冻结队列运行一个明确配置。
# 输入：name；case。
# 相关调用：write_jsonl、(root / 'progress.json').write_text。
# 返回形式含：outputs。
def run_condition(name,case):
    # 按已读取配置构造生成后端，实际模型身份和预算由该配置登记。
    model=make_model(case['model']);backup=make_model(case['fallback']) if case['fallback'] else None
    outputs=[];primaries=[]
    for sample in confirm:
        # 对当前样本运行完整推理入口，返回值包含答案、状态、错误和调用成本。
        if backup is None:out=predict(sample,model,settings('B5',**case['settings']))
        else:
            primary,out=predict_with_recovery(sample,model,backup,settings('B5',**case['settings']));primaries.append(primary)
        out['metrics']['latency_comparable']=False;outputs.append(out)
        # 保存逐条结构化产物到 temporary，供后续重放或审核读取。
        temporary=root/f'{name}.tmp';write_jsonl(temporary,outputs);temporary.replace(root/f'{name}-predictions.jsonl')
        (root/'progress.json').write_text(json.dumps({'condition':name,'completed':len(outputs),'expected':len(confirm)}))
    # 保存逐条结构化产物到 root / f'{name}-primary-predictions.jsonl'，供后续重放或审核读取。
    if primaries:write_jsonl(root/f'{name}-primary-predictions.jsonl',primaries)
    del model,backup;gc.collect();torch.cuda.empty_cache()
    return outputs

baseline=run_condition('baseline',cases['A'])
# 保存逐条结构化产物到 root / 'selected-predictions.jsonl'，供后续重放或审核读取。
if winner=='A':write_jsonl(root/'selected-predictions.jsonl',baseline)
else:run_condition('selected',selected)
scripts=Path(__file__).resolve().parent
for name in ['baseline','selected']:
    subprocess.run([sys.executable,str(scripts/'evaluate_wtq_official.py'),str(root/'samples.jsonl'),str(root/f'{name}-predictions.jsonl'),
        str(root/f'wtq-{name}'),'--evaluator','data/raw/wikitq-official/WikiTableQuestions-master/evaluator.py',
        '--tagged','data/raw/wikitq-official/WikiTableQuestions-master/tagged/data'],check=True)
subprocess.run([sys.executable,str(scripts/'analyze_runs.py'),str(root/'samples.jsonl'),str(root/'baseline-predictions.jsonl'),
    str(root/'comparison'),'--predictions-b',str(root/'selected-predictions.jsonl'),
    '--official-a',str(root/'wtq-baseline/scores.json'),'--official-b',str(root/'wtq-selected/scores.json')],check=True)
print('ACCURACY_CONFIRMATION_COMPLETE',winner,flush=True)
