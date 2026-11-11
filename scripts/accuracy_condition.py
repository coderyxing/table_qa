# 脚本用途：运行一个冻结的上下文、修复或回退配置，原子保存结果以支持中断恢复。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""One frozen accuracy condition, with resumable atomic outputs and no reference routing."""
import argparse
import json
import os
from pathlib import Path
import torch
from tableqa.cli import metadata
from tableqa.data import read_jsonl,write_jsonl,digest
from tableqa.analysis import index_records
from tableqa.models import make_model
from tableqa.inference import predict
from tableqa.experiments import settings
from tableqa.recovery import predict_with_recovery

# 定义本入口的参数解析器；位置参数、默认值和允许取值由相邻 add_argument 明确声明。
p=argparse.ArgumentParser(description=__doc__);p.add_argument('condition',choices=['A','B','C']);args=p.parse_args()
root=Path('runs/accuracy-development-v1');name=args.condition
# 读取结构化输入：root / 'samples.jsonl'，保留原有样本身份字段。
cfg=json.loads((root/f'{name}.json').read_text());samples=read_jsonl(root/'samples.jsonl')
if any(s['split']!='validation' for s in samples):raise ValueError('Validation tuning only')
run=metadata(cfg);identity=digest({'source':run['source_hash'],'config':cfg,'samples':samples})
meta=root/f'{name}-run.json'
if meta.exists() and json.loads(meta.read_text())['identity']!=identity:raise ValueError('Cannot resume changed accuracy condition')
meta.write_text(json.dumps({**run,'identity':identity,'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES')},indent=2))
canonical=root/(f'{name}-pairs.jsonl' if name=='C' else f'{name}-predictions.jsonl')
# 读取结构化输入：canonical，保留原有样本身份字段。
records=read_jsonl(canonical) if canonical.exists() else []
previous=[r['primary'] for r in records] if name=='C' else records
done=set(index_records(previous,'previous prediction'));expected={(r['dataset'],r['id']) for r in samples}
if done-expected:raise ValueError('Unknown prior accuracy outputs')
# 按已读取配置构造生成后端，实际模型身份和预算由该配置登记。
model=make_model(cfg['model_config']);backup=make_model(cfg['fallback_model_config']) if name=='C' else None
options=settings('B5',**cfg['settings']);torch.cuda.reset_peak_memory_stats()

# 先写临时文件，再原子替换目标，减少中断产生半条产物的风险。
# 输入：path（输入或输出路径）；rows。
# 相关调用：write_jsonl。
def atomic_write(path,rows):
    # 保存逐条结构化产物到 temporary，供后续重放或审核读取。
    temporary=path.with_suffix('.tmp');write_jsonl(temporary,rows);temporary.replace(path)

for sample in samples:
    if (sample['dataset'],sample['id']) in done:continue
    if name=='C':
        primary,routed=predict_with_recovery(sample,model,backup,options)
        primary['metrics']['latency_comparable']=False
        result={'primary':primary,'routed':routed}
    else:
        # 对当前样本运行完整推理入口，返回值包含答案、状态、错误和调用成本。
        result=predict(sample,model,options);result['metrics']['latency_comparable']=False
    records.append(result);atomic_write(canonical,records)
    (root/f'{name}-progress.json').write_text(json.dumps({'completed':len(records),'expected':len(samples),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'condition':name}))
    print('ACCURACY_PROGRESS',name,len(records),len(samples),flush=True)
if name=='C':
    atomic_write(root/'C-predictions.jsonl',[r['primary'] for r in records])
    atomic_write(root/'D-predictions.jsonl',[r['routed'] for r in records])
print('ACCURACY_CONDITION_COMPLETE',name,flush=True)
