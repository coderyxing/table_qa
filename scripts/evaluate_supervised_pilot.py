# 脚本用途：检查小规模答案监督适配器在验证划分上的运行情况。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Validation-split smoke evaluation of the tiny answer-supervised adapter."""
import json
from pathlib import Path
import random
import argparse
from tableqa.cli import metadata
from tableqa.data import read_jsonl,write_jsonl
from tableqa.models import LocalModel
from tableqa.inference import predict
from tableqa.experiments import settings
from tableqa.evaluation import evaluate

# 定义本入口的参数解析器；位置参数、默认值和允许取值由相邻 add_argument 明确声明。
parser=argparse.ArgumentParser(); parser.add_argument('--directory',default='runs/supervised-pilot'); args=parser.parse_args()
root=Path(args.directory)
if not (root/'statistics.json').exists(): raise RuntimeError('Training has not completed')
if (root/'validation-predictions.jsonl').exists(): raise ValueError('Validation output exists')
# 读取结构化输入：'data/normalized/joint_validation_clean.jsonl'，保留原有样本身份字段。
data=read_jsonl('data/normalized/joint_validation_clean.jsonl'); samples=[]
for dataset in ['wikitq','tabfact','fetaqa']:
    pool=sorted([r for r in data if r['dataset']==dataset],key=lambda r:r['id'])
    random.Random(42).shuffle(pool); samples+=pool[:3]
# 保存逐条结构化产物到 root / 'validation-samples.jsonl'，供后续重放或审核读取。
write_jsonl(root/'validation-samples.jsonl',samples)
model_path=str(Path('models/Qwen2.5-3B-Instruct').resolve())
config={'model':model_path,'adapter':str(root/'adapter'),'mode':'B0','split':'validation','seed':42}
(root/'validation-run.json').write_text(json.dumps(metadata(config),indent=2))
model=LocalModel(model_path,adapter=str(root/'adapter'),max_new_tokens=256)
predictions=[]
for r in samples:
    # 对当前样本运行完整推理入口，返回值包含答案、状态、错误和调用成本。
    predictions.append(predict(r,model,settings('B0')))
    # 保存逐条结构化产物到 root / 'validation-predictions.jsonl'，供后续重放或审核读取。
    write_jsonl(root/'validation-predictions.jsonl',predictions)
report={'pilot_only':True,'n_training_samples':24,'n_validation_samples':len(samples),
        'metrics':evaluate(predictions,samples),'official_test_evaluation':False}
(root/'validation-report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
