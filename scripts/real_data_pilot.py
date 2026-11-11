# 脚本用途：在小批量真实训练题上检查推理入口和结果保存。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Small training-split inference pilot. Never use these scores as test results."""
import json
from pathlib import Path
import random
from tableqa.data import read_jsonl,write_jsonl
from tableqa.models import LocalModel
from tableqa.inference import predict
from tableqa.experiments import settings
from tableqa.evaluation import evaluate
out=Path('runs/real-data-pilot'); out.mkdir(parents=True,exist_ok=True)
samples=[]
for dataset in ['wikitq','fetaqa']:
    # 读取结构化输入：f'data/normalized/{dataset}_train.jsonl'，保留原有样本身份字段。
    rows=[r for r in read_jsonl(f'data/normalized/{dataset}_train.jsonl') if len(r['table']['rows'])<=30]
    random.Random(42).shuffle(rows); samples+=rows[:4]
# 保存逐条结构化产物到 out / 'samples.jsonl'，供后续重放或审核读取。
write_jsonl(out/'samples.jsonl',samples)
model=LocalModel(str(Path('models/Qwen2.5-3B-Instruct').resolve()),max_new_tokens=256)
predictions=[]
for sample in samples:
    # 对当前样本运行完整推理入口，返回值包含答案、状态、错误和调用成本。
    p=predict(sample,model,settings('B0')); predictions.append(p)
    # 保存逐条结构化产物到 out / 'predictions.jsonl'，供后续重放或审核读取。
    write_jsonl(out/'predictions.jsonl',predictions)
report={'pilot_only':True,'split':'train','method':'unfinetuned 3B answer-only',
        'metrics':evaluate(predictions,samples)}
(out/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
