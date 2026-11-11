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
    rows=[r for r in read_jsonl(f'data/normalized/{dataset}_train.jsonl') if len(r['table']['rows'])<=30]
    random.Random(42).shuffle(rows); samples+=rows[:4]
write_jsonl(out/'samples.jsonl',samples)
model=LocalModel(str(Path('models/Qwen2.5-3B-Instruct').resolve()),max_new_tokens=256)
predictions=[]
for sample in samples:
    p=predict(sample,model,settings('B0')); predictions.append(p)
    write_jsonl(out/'predictions.jsonl',predictions)
report={'pilot_only':True,'split':'train','method':'unfinetuned 3B answer-only',
        'metrics':evaluate(predictions,samples)}
(out/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
