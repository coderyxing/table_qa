# 脚本用途：使用真实答案进行小规模 B0 训练，检查监督和模型保存流程。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Real-label B0 training pilot, distinct from teacher distillation experiments."""
import json
from pathlib import Path
import random
import argparse
from transformers import AutoTokenizer
from tableqa.cli import metadata
from tableqa.data import read_jsonl,write_jsonl
from tableqa.distillation import build_records
from tableqa.training import encode_record,train

# 定义本入口的参数解析器；位置参数、默认值和允许取值由相邻 add_argument 明确声明。
parser=argparse.ArgumentParser(); parser.add_argument('--output',default='runs/supervised-pilot'); args=parser.parse_args()
out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
if (out/'adapter').exists(): raise ValueError('Existing checkpoint: use a fresh pilot directory')
model=str(Path('models/Qwen2.5-3B-Instruct').resolve())
# 使用本阶段指定的 tokenizer；预算、提示模板与监督掩码应沿用同一版本。
tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
# 读取结构化输入：'data/normalized/joint_train_clean.jsonl'，保留原有样本身份字段。
data=read_jsonl('data/normalized/joint_train_clean.jsonl'); records=[]; selected=[]; rejected=[]
for dataset in ['wikitq','tabfact','fetaqa']:
    candidates=sorted([r for r in data if r['dataset']==dataset],key=lambda r:r['id'])
    random.Random(42).shuffle(candidates); count=0
    for sample in candidates:
        record=build_records(sample,[],'B0')[0]
        try: encode_record(record,tokenizer,4096)
        except ValueError:
            rejected.append({'dataset':dataset,'id':sample['id'],'reason':'overlength'}); continue
        records.append(record); selected.append({'dataset':dataset,'id':sample['id']}); count+=1
        if count==8: break
# 保存逐条结构化产物到 out / 'records.jsonl'，供后续重放或审核读取。
write_jsonl(out/'records.jsonl',records)
(out/'selection.json').write_text(json.dumps({'pilot_only':True,'seed':42,'selected':selected,'rejected':rejected},indent=2))
config={'model':model,'seed':42,'max_seq_length':4096,'lora_rank':8,'lora_alpha':16,
        'learning_rate':2e-4,'answer_weight':1,'micro_batch':1,'gradient_accumulation_steps':4,'max_epochs':1}
(out/'run.json').write_text(json.dumps(metadata(config),indent=2))
# 执行当前配置的训练阶段，输出检查点与实际训练记录统计。
statistics=train(config,records,out/'adapter')
(out/'statistics.json').write_text(json.dumps({'pilot_only':True,'method':'B0 ground-truth answers, no teacher trajectories',**statistics},indent=2))
print('SUPERVISED_PILOT_COMPLETE',json.dumps(statistics),flush=True)
