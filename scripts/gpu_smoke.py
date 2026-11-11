# 脚本用途：用合成样本进行实际单卡 LoRA 训练和适配器重新加载验收。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Actual 3B LoRA + adapter reload on a synthetic fixture, NOT thesis results."""
import json
from pathlib import Path
from tableqa.cli import demo, save
from tableqa.data import read_jsonl
from tableqa.training import train
from tableqa.models import LocalModel
from tableqa.inference import predict
from tableqa.experiments import settings

root=Path('runs/gpu-smoke'); root.mkdir(parents=True,exist_ok=True)
model=Path('models/Qwen2.5-3B-Instruct').resolve()
if not (model/'DOWNLOAD_COMPLETE').exists(): raise RuntimeError('Model download incomplete')
demo(root/'fixture')
config={'model':str(model),'seed':42,'max_seq_length':4096,'lora_rank':8,'lora_alpha':16,
        'learning_rate':2e-4,'answer_weight':1,'micro_batch':1,'gradient_accumulation_steps':1,'max_epochs':1}
# 执行当前配置的训练阶段，输出检查点与实际训练记录统计。
stats=train(config,read_jsonl(root/'fixture/records.jsonl'),root/'adapter')
save(root/'train-statistics.json',{'synthetic_smoke_only':True,**stats})
import gc
import torch
gc.collect(); torch.cuda.empty_cache()
backend=LocalModel(str(model),adapter=str(root/'adapter'),max_new_tokens=512)
# 对当前样本运行完整推理入口，返回值包含答案、状态、错误和调用成本。
result=predict(read_jsonl(root/'fixture/examples.jsonl')[0],backend,settings('B5',max_repairs=1))
save(root/'prediction.json',result)
save(root/'status.json',{'synthetic_smoke_only':True,'training_completed':True,
                       'adapter_reload_completed':True,'prediction_status':result['status']})
print('GPU_SMOKE_COMPLETE; synthetic fixture only',flush=True)
