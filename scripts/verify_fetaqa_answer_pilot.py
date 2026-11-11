# 脚本用途：重新加载已保存适配器，修正并审计答案监督的评价产物。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Correct and audit saved-adapter evaluation without repeating training.

The initial launcher omitted the SD adapter argument. Preserve its files as
invalid SD evaluations; use the frozen weights, inputs and inference settings
to produce a separate verified evaluation. Check the actual loaded PEFT model.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

from tableqa.cli import metadata
from tableqa.data import digest,read_jsonl,write_jsonl
from tableqa.experiments import settings
from tableqa.models import make_model
from tableqa.inference import predict
from tableqa.reporting import freeform_metrics,compare_freeform


# 读取指定结构化产物，交由当前阶段核验来源和配置。
# 输入：path（输入或输出路径）。
# 相关调用：Path(path).read_text。
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
# 计算文件或产物内容摘要，供模型、输入和协议绑定。
# 输入：path（输入或输出路径）。
# 相关调用：hashlib.sha256(Path(path).read_bytes()).hexdigest。
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
# 创建输出目录并保存当前阶段的结构化产物。
# 输入：path（输入或输出路径）；value。
# 相关调用：Path(path).write_text。
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


# 解析当前脚本参数，按模块说明执行本次冻结阶段并保存产物。
# 相关调用：read_jsonl、write_jsonl、(root / 'report.md').write_text、(root / 'COMPLETE').write_text。
# 异常约束：Preserve existing verified evaluation
def main():
    # 定义本入口的参数解析器；位置参数、默认值和允许取值由相邻 add_argument 明确声明。
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage');args=p.parse_args();stage=Path(args.stage)
    root=stage/'verified-evaluation'
    if root.exists():raise ValueError('Preserve existing verified evaluation')
    if not (stage/'results/COMPLETE').is_file():raise ValueError('Wait for the original training/evaluation attempt to finish')
    original=stage/'results';protocol=load(original/'protocol.json');stats=load(original/'statistics.json')
    config=load(original/'training-config.json');adapter=original/'adapter'
    if load(adapter/'training_config.json')!=config:raise ValueError('Saved adapter training config mismatch')
    # 读取结构化输入：original / 'records.jsonl'，保留原有样本身份字段。
    records=read_jsonl(original/'records.jsonl');samples=read_jsonl(original/'validation-samples.jsonl')
    if digest(records)!=protocol['records_sha256'] or digest(samples)!=protocol['validation_samples_sha256']:
        raise ValueError('Frozen records/validation changed')
    exposure=load(adapter/'training_exposure.json')['per_source'];history=load(adapter/'training_history.json')
    if max(r.get('step',0) for r in history)!=32 or stats['records_after_alignment']!=512:
        raise ValueError('Wrong actual training steps/records')
    if stats['padding_records'] or stats['dropped_budget_records']:raise ValueError('Unexpected exposure changes')
    if {(r['dataset'],r['id']) for r in exposure}!={(r['dataset'],r['id']) for r in records}:
        raise ValueError('Training source coverage differs')
    if sum(r['target_tokens'] for r in exposure)!=config['frozen_supervised_tokens_per_epoch']:
        raise ValueError('Target token mismatch')
    inference=protocol['inference_model_config']
    if any(k in inference for k in ['adapter','cache']):raise ValueError('QD must be the unmodified uncached base')
    # 读取结构化输入：original / 'QD-predictions.jsonl'，保留原有样本身份字段。
    qd=read_jsonl(original/'QD-predictions.jsonl')
    # Unlike training.py, the inference/reporting modules did not change.
    old_hashes={name:sha(stage/'code/tableqa'/name) for name in ['models.py','inference.py','reporting.py','core.py']}
    current=Path(__import__('tableqa').__file__).parent
    if old_hashes!={name:sha(current/name) for name in old_hashes}:raise ValueError('Use identical frozen inference/metric modules')
    adapter_sha=sha(adapter/'adapter_model.safetensors')
    bound={'development_only':True,'automatic_adoption':False,'old_SD_invalid':True,
           'old_SD_invalid_reason':'Launcher omitted adapter argument; those outputs were the base model.',
           'reuses_unchanged_QD_baseline':True,'QD_predictions_sha256':sha(original/'QD-predictions.jsonl'),
           'adapter_sha256':adapter_sha,'actual_gradient_normalization':stats.get('gradient_normalization','legacy-unaveraged-microbatches'),
           'validation_samples_sha256':digest(samples),'inference_config':{**inference,'adapter':str(adapter.resolve())},
           'inference_module_hashes':old_hashes,'human_factual_consistency':None}
    # 保存逐条结构化产物到 root / 'samples.jsonl'，供后续重放或审核读取。
    root.mkdir();save(root/'protocol.json',bound);save(root/'run.json',metadata(bound));write_jsonl(root/'samples.jsonl',samples)
    # 按已读取配置构造生成后端，实际模型身份和预算由该配置登记。
    model=make_model(bound['inference_config'])
    if not getattr(model.model,'peft_config',None) or not model.model.active_adapters:
        raise ValueError('Saved adapter is not active in the actual model')
    save(root/'loaded-adapter.json',{'active_adapters':model.model.active_adapters,
         'adapter_sha256':adapter_sha,'model_class':type(model.model).__name__})
    sd=[]
    for i,sample in enumerate(samples):
        # 对当前样本运行完整推理入口，返回值包含答案、状态、错误和调用成本。
        row=predict(sample,model,settings('B0',max_repairs=0));row['metrics']['latency_comparable']=False
        # 保存逐条结构化产物到 root / 'SD-predictions.jsonl'，供后续重放或审核读取。
        sd.append(row);write_jsonl(root/'SD-predictions.jsonl',sd)
        print('FETAQA_VERIFIED_SD',stage.name,i+1,len(samples),row['status'],flush=True)
    del model;gc.collect()
    report={'development_only':True,'automatic_adoption':False,'old_SD_invalid':True,
            'actual_gradient_normalization':bound['actual_gradient_normalization'],
            'conditions':{'QD':freeform_metrics(qd,samples),'SD':freeform_metrics(sd,samples)},
            'comparison_QD_to_SD':compare_freeform(qd,sd,samples,repeats=500)}
    save(root/'report.json',report)
    lines=['# FeTaQA实际加载适配器后的B0对照','',
           '512条官方训练答案、12条已用验证样本；普通B0监督，非创新/人工事实评价。',
           '原results/SD输出漏载适配器而无效，保留原文件；本目录使用实际激活并绑定哈希的LoRA。','',
           '|配置|BLEU|ROUGE-1|ROUGE-2|ROUGE-L|失败|非文本输出|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,m in report['conditions'].items():
        lines.append(f"|{name}|{m['sacrebleu']:.3f}|{m['rouge1_f1']:.3f}|{m['rouge2_f1']:.3f}|{m['rougeL_f1']:.3f}|{m['failures']}|{m['invalid_answer_text']}|")
    lines+=['','梯度归一化：'+bound['actual_gradient_normalization'],
            'BLEU为0—100，ROUGE为0—1；所有失败/非文本输出按空文本计分。',
            '样本配对区间见report.json，不代表训练种子波动；不测跨任务保持或正式效率。']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if sha(adapter/'adapter_model.safetensors')!=adapter_sha:raise ValueError('Adapter changed during evaluation')
    save(root/'completion-audit.json',{'passed':True,'scope':'artifact/loaded-model consistency, no semantic approval',
         'actual_adapter_loaded':True,'old_SD_invalid':True,'predictions':24,'actual_optimizer_steps':32,
         'source_records':512,'target_tokens':stats['supervised_tokens_per_epoch'],
         'artifact_sha256':{f:sha(root/f) for f in ['protocol.json','run.json','samples.jsonl',
                'SD-predictions.jsonl','loaded-adapter.json','report.json','report.md']}})
    (root/'COMPLETE').write_text('actual saved adapter evaluation complete\n');print('FETAQA_VERIFIED_EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':main()
