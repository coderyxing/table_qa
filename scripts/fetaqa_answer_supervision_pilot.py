# 脚本用途：使用 FeTaQA 官方训练答案构建独立答案监督基线，训练并保存自动评价。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Independent FeTaQA answer-only baseline, using official training answers.

This is ordinary B0 SFT, not reviewed operation-chain distillation or a proposed
innovation. Freeze 512 sources before scoring the already-used 12 validation
inputs; compare the unchanged base model and one fresh LoRA epoch.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil

from tableqa.cli import metadata
from tableqa.data import read_jsonl, write_jsonl, digest, table_fingerprint
from tableqa.distillation import build_records
from tableqa.experiments import settings
from tableqa.inference import predict
from tableqa.models import make_model
from tableqa.reporting import freeform_metrics, compare_freeform
from tableqa.training import encode_record, prepare_training_data, train


# 创建输出目录并保存当前阶段的结构化产物。
# 输入：path（输入或输出路径）；value。
# 相关调用：Path(path).write_text。
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
# 读取指定结构化产物，交由当前阶段核验来源和配置。
# 输入：path（输入或输出路径）。
# 相关调用：Path(path).read_text。
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
# 计算文件或产物内容摘要，供模型、输入和协议绑定。
# 输入：path（输入或输出路径）。
# 相关调用：hashlib.sha256(Path(path).read_bytes()).hexdigest。
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# 解析当前脚本参数，按模块说明执行本次冻结阶段并保存产物。
# 相关调用：read_jsonl、AutoTokenizer.from_pretrained、prepare_training_data、write_jsonl。
# 异常约束：Preserve existing attempt; output already exists
def main():
    # 定义本入口的参数解析器；位置参数、默认值和允许取值由相邻 add_argument 明确声明。
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output');args=parser.parse_args();root=Path(args.output)
    if root.exists():raise ValueError('Preserve existing attempt; output already exists')
    if shutil.disk_usage('.').free<8*1024**3:raise ValueError('Keep 8 GiB free before starting')
    previous=Path('runs/fetaqa-transfer-v1/results')
    if not load(previous/'completion-audit.json')['passed']:raise ValueError('Previous transfer audit required')
    # 读取结构化输入：previous / 'samples.jsonl'，保留原有样本身份字段。
    old=load(previous/'protocol.json');samples=read_jsonl(previous/'samples.jsonl')
    if digest(samples)!=old['samples_sha256']:raise ValueError('Validation cohort changed')
    model_config={k:v for k,v in old['model_config'].items() if k not in {'adapter','adapter_revision','cache'}}
    model_config['device_map']='auto'
    from transformers import AutoTokenizer
    # 使用本阶段指定的 tokenizer；预算、提示模板与监督掩码应沿用同一版本。
    tokenizer=AutoTokenizer.from_pretrained(model_config['model']);tokenizer.pad_token=tokenizer.eos_token
    # 读取结构化输入：'data/normalized/joint_train_clean.jsonl'，保留原有样本身份字段。
    maximum=8192;pool=read_jsonl('data/normalized/joint_train_clean.jsonl')
    if any(s['split']!='train' for s in pool):raise ValueError('Only training split may enter the pool')
    candidates=sorted([s for s in pool if s['dataset']=='fetaqa'],key=lambda s:s['id']);random.Random(2037).shuffle(candidates)
    # 读取结构化输入：'data/normalized/joint_validation_clean.jsonl'，保留原有样本身份字段。
    validation=read_jsonl('data/normalized/joint_validation_clean.jsonl')
    validation_fps={table_fingerprint(s['table']) for s in validation}
    selected=[];records=[];seen=set();skipped={'overlength':0,'duplicate_table':0,'validation_table_overlap':0}
    for sample in candidates:
        fp=table_fingerprint(sample['table'])
        if fp in validation_fps:skipped['validation_table_overlap']+=1;continue
        if fp in seen:skipped['duplicate_table']+=1;continue
        record,=build_records(sample,[],'B0')
        try:encode_record(record,tokenizer,maximum)
        except ValueError as exc:
            if not str(exc).startswith('Overlength training example'):raise
            skipped['overlength']+=1;continue
        selected.append(sample);records.append(record);seen.add(fp)
        if len(selected)==512:break
    if len(selected)!=512:raise ValueError('Insufficient distinct fitting FeTaQA training tables')
    target_tokens=sum(sum(t!=-100 for t in encode_record(r,tokenizer,maximum)['labels']) for r in records)
    config={'model':model_config['model'],'seed':42,'sampling_policy':'frozen',
            'frozen_records_sha256':digest(records),'frozen_supervised_tokens_per_epoch':target_tokens,
            'max_seq_length':maximum,'lora_rank':32,'lora_alpha':64,'learning_rate':1e-4,
            'answer_weight':1,'micro_batch':1,'gradient_accumulation_steps':16,'max_epochs':1,
            'loss_balance':'none'}
    _,preflight=prepare_training_data(config,records,tokenizer)
    model_dir=Path(config['model']);weight_hashes={p.name:sha(p) for p in sorted(model_dir.glob('*.safetensors'))}
    if not weight_hashes:raise ValueError('Expected verified local model weights')
    protocol={'development_only':True,'posthoc_exploration_after_transfer_diagnostic':True,
              'gradient_normalization':'explicit-microbatch-mean-v1',
              'n_training_sources':512,'selection_seed':2037,'one_training_question_per_exact_table':True,
              'selection_skipped_before_512':skipped,'maximum_train_length':maximum,
              'no_silent_training_or_inference_truncation':True,
              'training_source':'official FeTaQA train answers; no teacher operations/rationales',
              'train_samples_sha256':digest(selected),'records_sha256':digest(records),
              'validation_samples_sha256':digest(samples),'validation_n':len(samples),
              'validation_previously_used':True,'formal_test_read':False,
              'no_exact_table_overlap_with_all_clean_validation':True,
              'human_review_available':False,'freeform_operation_review_gate_unchanged':True,
              'training_config':config,'inference_model_config':model_config,
              'base_weight_sha256':weight_hashes,'conditions':{'QD':'unmodified base, B0','SD':'fresh 512-answer LoRA, B0'},
              'inference_settings':{'mode':'B0','max_repairs':0},'bootstrap_repeats':500,
              'automatic_adoption':False,'formal_latency_benchmark':False,
              'limits':['Ordinary answer-supervision feasibility baseline, not an innovation ablation.',
                        'One seed/epoch, 12 already-used validation examples; no independent confirmation.',
                        'No human factuality review or cross-task retention measurement.']}
    root.mkdir(parents=True);save(root/'protocol.json',protocol);save(root/'run.json',metadata(protocol))
    save(root/'preflight.json',preflight);save(root/'training-config.json',config)
    # 保存逐条结构化产物到 root / 'train-samples.jsonl'，供后续重放或审核读取。
    write_jsonl(root/'train-samples.jsonl',selected);write_jsonl(root/'records.jsonl',records);write_jsonl(root/'validation-samples.jsonl',samples)
    print('FETAQA_B0_FROZEN',len(records),target_tokens,flush=True)
    import torch
    outputs={}
    # 在约定的固定队列上生成预测并汇总评价。
    # 输入：name；adapter。
    # 相关调用：write_jsonl。
    # 异常约束：SD must actually load the saved LoRA adapter
    def evaluate(name,adapter=None):
        # 按已读取配置构造生成后端，实际模型身份和预算由该配置登记。
        model=make_model({**model_config,**({'adapter':str(adapter.resolve())} if adapter else {})});rows=[]
        if name=='SD' and (adapter is None or not getattr(model.model,'peft_config',None)):
            raise ValueError('SD must actually load the saved LoRA adapter')
        for i,sample in enumerate(samples):
            # 对当前样本运行完整推理入口，返回值包含答案、状态、错误和调用成本。
            prediction=predict(sample,model,settings('B0',max_repairs=0));prediction['metrics']['latency_comparable']=False
            # 保存逐条结构化产物到 root / f'{name}-predictions.jsonl'，供后续重放或审核读取。
            rows.append(prediction);write_jsonl(root/f'{name}-predictions.jsonl',rows)
            save(root/'progress.json',{'condition':name,'completed':i+1,'expected':len(samples)})
            print('FETAQA_B0_EVALUATION',name,i+1,len(samples),prediction['status'],flush=True)
        outputs[name]=rows;del model;gc.collect();torch.cuda.empty_cache()
    evaluate('QD')
    save(root/'phase.json',{'phase':'training','steps_expected':32})
    # 执行当前配置的训练阶段，输出检查点与实际训练记录统计。
    statistics=train(config,records,root/'adapter');save(root/'statistics.json',statistics)
    gc.collect();torch.cuda.empty_cache()
    history=load(root/'adapter/training_history.json');actual_steps=max(row.get('step',0) for row in history)
    if actual_steps!=32 or statistics['padding_records'] or statistics['dropped_budget_records']:
        raise ValueError('Actual training schedule differs from the frozen 512-source plan')
    if statistics['supervised_tokens_per_epoch']!=target_tokens:raise ValueError('Supervised token count changed')
    evaluate('SD',root/'adapter')
    report={'development_only':True,'ordinary_answer_supervision_not_innovation':True,
            'automatic_adoption':False,'human_factual_consistency':None,
            'conditions':{name:freeform_metrics(rows,samples) for name,rows in outputs.items()},
            'comparison_QD_to_SD':compare_freeform(outputs['QD'],outputs['SD'],samples,repeats=500)}
    save(root/'report.json',report)
    lines=['# FeTaQA官方答案B0监督试验','',
           '512条训练答案、一个种子/epoch；12条已用验证输入。普通答案监督可行性对照，不是论文创新效果。','',
           '|配置|BLEU|ROUGE-1|ROUGE-2|ROUGE-L|运行失败|非文本输出|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,m in report['conditions'].items():
        lines.append(f"|{name}|{m['sacrebleu']:.3f}|{m['rouge1_f1']:.3f}|{m['rouge2_f1']:.3f}|{m['rougeL_f1']:.3f}|{m['failures']}|{m['invalid_answer_text']}|")
    lines+=['','QD：原始3B；SD：同一底座新建LoRA，仅使用官方训练答案。相同B0提示和调用上限。',
            'BLEU范围0–100，ROUGE范围0–1；失败与非文本输出计空。配对区间见report.json。',
            '未验证跨任务保持能力或人工事实正确性，不自动替换现有系统，不与不同协议的公开成绩直接比较。']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for filename,value in weight_hashes.items():
        if sha(model_dir/filename)!=value:raise ValueError('Base weights changed during experiment')
    files=['protocol.json','run.json','training-config.json','preflight.json','train-samples.jsonl','records.jsonl',
           'validation-samples.jsonl','QD-predictions.jsonl','SD-predictions.jsonl','report.json','report.md','statistics.json']
    save(root/'completion-audit.json',{'passed':True,'scope':'runtime artifact/schedule consistency, no semantic approval',
         'actual_optimizer_steps':actual_steps,'training_sources':512,'predictions':24,
         'adapter_sha256':sha(root/'adapter/adapter_model.safetensors'),
         'artifact_sha256':{f:sha(root/f) for f in files},'automatic_adoption':False})
    (root/'COMPLETE').write_text('ordinary answer-supervision pilot complete\n')
    print('FETAQA_B0_PILOT_COMPLETE',flush=True)


if __name__=='__main__':main()
