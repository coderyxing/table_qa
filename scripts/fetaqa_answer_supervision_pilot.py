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


def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output');args=parser.parse_args();root=Path(args.output)
    if root.exists():raise ValueError('Preserve existing attempt; output already exists')
    if shutil.disk_usage('.').free<8*1024**3:raise ValueError('Keep 8 GiB free before starting')
    previous=Path('runs/fetaqa-transfer-v1/results')
    if not load(previous/'completion-audit.json')['passed']:raise ValueError('Previous transfer audit required')
    old=load(previous/'protocol.json');samples=read_jsonl(previous/'samples.jsonl')
    if digest(samples)!=old['samples_sha256']:raise ValueError('Validation cohort changed')
    model_config={k:v for k,v in old['model_config'].items() if k not in {'adapter','adapter_revision','cache'}}
    model_config['device_map']='auto'
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model_config['model']);tokenizer.pad_token=tokenizer.eos_token
    maximum=8192;pool=read_jsonl('data/normalized/joint_train_clean.jsonl')
    if any(s['split']!='train' for s in pool):raise ValueError('Only training split may enter the pool')
    candidates=sorted([s for s in pool if s['dataset']=='fetaqa'],key=lambda s:s['id']);random.Random(2037).shuffle(candidates)
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
    write_jsonl(root/'train-samples.jsonl',selected);write_jsonl(root/'records.jsonl',records);write_jsonl(root/'validation-samples.jsonl',samples)
    print('FETAQA_B0_FROZEN',len(records),target_tokens,flush=True)
    import torch
    outputs={}
    def evaluate(name,adapter=None):
        model=make_model({**model_config,**({'adapter':str(adapter.resolve())} if adapter else {})});rows=[]
        if name=='SD' and (adapter is None or not getattr(model.model,'peft_config',None)):
            raise ValueError('SD must actually load the saved LoRA adapter')
        for i,sample in enumerate(samples):
            prediction=predict(sample,model,settings('B0',max_repairs=0));prediction['metrics']['latency_comparable']=False
            rows.append(prediction);write_jsonl(root/f'{name}-predictions.jsonl',rows)
            save(root/'progress.json',{'condition':name,'completed':i+1,'expected':len(samples)})
            print('FETAQA_B0_EVALUATION',name,i+1,len(samples),prediction['status'],flush=True)
        outputs[name]=rows;del model;gc.collect();torch.cuda.empty_cache()
    evaluate('QD')
    save(root/'phase.json',{'phase':'training','steps_expected':32})
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
