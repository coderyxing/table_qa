"""Additional fixed-weight 100-example FeTaQA development comparison.

Independent of the 12-example cohort only in exact table/question membership;
not claimed as unseen across every historical project experiment. No tuning,
training or test data. Two GPUs may run conditions; timing is not comparable.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

from tableqa.cli import metadata
from tableqa.data import read_jsonl,write_jsonl,digest,table_fingerprint
from tableqa.experiments import settings
from tableqa.inference import predict
from tableqa.models import make_model
from tableqa.reporting import freeform_metrics,compare_freeform


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','QD','SD','report']);p.add_argument('output');args=p.parse_args();root=Path(args.output)
    stage=Path('runs/fetaqa-answer-supervision-v2');verified=stage/'verified-evaluation'
    if args.action=='prepare':
        if root.exists():raise ValueError('Preserve existing frozen development cohort')
        if not load(verified/'completion-audit.json')['actual_adapter_loaded']:raise ValueError('Verified model loading required')
        original=load(stage/'results/protocol.json');samples12=read_jsonl(verified/'samples.jsonl')
        excluded={table_fingerprint(s['table']) for s in samples12}
        candidates=sorted([s for s in read_jsonl('data/normalized/joint_validation_clean.jsonl') if s['dataset']=='fetaqa'],key=lambda s:s['id'])
        random.Random(2038).shuffle(candidates);samples=[];seen=set(excluded)
        for s in candidates:
            if s['split']!='validation':raise ValueError('Validation only')
            fp=table_fingerprint(s['table'])
            if fp in seen:continue
            seen.add(fp);samples.append(s)
            if len(samples)==100:break
        if len(samples)!=100:raise ValueError('Insufficient distinct validation tables')
        training=read_jsonl(stage/'results/train-samples.jsonl');train_fps={table_fingerprint(s['table']) for s in training}
        if any(table_fingerprint(s['table']) in train_fps for s in samples):raise ValueError('Training table overlap')
        adapter=stage/'results/adapter';config=original['inference_model_config']
        if any(k in config for k in ['adapter','cache']):raise ValueError('Base model configuration must not include adapter/cache')
        protocol={'development_only':True,'formal_test_read':False,'selection_seed':2038,'n':100,
            'no_exact_table_overlap_with_training_or_12_question_probe':True,
            'no_claim_all_historical_experiments_unseen':True,'selection_uses_labels_or_predictions':False,
            'samples_sha256':digest(samples),'model_config':config,'adapter':str(adapter.resolve()),
            'adapter_sha256':sha(adapter/'adapter_model.safetensors'),'training_sources':512,
            'fixed_checkpoint_no_parameter_selection':True,'gradient_normalization':'explicit-microbatch-mean-v1',
            'inference_settings':{'mode':'B0','max_repairs':0},'same_model_prompt_call_budget':True,
            'different_physical_A40s_allowed':True,'latency_comparable':False,
            'human_factual_consistency':None,'automatic_adoption':False,'bootstrap_repeats':500}
        root.mkdir(parents=True);save(root/'protocol.json',protocol);save(root/'run.json',metadata(protocol));write_jsonl(root/'samples.jsonl',samples)
        print('FETAQA_100_FROZEN',digest(samples),flush=True);return
    protocol=load(root/'protocol.json');samples=read_jsonl(root/'samples.jsonl')
    if len(samples)!=100 or digest(samples)!=protocol['samples_sha256']:raise ValueError('Frozen samples changed')
    if metadata({})['source_hash']!=load(root/'run.json')['source_hash']:raise ValueError('Execution modules changed')
    if sha(Path(protocol['adapter'])/'adapter_model.safetensors')!=protocol['adapter_sha256']:raise ValueError('Adapter changed')
    if args.action in ['QD','SD']:
        name=args.action;path=root/f'{name}-predictions.jsonl'
        if path.exists():raise ValueError('Preserve existing predictions; no silent reruns')
        config={**protocol['model_config'],**({'adapter':protocol['adapter']} if name=='SD' else {})}
        model=make_model(config)
        active=getattr(model.model,'peft_config',None)
        if bool(active)!=(name=='SD'):raise ValueError('Actual model adapter does not match condition')
        save(root/f'{name}-loaded-model.json',{'model_config':config,'actual_adapter_active':bool(active),
             'adapter_sha256':protocol['adapter_sha256'] if active else None})
        rows=[]
        for i,sample in enumerate(samples):
            row=predict(sample,model,settings('B0',max_repairs=0));row['metrics']['latency_comparable']=False
            rows.append(row);write_jsonl(path,rows);save(root/f'{name}-progress.json',{'completed':i+1,'expected':100})
            print('FETAQA_100',name,i+1,100,row['status'],flush=True)
        (root/f'{name}-COMPLETE').write_text('fixed condition complete\n');return
    if not all((root/f'{n}-COMPLETE').exists() for n in ['QD','SD']):raise ValueError('Wait for both conditions')
    if (root/'report.json').exists():raise ValueError('Preserve existing final report')
    outputs={n:read_jsonl(root/f'{n}-predictions.jsonl') for n in ['QD','SD']}
    report={'development_only':True,'automatic_adoption':False,'human_factual_consistency':None,
        'conditions':{n:freeform_metrics(rows,samples) for n,rows in outputs.items()},
        'comparison_QD_to_SD':compare_freeform(outputs['QD'],outputs['SD'],samples,repeats=500)}
    save(root/'report.json',report)
    lines=['# FeTaQA额外100题固定权重开发对照','',
           '与12题探针精确表不重叠，但不声称所有历史实验均未见；训练/参数/权重不因本批结果改变。','',
           '|配置|BLEU|ROUGE-1|ROUGE-2|ROUGE-L|失败|非文本输出|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for n,m in report['conditions'].items():
        lines.append(f"|{n}|{m['sacrebleu']:.3f}|{m['rouge1_f1']:.3f}|{m['rouge2_f1']:.3f}|{m['rougeL_f1']:.3f}|{m['failures']}|{m['invalid_answer_text']}|")
    lines+=['','QD原始3B，SD为同底座512条官方答案监督LoRA；相同B0提示/调用预算，实际加载检查已保存。',
        'BLEU为0—100，ROUGE为0—1；所有失败和非文本输出计空。',
        '不同物理A40并行，延迟不可直接比较；非正式test、非人工事实评价，也非两项创新的消融。']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files=['protocol.json','run.json','samples.jsonl','report.json','report.md']+[n+s for n in ['QD','SD'] for s in ['-predictions.jsonl','-loaded-model.json','-COMPLETE']]
    save(root/'completion-audit.json',{'passed':True,'scope':'artifact coverage and metric computation, no human approval',
        'predictions':200,'automatic_adoption':False,'artifact_sha256':{f:sha(root/f) for f in files}})
    (root/'COMPLETE').write_text('additional development comparison complete\n');print('FETAQA_100_COMPLETE',flush=True)


if __name__=='__main__':main()
