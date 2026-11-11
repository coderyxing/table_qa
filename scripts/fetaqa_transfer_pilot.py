"""No-human-review FeTaQA diagnostic: automatic metrics, no new training.

Compares fixed P/R/M checkpoints on zero target-task training transfer. No
reference/highlight is sent to a model; automatic overlap scores are not facts
or human trajectory approvals. This never changes the adopted model.
"""
import argparse
import gc
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


def select_samples(rows,n=12,seed=2036):
    if type(n) is not int or n<1:raise ValueError('Positive pilot size required')
    if any(r['split']!='validation' for r in rows):raise ValueError('Validation-only development pilot')
    candidates=sorted([r for r in rows if r['dataset']=='fetaqa'],key=lambda r:r['id'])
    random.Random(seed).shuffle(candidates);result=[];tables=set()
    for row in candidates:
        fp=table_fingerprint(row['table'])
        if fp in tables:continue
        result.append(row);tables.add(fp)
        if len(result)==n:return result
    raise ValueError('Insufficient distinct validation tables')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output');args=p.parse_args();root=Path(args.output)
    if root.exists():raise ValueError('Output exists; preserve the frozen diagnostic')
    previous=Path('runs/curriculum-student-v1')
    if not (previous/'completion-audit.json').exists():raise ValueError('Controlled student audit must complete first')
    audit=json.loads((previous/'completion-audit.json').read_text())
    if not audit['passed']:raise ValueError('Controlled student audit did not pass')
    old_protocol=json.loads((previous/'prepared/protocol.json').read_text())
    config=json.loads((previous/'prepared/R-config.json').read_text())
    adapters={'P':Path(config['initial_adapter']),'R':previous/'training-R/adapter','M':previous/'training-M/adapter'}
    hashes={};exposures={}
    for name,path in adapters.items():
        exposure=json.loads((path/'training_exposure.json').read_text())
        datasets={r['dataset'] for r in exposure['per_source']}
        if 'fetaqa' in datasets:raise ValueError('This experiment requires no project FeTaQA training exposure')
        exposures[name]=sorted(datasets)
        hashes[name]=hashlib.sha256((path/'adapter_model.safetensors').read_bytes()).hexdigest()
    if hashes['P']!=old_protocol['training']['initial_adapter_sha256']:
        raise ValueError('Original adapter changed')
    for name in ['R','M']:
        if hashes[name]!=audit['conditions'][name]['adapter_sha256']:raise ValueError('Trained adapter changed')
    samples=select_samples(read_jsonl('data/normalized/joint_validation_clean.jsonl'))
    source_samples=read_jsonl('runs/accuracy-training-pool/samples.jsonl')+read_jsonl('runs/tool-curriculum-v1/samples.jsonl')
    fps={table_fingerprint(s['table']) for s in source_samples}
    if any(table_fingerprint(s['table']) in fps for s in samples):raise ValueError('Transfer table overlaps known adapter training sources')
    chain_settings={'mode':'B5','typed_quantities':True,'semantic_extract':True,
                    'tool_demonstrations':True,'max_operations':8,'max_block':4,'max_repairs':2}
    conditions={'PD':{'adapter':'P','mode':'B0','max_repairs':0},'PC':{'adapter':'P',**chain_settings},
                'RC':{'adapter':'R',**chain_settings},'MC':{'adapter':'M',**chain_settings}}
    model_config={**old_protocol['model_config'],'device_map':'auto'}
    protocol={'development_only':True,'n':len(samples),'selection_seed':2036,'one_question_per_table':True,
              'selection_uses_reference_length_predictions_or_highlights':False,'samples_sha256':digest(samples),
              'conditions':conditions,'adapter_sha256':hashes,'training_dataset_exposure':exposures,
              'model_config':model_config,'human_review_available':False,'human_factual_consistency':None,
              'no_new_training':True,'no_target_task_training_samples_in_project_adapters':True,
              'pretraining_contamination_unknown':True,'no_references_or_highlights_to_model':True,
              'freeform_trajectory_training_gate_unchanged':True,'automatic_adoption':False,
              'corpus_bleu_and_rouge_1_2_L':True,'bootstrap_repeats':500,
              'formal_latency_benchmark':False,'PD_vs_PC_call_budgets_not_equal':True,
              'limits':['Small automatic-metric diagnostic, not human factuality or formal test.',
                        'PD/PC share weights but differ in prompt/interface and allowed calls.',
                        'RC/MC compare fixed data recipes, not an isolated innovation module.',
                        'M remains unadopted after WikiTQ regression; this probe cannot override that decision.']}
    root.mkdir(parents=True);write_jsonl(root/'samples.jsonl',samples)
    (root/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    (root/'run.json').write_text(json.dumps(metadata(protocol),indent=2),encoding='utf-8')
    outputs={name:[] for name in conditions}
    import torch
    for adapter_name in ['P','R','M']:
        model=make_model({**model_config,'adapter':str(adapters[adapter_name].resolve())})
        names=[name for name,c in conditions.items() if c['adapter']==adapter_name]
        for i,sample in enumerate(samples):
            ordered=names if i%2==0 else list(reversed(names))
            for name in ordered:
                options={k:v for k,v in conditions[name].items() if k not in {'adapter','mode'}}
                prediction=predict(sample,model,settings(conditions[name]['mode'],**options))
                prediction['metrics']['latency_comparable']=False
                outputs[name].append(prediction);write_jsonl(root/f'{name}-predictions.jsonl',outputs[name])
                (root/'progress.json').write_text(json.dumps({k:len(v) for k,v in outputs.items()}))
                print('FETAQA_TRANSFER',name,i+1,len(samples),prediction['status'],flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    report={'development_only':True,'automatic_overlap_metrics_only':True,
            'human_factual_consistency':None,'no_new_training':True,'automatic_adoption':False,
            'conditions':{name:freeform_metrics(preds,samples) for name,preds in outputs.items()},
            'comparisons':{a+'-vs-'+b:compare_freeform(outputs[a],outputs[b],samples,repeats=500)
                           for a,b in [('PD','PC'),('PC','RC'),('RC','MC')]}}
    (root/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# FeTaQA无人工审核的零目标任务训练迁移诊断','',
           '12条验证输入；只有自动重叠指标，无人工事实评价，不证明语义一致性或总体优越性。','',
           '|配置|BLEU (0–100)|ROUGE-1 F1|ROUGE-2 F1|ROUGE-L F1|失败|','|---|---:|---:|---:|---:|---:|']
    for name,m in report['conditions'].items():
        lines.append(f"|{name}|{m['sacrebleu']:.2f}|{m['rouge1_f1']:.3f}|{m['rouge2_f1']:.3f}|{m['rougeL_f1']:.3f}|{m['failures']}|")
    lines+=['','PD/PC使用相同原检查点，分别直接回答/操作链；RC/MC为真实重放/合成课程检查点的操作链。',
            '本项目未对这些适配器使用FeTaQA训练样本；基础模型预训练污染未知。',
            '所有失败按空假设计入；BLEU按语料重算；指标配置/版本、配对区间与原输出完整保存。',
            'PD和PC调用预算不同，不以其差值单独证明模块作用；所有配置均不据此自动采用。']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(root/'COMPLETE').write_text('automatic freeform transfer diagnostic complete\n')
    print('FETAQA_TRANSFER_COMPLETE',flush=True)


if __name__=='__main__':main()
