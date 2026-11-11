"""Fixed 32B teacher: dependency/evidence/repair development ablations.

Tests existing executable-constraint kernel, not the separate question-contract
prototype or a claim of novel semantic completeness. No labels enter prompts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

from tableqa.cli import metadata
from tableqa.data import read_jsonl,write_jsonl,digest
from tableqa.models import make_model
from tableqa.inference import predict
from tableqa.experiments import settings
from tableqa.analysis import summarize_run,compare_runs
from tableqa.reporting import freeform_metrics,compare_freeform


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class EvidenceExposureCounter:
    def __init__(self, model):
        self.model = model
        self.exposed_calls = 0

    def generate(self, messages):
        if json.loads(messages[1]["content"]).get("original_evidence_cells"):
            self.exposed_calls += 1
        return self.model.generate(messages)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','run']);parser.add_argument('output');args=parser.parse_args();root=Path(args.output)
    if args.action=='prepare':
        if root.exists():raise ValueError('Preserve frozen teacher protocol')
        rows=read_jsonl('runs/joint-core-20261107/prepared/samples.jsonl')
        short=[s for s in rows if s['dataset']!='fetaqa'];freeform=sorted([s for s in rows if s['dataset']=='fetaqa'],key=lambda s:s['id'])
        random.Random(2040).shuffle(freeform);samples=short+freeform[:20];random.Random(2041).shuffle(samples)
        if len(samples)!=60 or any(s['split']!='validation' for s in samples):raise ValueError('Expected 20/task development cases')
        conditions={'full':{},'no_dependency':{'dependency':False},'no_evidence':{'evidence':False},
                    'no_repair':{'max_repairs':0},'no_dependency_evidence':{'dependency':False,'evidence':False}}
        config={'model':str(Path('models/Qwen2.5-32B-Instruct').resolve()),'backend':'local',
                'max_input_tokens':8192,'max_new_tokens':512,'device_map':'balanced','seed':42}
        protocol={'development_only':True,'formal_test_read':False,'previously_used_development':True,
            'samples_sha256':digest(samples),'n':60,'per_task':20,'conditions':conditions,'model_config':config,
            'settings':{'mode':'B5','max_block':4,'max_operations':8,'max_repairs':1},
            'same_teacher_weights_and_interface':True,'greedy':True,'no_cache':True,'no_adapters':True,
            'latency_comparable':False,'automatic_adoption':False,'human_factual_consistency':None,
            'question_contract_prototype_active':False,'training_or_trajectory_approval':False,
            'limits':['Existing dependency check is block-local program consistency, not full question semantics.',
                      'Ablation of an existing kernel is not evidence that its components are novel.',
                      '60 previously used development cases; no formal test/SOTA or human factuality claim.']}
        root.mkdir(parents=True);write_jsonl(root/'samples.jsonl',samples);save(root/'protocol.json',protocol);save(root/'run.json',metadata(protocol));print('TEACHER_ABLATION_FROZEN',digest(samples),flush=True);return
    if (root/'STARTED').exists():raise ValueError('Preserve existing teacher attempt; inspect before recovery')
    protocol=load(root/'protocol.json');samples=read_jsonl(root/'samples.jsonl')
    if digest(samples)!=protocol['samples_sha256'] or metadata({})['source_hash']!=load(root/'run.json')['source_hash']:raise ValueError('Frozen code/input changed')
    if not Path('runs/joint-core-20261107/QUEUE_COMPLETE').is_file():raise ValueError('Student queue must release GPUs first')
    import torch
    if torch.cuda.device_count()!=2 or any(torch.cuda.mem_get_info(i)[0]<40*1024**3 for i in [0,1]):
        raise RuntimeError('Two idle A40s with >=40 GiB free each are required; do not stop other jobs')
    (root/'STARTED').write_text('fixed teacher ablations started\n');model=make_model(protocol['model_config'])
    if getattr(model.model,'peft_config',None):raise ValueError('Teacher must not load an adapter')
    outputs={name:[] for name in protocol['conditions']};names=list(outputs)
    evidence_study = protocol.get("study") == "readable_evidence"
    if evidence_study:
        model = EvidenceExposureCounter(model)
    sample_count = len(samples)
    for i,sample in enumerate(samples):
        order=names[i%len(names):]+names[:i%len(names)]
        for name in order:
            config={**protocol['settings'],**protocol['conditions'][name]};mode=config.pop('mode')
            if evidence_study:
                model.exposed_calls = 0
            row=predict(sample,model,settings(mode,**config));row['metrics']['latency_comparable']=False
            if evidence_study:
                row['metrics']['readable_evidence_calls'] = model.exposed_calls
            outputs[name].append(row);write_jsonl(root/f'{name}-predictions.jsonl',outputs[name])
            save(root/'progress.json',{'counts':{k:len(v) for k,v in outputs.items()},'expected_each':sample_count})
            print('TEACHER_ABLATION',name,i+1,sample_count,row['status'],flush=True)
    scripts=Path(__file__).resolve().parent;official={};report={'development_only':True,'automatic_adoption':False,'human_factual_consistency':None,'conditions':{},'comparisons':{}}
    for name,rows in outputs.items():
        subprocess.run([sys.executable,str(scripts/'evaluate_wtq_official.py'),str(root/'samples.jsonl'),str(root/f'{name}-predictions.jsonl'),str(root/f'wtq-{name}'),
            '--evaluator','data/raw/wikitq-official/WikiTableQuestions-master/evaluator.py','--tagged','data/raw/wikitq-official/WikiTableQuestions-master/tagged/data'],check=True)
        official[name]=load(root/f'wtq-{name}/scores.json')
        report['conditions'][name]={'task_summary':summarize_run(samples,rows,official[name]),'freeform':freeform_metrics(rows,samples)}
    baseline = names[0]
    for name in names[1:]:
        report['comparisons'][baseline+'-vs-'+name]={'task':compare_runs(samples,outputs[baseline],outputs[name],official[baseline],official[name]),
            'freeform':compare_freeform(outputs[baseline],outputs[name],samples)}
    if evidence_study:
        report['readable_evidence_exposure'] = {
            name: {"samples": sum(r['metrics']['readable_evidence_calls'] > 0 for r in rows),
                   "calls": sum(r['metrics']['readable_evidence_calls'] for r in rows)}
            for name, rows in outputs.items()
        }
    save(root/'report.json',report)
    title = '32B教师证据表示对照（开发）' if evidence_study else '32B教师既有约束内核消融（开发）'
    cohort = (f'{sample_count}题。按历史提示是否有机会注入可读证据选样。各组使用同一教师和操作块接口。'
              if evidence_study else '20题/任务。各组使用同一教师和操作块接口。')
    lines=[f'# {title}','',cohort,'',
        '|配置|WTQ官方指称准确率|TabFact准确率|FeTaQA BLEU|ROUGE-1|ROUGE-2|ROUGE-L|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,value in report['conditions'].items():
        scores={r['dataset']:r['observed_score'] for r in value['task_summary']['rows']};ft=value['freeform']
        lines.append(f"|{name}|{scores['wikitq']:.3f}|{scores['tabfact']:.3f}|{ft['sacrebleu']:.3f}|{ft['rouge1_f1']:.3f}|{ft['rouge2_f1']:.3f}|{ft['rougeL_f1']:.3f}|")
    lines+=['','失败、修复与成本完整保留；配对区间见report.json。不是正式test、人工事实评价或效率基准。']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(root/'completion-audit.json',{'passed':True,'predictions':sample_count*len(names),'scope':'coverage and evaluator consistency, not semantic approval',
        'files_sha256':{name:sha(root/name) for name in ['protocol.json','run.json','samples.jsonl','report.json','report.md']+[n+'-predictions.jsonl' for n in names]}})
    (root/'COMPLETE').write_text('fixed teacher module ablations complete\n');print('TEACHER_ABLATION_COMPLETE',flush=True)


if __name__=='__main__':main()
