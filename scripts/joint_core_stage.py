"""Frozen corrected-training B0/B2/B5 joint development experiment.

Same 172 weakly checked WTQ/TabFact sources + 512 official FeTaQA answers.
FeTaQA always uses B0, not unreviewed operation supervision. Same-source control,
not equal-token or an isolated state-dependent partition ablation.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from tableqa.cli import metadata
from tableqa.data import read_jsonl,write_jsonl,digest,table_fingerprint
from tableqa.distillation import build_records
from tableqa.training import train,prepare_training_data
from tableqa.models import make_model
from tableqa.inference import predict
from tableqa.experiments import settings
from tableqa.analysis import summarize_run,compare_runs
from tableqa.reporting import freeform_metrics,compare_freeform


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','B0','B2','B5','report']);p.add_argument('stage');args=p.parse_args();root=Path(args.stage)
    if args.action=='prepare':
        if root.exists():raise ValueError('Preserve existing experiment')
        real=Path('runs/accuracy-training-pool');ft=Path('runs/fetaqa-answer-supervision-v2/results')
        sources=read_jsonl(real/'samples.jsonl');preds=read_jsonl(real/'predictions.jsonl')
        if len(sources)!=172 or len(preds)!=172:raise ValueError('Unexpected teacher pool')
        lookup={(r['dataset'],r['id']):r for r in preds}
        freeform=read_jsonl(ft/'train-samples.jsonl');sources+=freeform
        if len(freeform)!=512 or any(s['split']!='train' for s in sources):raise ValueError('Train-only sources required')
        validation=read_jsonl('runs/curriculum-student-v1/prepared/development-samples.jsonl')+read_jsonl('runs/fetaqa-answer-development100-v1/samples.jsonl')
        train_fps={table_fingerprint(s['table']) for s in sources}
        if any(s['split']!='validation' or table_fingerprint(s['table']) in train_fps for s in validation):raise ValueError('Validation overlap/split conflict')
        model=str(Path('models/Qwen2.5-3B-Instruct').resolve())
        config={'model':model,'seed':42,'sampling_policy':'coverage','loss_balance':'task_label',
                'max_seq_length':8192,'lora_rank':32,'lora_alpha':64,'learning_rate':1e-4,
                'answer_weight':1,'micro_batch':1,'gradient_accumulation_steps':16,'max_epochs':1}
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
        records={};preflight={}
        for mode in ['B0','B2','B5']:
            rows=[]
            for sample in sources:
                actual='B0' if sample['dataset']=='fetaqa' else mode
                operations=[] if actual=='B0' else lookup[(sample['dataset'],sample['id'])]['operations']
                group=build_records(sample,operations,actual,compression_policy='guarded_search' if actual=='B5' else 'legacy')
                for r in group:
                    if sample['dataset']=='tabfact':r['sampling_label']=int(sample['reference'])
                rows+=group
            _,stats=prepare_training_data(config,rows,tokenizer)
            if stats['source_samples_after_alignment']!=684:raise ValueError('Lost training source coverage')
            records[mode]=rows;preflight[mode]=stats
        protocol={'development_only':True,'single_seed_one_epoch':True,'formal_test_read':False,
            'evaluation_previously_used':True,'source_count':684,'source_ids_sha256':digest(sorted((s['dataset'],s['id']) for s in sources)),
            'sources_sha256':digest(sources),'validation_sha256':digest(validation),'validation_counts':{'wikitq':20,'tabfact':20,'fetaqa':100},
            'same_source_control':True,'equal_training_tokens':False,'equal_training_compute':False,
            'weak_teacher_process_semantics_unproven':True,'freeform_operation_gate_unchanged':True,
            'freeform_train_and_inference_mode':'B0 for ALL conditions; no FeTaQA operation claims',
            'fresh_adapters_from_same_base':True,'no_legacy_initial_adapter':True,'config':config,
            'records_sha256':{m:digest(r) for m,r in records.items()},
            'max_repairs':1,'max_operations':8,'inference_input_limit':8192,'inference_output_limit':512,
            'latency_comparable':False,'automatic_adoption':False,
            'limitations':['B2 vs B5 changes conservative compression/blocking, not isolated adaptive-state partition.',
                           'Task/label loss balancing retained; factor 2 applies when both supervision kinds exist.',
                           'One seed development stage; formal ablations/efficiency/multiple seeds still required.']}
        root.mkdir(parents=True);save(root/'protocol.json',protocol);save(root/'run.json',metadata(protocol));save(root/'preflight.json',preflight)
        write_jsonl(root/'sources.jsonl',sources);write_jsonl(root/'samples.jsonl',validation)
        for mode,rows in records.items():write_jsonl(root/f'{mode}-records.jsonl',rows)
        print('JOINT_CORE_FROZEN',json.dumps({m:{k:v[k] for k in ['records_after_alignment','supervised_tokens_per_epoch']} for m,v in preflight.items()}),flush=True);return
    protocol=load(root/'protocol.json');samples=read_jsonl(root/'samples.jsonl')
    if metadata({})['source_hash']!=load(root/'run.json')['source_hash'] or digest(samples)!=protocol['validation_sha256']:raise ValueError('Frozen code/input changed')
    if args.action!='report':
        name=args.action;out=root/name
        if out.exists():raise ValueError('Preserve existing condition; inspect before any recovery')
        if shutil.disk_usage('.').free<8*1024**3:raise ValueError('Keep 8 GiB free')
        records=read_jsonl(root/f'{name}-records.jsonl')
        if digest(records)!=protocol['records_sha256'][name]:raise ValueError('Records changed')
        out.mkdir();save(out/'run.json',metadata(protocol['config']))
        import torch
        statistics=train(protocol['config'],records,out/'adapter');save(out/'statistics.json',statistics)
        expected=load(root/'preflight.json')[name]
        for key in ['sampling_sequence_sha256','supervised_tokens_per_epoch','source_samples_after_alignment','records_after_alignment']:
            if statistics[key]!=expected[key]:raise ValueError('Actual training differs from preflight: '+key)
        if statistics['gradient_normalization']!='explicit-microbatch-mean-v1':raise ValueError('Uncorrected gradients')
        history=load(out/'adapter/training_history.json')
        if max(r.get('step',0) for r in history)!=statistics['records_after_alignment']//16:raise ValueError('Actual optimizer steps differ')
        (out/'TRAINED').write_text('fixed one epoch complete\n');gc.collect();torch.cuda.empty_cache()
        model=make_model({'model':protocol['config']['model'],'adapter':str((out/'adapter').resolve()),'max_input_tokens':8192,'max_new_tokens':512,'seed':42})
        if not getattr(model.model,'peft_config',None) or not model.model.active_adapters:raise ValueError('Adapter not active')
        save(out/'loaded-model.json',{'adapter_sha256':sha(out/'adapter/adapter_model.safetensors'),'active_adapters':model.model.active_adapters})
        predictions=[]
        for i,s in enumerate(samples):
            mode='B0' if s['dataset']=='fetaqa' else name
            prediction=predict(s,model,settings(mode,max_repairs=0 if mode=='B0' else 1))
            prediction['metrics']['latency_comparable']=False;predictions.append(prediction);write_jsonl(out/'predictions.jsonl',predictions)
            save(out/'progress.json',{'completed':i+1,'expected':len(samples)})
            print('JOINT_CORE',name,i+1,len(samples),prediction['status'],flush=True)
        subprocess.run([sys.executable,str(Path(__file__).parent/'evaluate_wtq_official.py'),str(root/'samples.jsonl'),str(out/'predictions.jsonl'),str(out/'wtq'),
            '--evaluator','data/raw/wikitq-official/WikiTableQuestions-master/evaluator.py','--tagged','data/raw/wikitq-official/WikiTableQuestions-master/tagged/data'],check=True)
        scores=load(out/'wtq/scores.json');save(out/'summary.json',{'task_summary':summarize_run(samples,predictions,scores),'freeform':freeform_metrics(predictions,samples)})
        (out/'COMPLETE').write_text('condition training and development evaluation complete\n');return
    if not all((root/m/'COMPLETE').exists() for m in ['B0','B2','B5']):raise ValueError('Wait for every condition')
    if (root/'report.json').exists():raise ValueError('Preserve final report')
    outputs={m:read_jsonl(root/m/'predictions.jsonl') for m in ['B0','B2','B5']};official={m:load(root/m/'wtq/scores.json') for m in outputs}
    report={'development_only':True,'automatic_adoption':False,'same_sources_not_equal_tokens':True,
            'conditions':{m:load(root/m/'summary.json') for m in outputs},'comparisons':{}}
    for a,b in [('B0','B2'),('B0','B5'),('B2','B5')]:
        report['comparisons'][a+'-vs-'+b]={'task':compare_runs(samples,outputs[a],outputs[b],official[a],official[b]),
            'freeform':compare_freeform(outputs[a],outputs[b],samples)}
    save(root/'report.json',report);(root/'COMPLETE').write_text('all controlled development conditions complete\n');print('JOINT_CORE_COMPLETE',flush=True)


if __name__=='__main__':main()
