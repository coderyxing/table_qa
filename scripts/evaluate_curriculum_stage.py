"""Same-prompt P/R/M evaluation on frozen development and synthetic instances."""
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from tableqa.analysis import compare_runs, summarize_run
from tableqa.cli import metadata
from tableqa.data import read_jsonl, write_jsonl, digest
from tableqa.evaluation import diagnostic_score, paired_bootstrap
from tableqa.experiments import settings
from tableqa.inference import predict
from tableqa.models import make_model
from tableqa.quantities import uses_quantities
from tableqa.semantic_columns import uses_extraction


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('stage')
    args=parser.parse_args(); stage=Path(args.stage).resolve(); root=stage.parent/'evaluation'
    if root.exists(): raise ValueError('Evaluation exists; preserve all outputs')
    protocol=json.loads((stage/'protocol.json').read_text())
    frozen=json.loads((stage/'run.json').read_text())
    if metadata({})['source_hash']!=frozen['source_hash']: raise ValueError('Evaluation module code changed')
    samples={'development':read_jsonl(stage/'development-samples.jsonl'),
             'synthetic':read_jsonl(stage/'synthetic-probe.jsonl')}
    if digest(samples['development'])!=protocol['development']['sha256'] or digest(samples['synthetic'])!=protocol['synthetic_probe']['sha256']:
        raise ValueError('Frozen evaluation inputs changed')
    if any(s['split']!='validation' for s in samples['development']): raise ValueError('Development split changed')
    cfg=json.loads((stage/'R-config.json').read_text()); initial=Path(cfg['initial_adapter'])
    if hashlib.sha256((initial/'adapter_model.safetensors').read_bytes()).hexdigest()!=protocol['training']['initial_adapter_sha256']:
        raise ValueError('Initial adapter changed')
    for condition in ['R','M']:
        if not (stage.parent/f'training-{condition}'/'COMPLETE').exists():
            raise ValueError('Both controlled training runs must complete first')
    root.mkdir(); (root/'run.json').write_text(json.dumps(metadata(protocol),indent=2))
    for kind,rows in samples.items(): write_jsonl(root/f'{kind}-samples.jsonl',rows)
    infer_settings=dict(protocol['inference_settings']); mode=infer_settings.pop('mode')
    outputs={}
    import torch
    for name in ['P','R','M']:
        adapter=initial if name=='P' else stage.parent/f'training-{name}'/'adapter'
        model_config={**protocol['model_config'],'adapter':str(adapter)}
        model=make_model(model_config); outputs[name]={}; out=root/name; out.mkdir()
        (out/'config.json').write_text(json.dumps({'model_config':model_config,'settings':protocol['inference_settings']},indent=2))
        for kind,rows in samples.items():
            predictions=[]
            with (out/f'{kind}-predictions.jsonl').open('w',encoding='utf-8') as stream:
                for i,sample in enumerate(rows):
                    result=predict(sample,model,settings(mode,**infer_settings))
                    result['metrics']['latency_comparable']=False
                    result['extraction_used']=uses_extraction(result['operations'])
                    result['quantity_used']=uses_quantities(result['operations'])
                    stream.write(json.dumps(result,ensure_ascii=False)+'\n');stream.flush();predictions.append(result)
                    (root/'progress.json').write_text(json.dumps({'condition':name,'cohort':kind,'completed':i+1,'expected':len(rows)}))
                    if (i+1)%5==0: print('EVALUATING',name,kind,i+1,len(rows),flush=True)
            outputs[name][kind]=predictions
        del model;gc.collect();torch.cuda.empty_cache()
    scripts=Path(__file__).resolve().parent
    for name in ['P','R','M']:
        subprocess.run([sys.executable,str(scripts/'evaluate_wtq_official.py'),
            str(root/'development-samples.jsonl'),str(root/name/'development-predictions.jsonl'),str(root/name/'wtq'),
            '--evaluator','data/raw/wikitq-official/WikiTableQuestions-master/evaluator.py',
            '--tagged','data/raw/wikitq-official/WikiTableQuestions-master/tagged/data'],check=True)
    report={'development_only':True,'not_independent_confirmation':True,'single_seed_one_epoch':True,
            'latency_comparable':False,'automatic_adoption':False,'development':{},'synthetic':{},'comparisons':{}}
    official={name:json.loads((root/name/'wtq'/'scores.json').read_text()) for name in outputs}
    for name,cohorts in outputs.items():
        report['development'][name]=summarize_run(samples['development'],cohorts['development'],official[name])
        synthetic=cohorts['synthetic'];refs=samples['synthetic']
        by_family={}
        for family in sorted({r['template_family'] for r in refs}):
            pairs=[(p,s) for p,s in zip(synthetic,refs) if s['template_family']==family]
            by_family[family]={'n':len(pairs),'accuracy':sum(diagnostic_score(p,s) for p,s in pairs)/len(pairs),
                              'execution_success':sum(p['status']=='ok' for p,_ in pairs),
                              'extraction_used':sum(p['extraction_used'] for p,_ in pairs),
                              'quantity_used':sum(p['quantity_used'] for p,_ in pairs)}
        report['synthetic'][name]={'not_benchmark_score':True,'n':len(refs),
            'accuracy':sum(diagnostic_score(p,s) for p,s in zip(synthetic,refs))/len(refs),'by_family':by_family}
    for a,b in [('P','R'),('R','M'),('P','M')]:
        report['comparisons'][a+'-vs-'+b]=compare_runs(samples['development'],outputs[a]['development'],outputs[b]['development'],official[a],official[b])
    (root/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    lines=['# 受控学生训练开发试验','',
        '一轮训练、一个种子；真实开发集已使用，合成样本是同模板新实例。不得作为正式测试或泛化结论。',
        '', '|配置|WikiTQ官方端口|TabFact准确率|合成技能准确率|', '|---|---:|---:|---:|']
    for name in ['P','R','M']:
        scores={r['dataset']:r['observed_score'] for r in report['development'][name]['rows']}
        lines.append(f"|{name}|{scores['wikitq']:.3f}|{scores['tabfact']:.3f}|{report['synthetic'][name]['accuracy']:.3f}|")
    lines += ['', 'P为原检查点；R为共同真实记录加长度匹配的真实重放；M为共同真实记录加合成材料。',
              'R/M实际目标token、记录数和监督类型匹配；输入token与来源组成不同，不声称同GPU计算量或单个创新模块的因果作用。',
              '所有失败、修复、原始输出及配对区间保留在report.json及各配置目录。未自动替换已有最佳系统。']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (stage.parent/'COMPLETE').write_text('controlled curriculum development experiment complete\n')
    print('CURRICULUM_STUDENT_STAGE_COMPLETE',flush=True)


if __name__=='__main__':main()
