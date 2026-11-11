# 脚本用途：在训练划分采集新的真实教师轨迹，保存重放、弱筛选和来源记录。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Acquire 512 new train-only traces after the fixed teacher ablations.

Half per task use a question-only complexity proxy; half are randomly drawn
from remaining tables. Proxy strata are not claims about actual chain length.
Replay/official agreement is weak screening, never human semantic approval.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import subprocess
import sys

from tableqa.cli import metadata
from tableqa.core import TableState
from tableqa.data import read_jsonl,write_jsonl,digest,table_fingerprint
from tableqa.inference import context,system_prompt,predict
from tableqa.experiments import settings
from tableqa.models import make_model
from tableqa.quality import audit_run_trajectories


# 读取指定结构化产物，交由当前阶段核验来源和配置。
# 输入：p。
# 相关调用：Path(p).read_text。
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
# 创建输出目录并保存当前阶段的结构化产物。
# 输入：p；v。
# 相关调用：Path(p).write_text。
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')


# 从题面和表结构计算采样用复杂度特征。
# 输入：question（原题文本）。
def complexity(question):
    return len(set(re.findall(r'\b(?:and|after|before|than|most|least|total|average|difference|both|between|combined|more|less)\b',question.casefold())))


# 解析当前脚本参数，按模块说明执行本次冻结阶段并保存产物。
# 相关调用：read_jsonl、(root / 'STARTED').write_text、write_jsonl、(root / 'COMPLETE').write_text。
# 异常约束：Preserve existing acquisition attempt
def main():
    # 定义本入口的参数解析器；位置参数、默认值和允许取值由相邻 add_argument 明确声明。
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run']);p.add_argument('output');args=p.parse_args();root=Path(args.output)
    if args.action=='prepare':
        if root.exists():raise ValueError('Preserve the frozen acquisition cohort')
        # 读取结构化输入：'data/normalized/joint_train_clean.jsonl'，保留原有样本身份字段。
        train=read_jsonl('data/normalized/joint_train_clean.jsonl');old=read_jsonl('runs/accuracy-training-pool/samples.jsonl')
        if any(s['split']!='train' for s in train):raise ValueError('Train-only acquisition')
        blocked={table_fingerprint(s['table']) for s in old}
        # 读取结构化输入：'data/normalized/joint_validation_clean.jsonl'，保留原有样本身份字段。
        blocked.update(table_fingerprint(s['table']) for s in read_jsonl('data/normalized/joint_validation_clean.jsonl'))
        model=str(Path('models/Qwen2.5-32B-Instruct').resolve())
        from transformers import AutoTokenizer
        # 使用本阶段指定的 tokenizer；预算、提示模板与监督掩码应沿用同一版本。
        tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True);selected=[];skips=Counter();rng=random.Random(2042)
        for task in ['wikitq','tabfact']:
            candidates=sorted([s for s in train if s['dataset']==task],key=lambda s:s['id']);rng.shuffle(candidates)
            per_task=[]
            for stratum,goal in [('complex_proxy',128),('remaining_random',256)]:
                for sample in candidates:
                    if len(per_task)>=goal:break
                    if stratum=='complex_proxy' and complexity(sample['question'])<2:continue
                    fp=table_fingerprint(sample['table'])
                    if fp in blocked:continue
                    messages=[{'role':'system','content':system_prompt('B5',True,True,True)},
                        {'role':'user','content':context(sample,TableState.from_table(sample['table']),[],'B5',max_block=4)}]
                    if len(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))>8192:
                        skips[task+':initial_input_overlength']+=1;continue
                    per_task.append({**sample,'acquisition_stratum':stratum,'question_complexity_proxy':complexity(sample['question'])});blocked.add(fp)
            if len(per_task)!=256:raise ValueError('Insufficient distinct fitting training tables')
            selected+=per_task
        rng.shuffle(selected)
        config={'model':model,'backend':'local','max_input_tokens':8192,'max_new_tokens':512,'device_map':'balanced','seed':42}
        options={'mode':'B5','max_block':4,'max_operations':12,'max_repairs':2,'typed_quantities':True,'semantic_extract':True,'tool_demonstrations':True}
        protocol={'train_only':True,'no_validation_or_test_scores_for_selection':True,'formal_test_read':False,
            'n':512,'per_task':256,'selection_seed':2042,'samples_sha256':digest(selected),'model_config':config,'settings':options,
            'selection_strata':dict(Counter(s['dataset']+':'+s['acquisition_stratum'] for s in selected)),
            'selection_skips':dict(skips),'no_exact_table_overlap_with_previous_172_or_clean_validation':True,
            'complexity_proxy_not_actual_chain_length':True,'weak_screening_not_semantic_certification':True,
            'no_fetaqa_operation_supervision':True,'automatic_model_adoption':False,
            'limits':['Only fitting initial inputs selected; later context failures remain in outputs.',
                      'Operation budget/tools differ from the prior ablation; not a controlled accuracy comparison.',
                      'Official answer agreement and replay can accept semantically wrong processes.']}
        # 保存逐条结构化产物到 root / 'samples.jsonl'，供后续重放或审核读取。
        root.mkdir(parents=True);write_jsonl(root/'samples.jsonl',selected);save(root/'protocol.json',protocol);save(root/'run.json',metadata(protocol));print('REAL_ACQUISITION_FROZEN',protocol['selection_strata'],flush=True);return
    if (root/'STARTED').exists():raise ValueError('Preserve existing acquisition attempt')
    if not Path('runs/teacher-ablation-20261107/QUEUE_COMPLETE').exists():raise ValueError('Teacher ablation must finish first')
    # 读取结构化输入：root / 'samples.jsonl'，保留原有样本身份字段。
    protocol=load(root/'protocol.json');samples=read_jsonl(root/'samples.jsonl')
    if digest(samples)!=protocol['samples_sha256'] or metadata({})['source_hash']!=load(root/'run.json')['source_hash']:raise ValueError('Frozen input/code changed')
    import torch
    if torch.cuda.device_count()!=2 or any(torch.cuda.mem_get_info(i)[0]<40*1024**3 for i in [0,1]):raise RuntimeError('Require two idle GPUs; do not stop other tasks')
    # 按已读取配置构造生成后端，实际模型身份和预算由该配置登记。
    (root/'STARTED').write_text('new real training acquisition started\n');model=make_model(protocol['model_config']);outputs=[];consecutive_oom=0
    if getattr(model.model,'peft_config',None):raise ValueError('Unmodified teacher required')
    for i,sample in enumerate(samples):
        # 对当前样本运行完整推理入口，返回值包含答案、状态、错误和调用成本。
        options=dict(protocol['settings']);mode=options.pop('mode');row=predict(sample,model,settings(mode,**options));row['metrics']['latency_comparable']=False
        # 保存逐条结构化产物到 root / 'predictions.jsonl'，供后续重放或审核读取。
        outputs.append(row);write_jsonl(root/'predictions.jsonl',outputs);save(root/'progress.json',{'completed':i+1,'expected':512})
        print('REAL_ACQUISITION',i+1,512,row['status'],flush=True)
        oom='out of memory' in str(row.get('failure')).lower();consecutive_oom=consecutive_oom+1 if oom else 0
        if oom:torch.cuda.empty_cache()
        if consecutive_oom>=3:raise RuntimeError('Repeated GPU memory failures; preserve outputs and inspect')
    subprocess.run([sys.executable,str(Path(__file__).parent/'evaluate_wtq_official.py'),str(root/'samples.jsonl'),str(root/'predictions.jsonl'),str(root/'wtq'),
        '--evaluator','data/raw/wikitq-official/WikiTableQuestions-master/evaluator.py','--tagged','data/raw/wikitq-official/WikiTableQuestions-master/tagged/data'],check=True)
    # 保存逐条结构化产物到 root / 'trajectory-audits.jsonl'，供后续重放或审核读取。
    audits=audit_run_trajectories(samples,outputs,load(root/'wtq/scores.json'));write_jsonl(root/'trajectory-audits.jsonl',audits)
    accepted={(r['dataset'],r['id']) for r in audits if r['decision'].startswith('accepted_')}
    # 保存逐条结构化产物到 root / 'weak-accepted-samples.jsonl'，供后续重放或审核读取。
    write_jsonl(root/'weak-accepted-samples.jsonl',[s for s in samples if (s['dataset'],s['id']) in accepted])
    # 保存逐条结构化产物到 root / 'weak-accepted-predictions.jsonl'，供后续重放或审核读取。
    write_jsonl(root/'weak-accepted-predictions.jsonl',[r for r in outputs if (r['dataset'],r['id']) in accepted])
    report={'train_only':True,'n':512,'accepted_weak_sources':len(accepted),'decision_counts':dict(Counter(a['decision'] for a in audits)),
        'accepted_by_dataset':dict(Counter(k[0] for k in accepted)),
        'accepted_chain_lengths':dict(Counter(len(r['operations']) for r in outputs if (r['dataset'],r['id']) in accepted)),
        'accepted_feedback_flags':sum(bool(op.get('requires_feedback')) for r in outputs if (r['dataset'],r['id']) in accepted for op in r['operations']),
        'human_approved':0,'semantic_correctness_proven':False,'automatic_training_or_adoption':False}
    save(root/'report.json',report);(root/'COMPLETE').write_text('new train-only acquisition and weak screening complete\n');print('REAL_ACQUISITION_COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':main()
