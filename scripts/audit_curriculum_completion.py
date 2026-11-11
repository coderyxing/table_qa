# 脚本用途：检查课程训练的实际日程、优化步数、记录身份和来源覆盖是否符合冻结协议。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Post-run audit of actual schedules, optimizer steps, identities and coverage."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from tableqa.data import read_jsonl, digest


# 解析当前脚本参数，按模块说明执行本次冻结阶段并保存产物。
# 相关调用：out.write_text、(stage / 'protocol.json').read_text、read_jsonl、(stage / f'{name}-config.json').read_text。
# 异常约束：Completion audit already exists
def main():
    # 定义本入口的参数解析器；位置参数、默认值和允许取值由相邻 add_argument 明确声明。
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');args=p.parse_args()
    root=Path(args.root);stage=root/'prepared';out=root/'completion-audit.json'
    if out.exists():raise ValueError('Completion audit already exists')
    if not (root/'COMPLETE').exists():raise ValueError('Training/evaluation is incomplete')
    protocol=json.loads((stage/'protocol.json').read_text());audit={'development_only':True,'conditions':{}}
    schedules={}
    for name in ['R','M']:
        training=root/f'training-{name}';config=json.loads((stage/f'{name}-config.json').read_text())
        # 读取结构化输入：stage / f'{name}-records.jsonl'，保留原有样本身份字段。
        records=read_jsonl(stage/f'{name}-records.jsonl')
        schedules[name]=records
        if digest(records)!=config['frozen_records_sha256']:raise ValueError('Frozen records changed')
        stats=json.loads((training/'statistics.json').read_text())
        exposure=json.loads((training/'adapter'/'training_exposure.json').read_text())
        history=json.loads((training/'adapter'/'training_history.json').read_text())
        for row in history:
            for key in ['loss','grad_norm','train_loss','train_runtime']:
                if key in row and not math.isfinite(row[key]):
                    raise ValueError('Non-finite training history value')
        expected_steps=len(records)//(config['micro_batch']*config['gradient_accumulation_steps'])
        actual_steps=max(h.get('step',0) for h in history)
        if actual_steps!=expected_steps:raise ValueError('Unexpected actual optimizer steps')
        if stats['padding_records'] or stats['dropped_budget_records'] or stats['records_after_alignment']!=len(records):
            raise ValueError('Frozen training changed record exposure')
        if stats['supervised_tokens_per_epoch']!=config['frozen_supervised_tokens_per_epoch']:
            raise ValueError('Actual supervised token total changed')
        if sum(r['record_exposures'] for r in exposure['per_source'])!=len(records):
            raise ValueError('Actual source exposure does not cover the frozen schedule')
        if sum(r['target_tokens'] for r in exposure['per_source'])!=stats['supervised_tokens_per_epoch']:
            raise ValueError('Exposure and target budgets disagree')
        last=history[-1]
        labels={(r['dataset'],r['id']):r.get('sampling_label') for r in records}
        strata=defaultdict(lambda:{'record_exposures':0,'target_tokens':0})
        for row in exposure['per_source']:
            key=(row['dataset'],row['kind'],labels[(row['dataset'],row['id'])] if row['dataset']=='tabfact' else None)
            strata[key]['record_exposures']+=row['record_exposures']
            strata[key]['target_tokens']+=row['target_tokens']
        audit['conditions'][name]={'optimizer_steps':actual_steps,'records':len(records),
            'actual_target_tokens':stats['supervised_tokens_per_epoch'],
            'actual_input_tokens':stats['input_tokens_per_epoch'],
            'actual_source_samples':stats['source_samples_after_alignment'],
            'padding_records':stats['padding_records'],'dropped_records':stats['dropped_budget_records'],
            'target_by_kind':stats['target_tokens_by_kind'],
            'trainer_runtime_seconds':last.get('train_runtime'),'trainer_loss':last.get('train_loss'),
            'peak_allocated_bytes':stats['peak_allocated_bytes'],
            'adapter_sha256':hashlib.sha256((training/'adapter'/'adapter_model.safetensors').read_bytes()).hexdigest(),
            'strata':[{'dataset':key[0],'kind':key[1],'label':key[2],**values} for key,values in sorted(strata.items())]}
    a,b=[audit['conditions'][n] for n in ['R','M']]
    if (a['optimizer_steps'],a['records'],a['target_by_kind'])!=(b['optimizer_steps'],b['records'],b['target_by_kind']):
        raise ValueError('Actual controlled budgets differ')
    if protocol['matching']['matching_level']=='task_kind_label' and a['strata']!=b['strata']:
        raise ValueError('Per-task/kind/label exposure differs')
    common=0
    for left,right in zip(schedules['R'],schedules['M']):
        if left['frozen_slot']!=right['frozen_slot'] or left['schedule_part']!=right['schedule_part']:
            raise ValueError('Control/treatment slot layout changed')
        for row in [left,right]:
            if row['messages'][0]['content']!=protocol['same_system_prompt']:
                raise ValueError('Training system prompts differ')
        if left['schedule_part']=='common':
            if any(left[k]!=right[k] for k in ['id','dataset','messages','target']):
                raise ValueError('Common training record content differs')
            common+=1
    audit['identical_common_slot_records']=common
    audit['prediction_files']={}
    for name in ['P','R','M']:
        for cohort,filename in [('development','development-samples.jsonl'),('synthetic','synthetic-probe.jsonl')]:
            # 读取结构化输入：stage / filename，保留原有样本身份字段。
            samples=read_jsonl(stage/filename);predictions=read_jsonl(root/'evaluation'/name/f'{cohort}-predictions.jsonl')
            key=lambda r:(r['dataset'],r['id'])
            if len(predictions)!=len(samples) or {key(p) for p in predictions}!={key(s) for s in samples}:
                raise ValueError('Evaluation coverage differs; failures cannot be omitted')
            audit['prediction_files'][name+'-'+cohort]={'records':len(predictions),'sha256_structured':digest(predictions)}
    audit['matching_level']=protocol['matching']['matching_level']
    audit['passed']=True;audit['equal_gpu_compute_claimed']=False
    out.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit['conditions']),flush=True)


if __name__=='__main__':main()
