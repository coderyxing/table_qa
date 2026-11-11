# 脚本用途：在同一接受池比较压缩策略与分块策略，区分代理收益和实际运行结果。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Audit a compression x partition design on the exact accepted training pool."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
from transformers import AutoTokenizer
from tableqa.analysis import validate_run,index_records
from tableqa.core import TableState
from tableqa.data import read_jsonl,write_jsonl,digest
from tableqa.distillation import compress,build_records
from tableqa.optimization import optimize_operations,state_cost_partition,effective_fixed_blocks
from tableqa.training import encode_record

# 定义本入口的参数解析器；位置参数、默认值和允许取值由相邻 add_argument 明确声明。
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',default='runs/optimization-audit')
p.add_argument('--samples',default='runs/medium-development/samples.jsonl')
p.add_argument('--predictions',default='runs/medium-development/A4-predictions.jsonl')
p.add_argument('--accepted-ids',default='runs/medium-development/students/accepted-samples.jsonl')
p.add_argument('--tokenizer',default='models/Qwen2.5-3B-Instruct')
p.add_argument('--max-block',type=int,default=4);p.add_argument('--max-length',type=int,default=4096)
# 解析本次命令的实际参数，后续样本路径和实验开关均由此确定。
p.add_argument('--max-target-tokens',type=int,default=512);args=p.parse_args()
root=Path(args.output)
if root.exists():raise ValueError('Output exists; preserve it and use a new directory')
root.mkdir(parents=True)
# 读取结构化输入：args.samples，保留原有样本身份字段。
samples=read_jsonl(args.samples);predictions=read_jsonl(args.predictions)
refs,preds,_=validate_run(samples,predictions)
# 读取结构化输入：args.accepted_ids，保留原有样本身份字段。
accepted=read_jsonl(args.accepted_ids);keys=list(index_records(accepted,'accepted source'))
if set(keys)-set(refs) or not keys:raise ValueError('Empty or unknown accepted source pool')
# 使用本阶段指定的 tokenizer；预算、提示模板与监督掩码应沿用同一版本。
tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
partition_config={'max_size':args.max_block,'max_length':args.max_length,'max_output_tokens':args.max_target_tokens}
variants={name:[] for name in ['legacy_fixed','search_fixed','legacy_state','legacy_blind','search_state','search_blind']}
details=[];start=time.monotonic()
for key in keys:
    sample,pred=refs[key],preds[key]
    if sample['split']!='train' or pred['status']!='ok':raise ValueError('Executable source training trajectories required')
    original=pred['operations'];state=TableState.from_table(sample['table'])
    legacy=compress(state,original);searched,search_report=optimize_operations(state,original)
    legacy_state,ls=state_cost_partition(sample,legacy,tokenizer,**partition_config)
    legacy_blind,lb=state_cost_partition(sample,legacy,tokenizer,state_sensitive=False,**partition_config)
    search_state,ss=state_cost_partition(sample,searched,tokenizer,**partition_config)
    search_blind,sb=state_cost_partition(sample,searched,tokenizer,state_sensitive=False,**partition_config)
    plans={'legacy_fixed':(legacy,effective_fixed_blocks(legacy,args.max_block)),
           'search_fixed':(searched,effective_fixed_blocks(searched,args.max_block)),
           'legacy_state':(legacy,legacy_state),'legacy_blind':(legacy,legacy_blind),
           'search_state':(searched,search_state),'search_blind':(searched,search_blind)}
    source_records={}
    for name,(ops,groups) in plans.items():
        records=build_records(sample,original,'B5',args.max_block,prepared_operations=ops,partition=groups)
        for r in records:encode_record(r,tokenizer,args.max_length)
        variants[name]+=records;source_records[name]=records
    # Exact raw control equality before any student retraining.
    old=build_records(sample,original,'B5',args.max_block)
    if source_records['legacy_fixed']!=old:raise ValueError('Audit changed the frozen fixed control')
    details.append({'dataset':key[0],'id':key[1],'search':search_report,'legacy_state':ls,'legacy_blind':lb,'search_state':ss,'search_blind':sb,
        'original_operations':len(original),'legacy_operations':len(legacy),'search_operations':len(searched),
        'record_hashes':{name:digest(r) for name,r in source_records.items()},
        'record_counts':{name:len(r) for name,r in source_records.items()},
        'fixed_boundaries':[len(g) for g in plans['search_fixed'][1]]})
    # 保存逐条结构化产物到 root / 'per-source.jsonl'，供后续重放或审核读取。
    write_jsonl(root/'per-source.jsonl',details)
    (root/'progress.json').write_text(json.dumps({'completed':len(details),'expected':len(keys),'seconds':time.monotonic()-start}))
    print('OPTIMIZATION_AUDIT_PROGRESS',len(details),len(keys),flush=True)
summary={'development_only':True,'n':len(keys),'accepted_ids_sha256':digest(accepted),'variants':{},
         'samples_sha256':digest(samples),'predictions_sha256':digest(predictions),'partition_config':partition_config,
         'original_operations':sum(r['original_operations'] for r in details),
         'legacy_operations':sum(r['legacy_operations'] for r in details),
         'search_operations':sum(r['search_operations'] for r in details),
         'certified_search_count':sum(r['search']['optimal_in_deletion_family'] for r in details),
         'state_vs_fixed_changed_sources':sum(r['record_hashes']['search_state']!=r['record_hashes']['search_fixed'] for r in details),
         'state_vs_blind_changed_sources':sum(r['record_hashes']['search_state']!=r['record_hashes']['search_blind'] for r in details),
         'legacy_state_vs_blind_changed_sources':sum(r['record_hashes']['legacy_state']!=r['record_hashes']['legacy_blind'] for r in details),
         'state_proxy_cost':sum(r['search_state']['cost'] for r in details),
         'fixed_proxy_cost':sum(r['search_state']['fixed_control_cost'] for r in details),
         'legacy_state_proxy_cost':sum(r['legacy_state']['cost'] for r in details),
         'legacy_blind_path_actual_cost':sum(sum(e['input_tokens']+e['output_tokens']+128 for e in r['legacy_blind']['selected_edges']) for r in details),
         'search_blind_path_actual_cost':sum(sum(e['input_tokens']+e['output_tokens']+128 for e in r['search_blind']['selected_edges']) for r in details),
         'limitations':'Training-time replay/cost audit only. Fewer proxy tokens is not an online speed or task-quality result.'}
for name,records in variants.items():
    # 保存逐条结构化产物到 root / f'{name}-records.jsonl'，供后续重放或审核读取。
    write_jsonl(root/f'{name}-records.jsonl',records)
    target_tokens=sum(sum(v!=-100 for v in encode_record(r,tokenizer,args.max_length)['labels']) for r in records)
    summary['variants'][name]={'records':len(records),'record_sha256':digest(records),'raw_target_tokens':target_tokens,
        'changed_sources_vs_legacy_fixed':sum(r['record_hashes'][name]!=r['record_hashes']['legacy_fixed'] for r in details)}
(root/'report.json').write_text(json.dumps(summary,indent=2))
print('OPTIMIZATION_AUDIT_COMPLETE',json.dumps(summary),flush=True)
