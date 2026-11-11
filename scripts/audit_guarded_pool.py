"""Offline mechanism audit on real teacher trajectories, no synthetic padding."""
import argparse
from collections import Counter
import json
from pathlib import Path
from tableqa.core import TableState,run_plan
from tableqa.data import read_jsonl,write_jsonl,digest
from tableqa.optimization import optimize_operations,state_cost_partition,effective_fixed_blocks


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output');args=p.parse_args();root=Path(args.output)
    if root.exists():raise ValueError('Preserve previous mechanism audit')
    source=Path('runs/accuracy-training-pool');samples=read_jsonl(source/'samples.jsonl');predictions=read_jsonl(source/'predictions.jsonl')
    lookup={(r['dataset'],r['id']):r for r in predictions}
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('models/Qwen2.5-3B-Instruct',local_files_only=True)
    root.mkdir();details=[]
    for i,sample in enumerate(samples):
        if sample['split']!='train' or sample['dataset']=='fetaqa':raise ValueError('Only the real short/verification training pool')
        operations=lookup[(sample['dataset'],sample['id'])]['operations'];state=TableState.from_table(sample['table'])
        compressed,certificate=optimize_operations(state,operations,preserve_kinds=('filter','sort','select','group_aggregate'))
        if run_plan(state,operations)[0].signature()!=run_plan(state,compressed)[0].signature():raise ValueError('Replay mismatch')
        sensitive,sa=state_cost_partition(sample,compressed,tokenizer,max_size=4,max_length=8192)
        blind,ba=state_cost_partition(sample,compressed,tokenizer,max_size=4,max_length=8192,state_sensitive=False)
        fixed=effective_fixed_blocks(compressed,4)
        details.append({'dataset':sample['dataset'],'id':sample['id'],'original_operations':len(operations),'compressed_operations':len(compressed),
            'operation_types':dict(Counter(op['op'] for op in operations)),
            'state_vs_blind_boundaries_different':sensitive!=blind,'state_vs_fixed_boundaries_different':sensitive!=fixed,
            'state_proxy_cost':sa['cost'],'fixed_proxy_cost':sa['fixed_control_cost'],
            'blind_path_actual_proxy_cost':sum(e['input_tokens']+e['output_tokens']+128 for e in ba['selected_edges']),
            'fixed_boundaries':[len(g) for g in fixed],'state_boundaries':[len(g) for g in sensitive],
            'feedback_flags':sum(bool(op.get('requires_feedback')) for op in operations),
            'search_certificate':certificate})
        write_jsonl(root/'per-source.jsonl',details)
        if (i+1)%20==0:print('GUARDED_POOL_AUDIT',i+1,len(samples),flush=True)
    summary={'offline_only':True,'no_semantic_correctness_claim':True,'uses_future_teacher_chain_offline':True,
        'n':len(samples),'samples_sha256':digest(samples),'predictions_sha256':digest(predictions),
        'original_operations':sum(r['original_operations'] for r in details),'compressed_operations':sum(r['compressed_operations'] for r in details),
        'chain_length_histogram':dict(Counter(r['original_operations'] for r in details)),
        'state_vs_blind_changed_sources':sum(r['state_vs_blind_boundaries_different'] for r in details),
        'state_vs_fixed_changed_sources':sum(r['state_vs_fixed_boundaries_different'] for r in details),
        'state_proxy_cost':sum(r['state_proxy_cost'] for r in details),
        'fixed_proxy_cost':sum(r['fixed_proxy_cost'] for r in details),
        'blind_path_actual_proxy_cost':sum(r['blind_path_actual_proxy_cost'] for r in details),
        'feedback_flags':sum(r['feedback_flags'] for r in details),
        'units':'token proxy, not GPU seconds or online oracle partitioning'}
    (root/'report.json').write_text(json.dumps(summary,indent=2));(root/'COMPLETE').write_text('offline guarded mechanism audit complete\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
