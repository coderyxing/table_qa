"""Audit replay invariants and actual block boundaries on accepted training traces."""
import argparse
from collections import Counter
import json
from pathlib import Path
from tableqa.analysis import validate_run,index_records
from tableqa.core import TableState,run_plan
from tableqa.data import read_jsonl,write_jsonl,digest
from tableqa.distillation import compress,blocks

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('samples');p.add_argument('predictions');p.add_argument('accepted_ids');p.add_argument('output')
args=p.parse_args()
refs,preds,_=validate_run(read_jsonl(args.samples),read_jsonl(args.predictions))
accepted=read_jsonl(args.accepted_ids);keys=index_records(accepted,'accepted ID')
rows=[]
for key in keys:
    sample,pred=refs[key],preds[key]
    if sample['split']!='train' or pred['status']!='ok':raise ValueError('Only executable accepted training traces can be audited')
    state=TableState.from_table(sample['table']);original=pred['operations'];shortened=compress(state,original)
    final,_=run_plan(state,original);replayed,_=run_plan(state,shortened)
    if final.signature()!=replayed.signature():raise ValueError('Compression changed execution semantics/provenance')
    grouped=blocks(shortened,4)
    rows.append({'dataset':key[0],'id':key[1],'original_operations':len(original),
                 'compressed_operations':len(shortened),'execution_signature_equal':True,
                 'max_block':4,'block_lengths':[len(b) for b in grouped],
                 'feedback_flags':sum(bool(o.get('requires_feedback')) for o in shortened),
                 'feedback_flags_after_first_operation':sum(bool(o.get('requires_feedback')) for o in shortened[1:]),
                 'note':'Training records separate final answer supervision from pending operations.'})
root=Path(args.output);root.mkdir(parents=True,exist_ok=True);write_jsonl(root/'per-source.jsonl',rows)
report={'development_only':True,'n':len(rows),'accepted_ids_sha256':digest(accepted),'datasets':{}}
for d in sorted({r['dataset'] for r in rows}):
    selected=[r for r in rows if r['dataset']==d]
    before=sum(r['original_operations'] for r in selected);after=sum(r['compressed_operations'] for r in selected)
    report['datasets'][d]={'n':len(selected),'original_operations':before,'compressed_operations':after,
        'operation_reduction':1-after/before,'signature_preserved_count':len(selected),
        'block_lengths':dict(Counter(n for r in selected for n in r['block_lengths'])),
        'feedback_flags':sum(r['feedback_flags'] for r in selected),
        'feedback_flags_after_first_operation':sum(r['feedback_flags_after_first_operation'] for r in selected)}
report['limits']='Replay equality is not semantic correctness. Zero feedback flags means this cohort does not test adaptive feedback boundaries.'
(root/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
