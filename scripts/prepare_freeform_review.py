"""Prepare complete source tables + trajectories for two independent reviewers."""
import json
import argparse
from pathlib import Path
from tableqa.data import read_jsonl,write_jsonl
parser=argparse.ArgumentParser(); parser.add_argument('--source',default='runs/teacher-pilot-fresh30'); args=parser.parse_args()
root=Path(args.source)
samples={(r['dataset'],r['id']):r for r in read_jsonl(root/'samples.jsonl')}
reviews=[]
for prediction in read_jsonl(root/'predictions.jsonl'):
    if prediction['dataset']!='fetaqa': continue
    sample=samples[(prediction['dataset'],prediction['id'])]
    reviews.append({'dataset':'fetaqa','id':sample['id'],'question':sample['question'],
       'reference':sample['reference'],'teacher_answer':prediction['answer'],
       'full_source_table':sample['table'],'source_highlighted_cells':sample.get('highlighted_cells',[]),
       'operations':prediction['operations'],'evidence':prediction['evidence'],'status':prediction['status'],
       'reviewer1':None,'reviewer2':None,'decision1':None,'decision2':None,
       'source_annotation_conflict':None,'unsupported_claims':None,'missing_facts':None,
       'notes':None,'trajectory_review_approved':False})
write_jsonl(root/'freeform-review.jsonl',reviews)
print('FREEFORM_REVIEW_READY',len(reviews),flush=True)
