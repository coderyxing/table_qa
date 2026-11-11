# 脚本用途：为自由文本题导出完整原表与操作轨迹，供两人独立审核。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Prepare complete source tables + trajectories for two independent reviewers."""
import json
import argparse
from pathlib import Path
from tableqa.data import read_jsonl,write_jsonl
# 定义本入口的参数解析器；位置参数、默认值和允许取值由相邻 add_argument 明确声明。
parser=argparse.ArgumentParser(); parser.add_argument('--source',default='runs/teacher-pilot-fresh30'); args=parser.parse_args()
root=Path(args.source)
# 读取结构化输入：root / 'samples.jsonl'，保留原有样本身份字段。
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
# 保存逐条结构化产物到 root / 'freeform-review.jsonl'，供后续重放或审核读取。
write_jsonl(root/'freeform-review.jsonl',reviews)
print('FREEFORM_REVIEW_READY',len(reviews),flush=True)
