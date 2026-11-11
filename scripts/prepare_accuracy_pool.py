# 脚本用途：构建新的训练来源池，保存版本以便与历史控制区分。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Build an explicitly new training pool; never rewrites the fixed 149-source control."""
from collections import Counter
import json
from pathlib import Path
from transformers import AutoTokenizer
from tableqa.data import read_jsonl,write_jsonl,digest
from tableqa.quality import audit_run_trajectories
from tableqa.distillation import build_records
from tableqa.training import encode_record

root=Path('runs/accuracy-training-pool')
if root.exists():raise ValueError('Training pool exists; preserve it and use a versioned revision')
sources=[('medium',Path('runs/medium-development/samples.jsonl'),Path('runs/medium-development/A4-predictions.jsonl'),Path('runs/medium-analysis/wtq-A4/scores.json')),
    ('whole',Path('runs/interactive-teacher/samples.jsonl'),Path('runs/interactive-teacher/whole-predictions.jsonl'),Path('runs/interactive-teacher/wtq-whole/scores.json')),
    ('interactive',Path('runs/interactive-teacher/samples.jsonl'),Path('runs/interactive-teacher/interactive-predictions.jsonl'),Path('runs/interactive-teacher/wtq-interactive/scores.json'))]
# 使用本阶段指定的 tokenizer；预算、提示模板与监督掩码应沿用同一版本。
tokenizer=AutoTokenizer.from_pretrained('models/Qwen2.5-3B-Instruct',local_files_only=True)
all_audits=[];candidates={};raw_unique=set()
for priority,(name,samples_path,predictions_path,official_path) in enumerate(sources):
    # 读取结构化输入：samples_path，保留原有样本身份字段。
    samples=read_jsonl(samples_path);predictions=read_jsonl(predictions_path)
    if any(s['split']!='train' for s in samples):raise ValueError('Only source training samples may be selected')
    refs={(s['dataset'],s['id']):s for s in samples};lookup={(p['dataset'],p['id']):p for p in predictions}
    raw_unique.update(refs)
    audits=audit_run_trajectories(samples,predictions,json.loads(official_path.read_text()))
    for audit in audits:
        audit={**audit,'teacher_source':name};all_audits.append(audit)
        if not audit['decision'].startswith('accepted_'):continue
        key=(audit['dataset'],audit['id']);sample=refs[key];pred=lookup[key]
        # Fixed tie-break, no validation/test scores involved. All are still weakly audited.
        rank=(pred['metrics']['repairs'],priority,len(pred['operations']))
        if key not in candidates or rank<candidates[key]['rank']:
            candidates[key]={'rank':rank,'sample':sample,'prediction':pred,'audit':audit}
records=[];accepted=[];predictions=[];selected=[];rejected=[]
for key,entry in sorted(candidates.items()):
    sample,pred=entry['sample'],entry['prediction']
    rows=build_records(sample,pred['operations'],'B5')  # conservative, preserves question conditions
    try:
        for row in rows:encode_record(row,tokenizer,8192)
    except ValueError as exc:
        rejected.append({'dataset':key[0],'id':key[1],'reason':str(exc)});continue
    for row in rows:
        if sample['dataset']=='tabfact':row['sampling_label']=int(sample['reference'])
    records+=rows;accepted.append(sample);predictions.append(pred)
    selected.append({'dataset':key[0],'id':key[1],'teacher_source':entry['audit']['teacher_source'],
                     'audit_decision':entry['audit']['decision'],'semantic_correctness_proven':False})
if not accepted:raise ValueError('No accepted training sources')
# 保存逐条结构化产物到 root / 'samples.jsonl'，供后续重放或审核读取。
root.mkdir();write_jsonl(root/'samples.jsonl',accepted);write_jsonl(root/'predictions.jsonl',predictions)
# 保存逐条结构化产物到 root / 'records.jsonl'，供后续重放或审核读取。
write_jsonl(root/'records.jsonl',records);write_jsonl(root/'selection.jsonl',selected)
# 保存逐条结构化产物到 root / 'candidate-audits.jsonl'，供后续重放或审核读取。
write_jsonl(root/'candidate-audits.jsonl',all_audits);write_jsonl(root/'overlength-rejections.jsonl',rejected)
summary={'development_only':True,'raw_unique_sources':len(raw_unique),'candidate_trajectories':len(all_audits),
    'accepted_sources':len(accepted),'datasets':dict(Counter(s['dataset'] for s in accepted)),
    'tabfact_training_labels':dict(Counter(str(s['reference']) for s in accepted if s['dataset']=='tabfact')),
    'records':len(records),'records_sha256':digest(records),'source_ids_sha256':digest(sorted((s['dataset'],s['id']) for s in accepted)),
    'candidate_teacher_sources':dict(Counter(s['teacher_source'] for s in selected)),
    'feTaQA_auto_approved':False,'selection_uses_validation_or_test_scores':False,
    'max_training_length':8192,'compression':'legacy; no task-condition deletion',
    'limits':'Official answer agreement and replay are weak checks, not proof of semantic process correctness; more candidates may add selection bias.'}
(root/'report.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
