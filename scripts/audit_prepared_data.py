# 脚本用途：检查已准备的规范化划分、样本覆盖与跨任务重表。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Audit currently prepared canonical splits without choosing hyperparameters."""
from pathlib import Path
import json
from tableqa.data import read_jsonl,audit
records=[]; counts={}
for path in sorted(Path('data/normalized').glob('*_*.jsonl')):
    if 'pilot' in path.name: continue
    # 读取结构化输入：path，保留原有样本身份字段。
    rows=read_jsonl(path); records.extend(rows); counts[path.name]=len(rows)
report=audit(records)
report['files']=counts
Path('runs/setup/data-audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
print(json.dumps({'files':counts,'duplicate_ids':len(report['duplicate_ids']),
                  'overlap_fingerprints':len(report['table_overlaps'])},ensure_ascii=False),flush=True)
