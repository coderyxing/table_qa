"""Audit currently prepared canonical splits without choosing hyperparameters."""
from pathlib import Path
import json
from tableqa.data import read_jsonl,audit
records=[]; counts={}
for path in sorted(Path('data/normalized').glob('*_*.jsonl')):
    if 'pilot' in path.name: continue
    rows=read_jsonl(path); records.extend(rows); counts[path.name]=len(rows)
report=audit(records)
report['files']=counts
Path('runs/setup/data-audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
print(json.dumps({'files':counts,'duplicate_ids':len(report['duplicate_ids']),
                  'overlap_fingerprints':len(report['table_overlaps'])},ensure_ascii=False),flush=True)
