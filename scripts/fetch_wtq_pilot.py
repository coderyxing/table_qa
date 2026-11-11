"""Fetch 200 training examples + their tables from the author's WTQ repo.

Pilot only: not a substitute for the final official train/dev/test protocol.
"""
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import subprocess

root=Path('data/raw/wikitq'); root.mkdir(parents=True,exist_ok=True)
base='https://raw.githubusercontent.com/ppasupat/WikiTableQuestions/master/'
def fetch(relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()): raise ValueError('Unsafe table path')
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        temp=path.with_suffix(path.suffix+'.part')
        subprocess.run(['curl','-4','-fLsS','--connect-timeout','10','--max-time','120','--retry','2',base+relative,'-o',str(temp)],check=True)
        temp.replace(path)
    return {'path':relative,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source':base+relative}
manifest=[fetch('data/training.tsv')]
with (root/'data/training.tsv').open(encoding='utf-8') as f:
    rows=list(csv.DictReader(f,delimiter='\t'))
random.Random(42).shuffle(rows); selected=rows[:200]
with ThreadPoolExecutor(max_workers=4) as pool:
    manifest+=list(pool.map(fetch,sorted({r['context'] for r in selected})))
with (root/'data/pilot200.tsv').open('w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(selected[0]),delimiter='\t'); writer.writeheader(); writer.writerows(selected)
(root/'pilot-manifest.json').write_text(json.dumps({'pilot_only':True,'seed':42,'n':200,'files':manifest},indent=2))
print('WTQ_TRAIN_PILOT_READY',len(selected),flush=True)
