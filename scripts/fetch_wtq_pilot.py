# 脚本用途：下载作者发布的 WTQ 训练题及对应原表，构建小批量开发输入。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
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
# 按请求的 URL 或字节区间下载文件，交由后续大小与哈希检查验收。
# 输入：relative。
# 相关调用：hashlib.sha256(path.read_bytes()).hexdigest。
# 返回形式含：字段 path、sha256、source。
# 异常约束：Unsafe table path
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
