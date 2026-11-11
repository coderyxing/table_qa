# 脚本用途：读取正在写入的开发产物，计算已完成记录和阶段进度。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Local progress accounting for one running batch; no external notifications."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time
# 定义本入口的参数解析器；位置参数、默认值和允许取值由相邻 add_argument 明确声明。
parser=argparse.ArgumentParser(); parser.add_argument('--pid',type=int,required=True); args=parser.parse_args()
root=Path('runs/medium-development'); previous=None
while True:
    stage=(root/'stage.txt').read_text().strip() if (root/'stage.txt').exists() else 'starting'
    counts={}
    for mode in ['A4','A0']:
        path=root/f'{mode}-predictions.jsonl'
        counts[mode]=sum(1 for line in path.open(encoding='utf-8') if line.endswith('\n')) if path.exists() else 0
    try: os.kill(args.pid,0); running=True
    except ProcessLookupError: running=False
    report={'stage':stage,'completed':counts,'expected_each':600,'batch_process_running':running,
            'updated_utc':datetime.now(timezone.utc).isoformat()}
    temporary=root/'progress.tmp'; temporary.write_text(json.dumps(report,indent=2)); temporary.replace(root/'progress.json')
    signature=(stage,tuple(counts.values()),running)
    if signature!=previous: print(json.dumps(report),flush=True); previous=signature
    if not running or stage in {'failed_or_interrupted','interrupted','development_complete_freeform_review_still_pending'}: break
    time.sleep(30)
