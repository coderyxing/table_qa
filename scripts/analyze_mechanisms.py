# 脚本用途：读取预测并分析错误类型、修复漏斗与操作分布。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Reproducible observed-error and repair funnel; never changes predictions."""
import argparse
import json
from pathlib import Path
from tableqa.data import read_jsonl, write_jsonl
from tableqa.mechanisms import analyze_mechanisms

# 定义本入口的参数解析器；位置参数、默认值和允许取值由相邻 add_argument 明确声明。
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('samples'); p.add_argument('predictions'); p.add_argument('output'); p.add_argument('--official-wtq')
# 解析本次命令的实际参数，后续样本路径和实验开关均由此确定。
args = p.parse_args()
official = json.loads(Path(args.official_wtq).read_text(encoding='utf-8')) if args.official_wtq else None
# 读取结构化输入：args.samples，保留原有样本身份字段。
report, events, examples = analyze_mechanisms(read_jsonl(args.samples), read_jsonl(args.predictions), official)
root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
(root/'mechanisms.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
# 保存逐条结构化产物到 root / 'repair-events.jsonl'，供后续重放或审核读取。
write_jsonl(root/'repair-events.jsonl', events); write_jsonl(root/'error-examples.jsonl', examples)
print(json.dumps(report, ensure_ascii=False, indent=2))
