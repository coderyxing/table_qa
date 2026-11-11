"""Reproducible observed-error and repair funnel; never changes predictions."""
import argparse
import json
from pathlib import Path
from tableqa.data import read_jsonl, write_jsonl
from tableqa.mechanisms import analyze_mechanisms

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('samples'); p.add_argument('predictions'); p.add_argument('output'); p.add_argument('--official-wtq')
args = p.parse_args()
official = json.loads(Path(args.official_wtq).read_text(encoding='utf-8')) if args.official_wtq else None
report, events, examples = analyze_mechanisms(read_jsonl(args.samples), read_jsonl(args.predictions), official)
root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
(root/'mechanisms.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
write_jsonl(root/'repair-events.jsonl', events); write_jsonl(root/'error-examples.jsonl', examples)
print(json.dumps(report, ensure_ascii=False, indent=2))
