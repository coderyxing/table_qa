"""Export independent review packets or validate/import two completed JSONL files."""
import argparse
import json
from pathlib import Path
from tableqa.data import read_jsonl, write_jsonl
from tableqa.review import make_packet, review_template, render_packet, merge_reviews

parser = argparse.ArgumentParser(description=__doc__)
sub = parser.add_subparsers(dest='command', required=True)
p = sub.add_parser('export')
p.add_argument('samples'); p.add_argument('predictions'); p.add_argument('output')
p.add_argument('--purpose', choices=['train_filter', 'validation_filter', 'evaluation'], required=True)
p = sub.add_parser('merge')
p.add_argument('packet'); p.add_argument('reviewer1'); p.add_argument('reviewer2'); p.add_argument('output')
args = parser.parse_args()
root = Path(args.output)
if root.exists(): raise ValueError('Output already exists; retain the previous audit and use a new directory')
if args.command == 'export':
    packet = make_packet(read_jsonl(args.samples), read_jsonl(args.predictions), args.purpose)
    root.mkdir(parents=True)
    (root/'packet.json').write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding='utf-8')
    (root/'review.html').write_text(render_packet(packet), encoding='utf-8')
    for reviewer in ['reviewer1', 'reviewer2']:
        write_jsonl(root/f'{reviewer}.jsonl', review_template(packet))
    print(json.dumps({'review_items': len(packet['items']), 'purpose': args.purpose, 'auto_approved': 0}))
else:
    packet = json.loads(Path(args.packet).read_text(encoding='utf-8'))
    report, approved = merge_reviews(packet, read_jsonl(args.reviewer1), read_jsonl(args.reviewer2))
    root.mkdir(parents=True)
    (root/'review-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    write_jsonl(root/'approved-samples.jsonl', approved)
    print(json.dumps({'review_items': report['n'], 'approved_samples': len(approved),
                      'decision_counts': report['decision_counts']}))
