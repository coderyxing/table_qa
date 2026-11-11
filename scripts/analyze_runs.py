"""Compare two completed runs or explicitly inspect an incomplete run."""
import argparse
import json
from pathlib import Path
from tableqa.data import read_jsonl
from tableqa.analysis import summarize_run,compare_runs,write_analysis,read_completed_prefix
parser=argparse.ArgumentParser(); parser.add_argument('samples'); parser.add_argument('predictions_a'); parser.add_argument('output')
parser.add_argument('--predictions-b'); parser.add_argument('--official-a'); parser.add_argument('--official-b')
parser.add_argument('--allow-partial',action='store_true'); args=parser.parse_args()
samples=read_jsonl(args.samples); a=read_completed_prefix(args.predictions_a) if args.allow_partial else read_jsonl(args.predictions_a)
load=lambda p:json.loads(Path(p).read_text()) if p else None
oa,ob=load(args.official_a),load(args.official_b)
summaries={'A':summarize_run(samples,a,oa,args.allow_partial)}; comparisons=[]
if args.predictions_b:
    if args.allow_partial: raise ValueError('Paired comparisons require complete coverage')
    b=read_jsonl(args.predictions_b); summaries['B']=summarize_run(samples,b,ob)
    comparisons=compare_runs(samples,a,b,oa,ob)
write_analysis(args.output,summaries,comparisons)
print(json.dumps({'output':args.output,'partial':summaries['A']['partial']}),flush=True)
