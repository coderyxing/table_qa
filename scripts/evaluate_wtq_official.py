"""Run WTQ v1.0.2's official denotation functions via a recorded Python3 port.

Supply the evaluator and CoreNLP tagged files from the official archive.
No algorithm rewrite; lib2to3 performs syntax conversion under Python 3.10-3.12.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import warnings
from tableqa.data import read_jsonl,digest

parser=argparse.ArgumentParser()
parser.add_argument('samples'); parser.add_argument('predictions'); parser.add_argument('output')
parser.add_argument('--evaluator',required=True); parser.add_argument('--tagged',required=True)
args=parser.parse_args()
source=Path(args.evaluator).read_text(encoding='utf-8')
with warnings.catch_warnings():
    warnings.simplefilter('ignore',DeprecationWarning)
    from lib2to3.refactor import RefactoringTool,get_fixers_from_package
    converted=str(RefactoringTool(get_fixers_from_package('lib2to3.fixes')).refactor_string(source,'official_wtq'))
out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
module_path=out/'official_evaluator_py3.py'; module_path.write_text(converted,encoding='utf-8')
spec=importlib.util.spec_from_file_location('_official_wtq',module_path)
official=importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
if official.__version__!='1.0.2': raise ValueError('Unexpected official evaluator version')
def match(target,prediction,canonical=None):
    return official.check_denotation(official.to_value_list(target,canonical),official.to_value_list(prediction))
assert match(['Café'],['cafe'])
assert match(['2,000'],['2000'],['2000'])
assert match(['Jan 2 1990'],['1990-01-02'],['1990-01-02'])
assert not match(['Jan 2'],['1990-01-02'],['xx-01-02'])
targets={}
for path in sorted(Path(args.tagged).glob('*.tagged')):
    with path.open(encoding='utf-8') as f:
        for row in csv.DictReader(f,delimiter='\t',quoting=csv.QUOTE_NONE):
            targets[row['id']]=(official.tsv_unescape_list(row['targetValue']),official.tsv_unescape_list(row['targetCanon']))
samples=[r for r in read_jsonl(args.samples) if r['dataset']=='wikitq']
predictions=[r for r in read_jsonl(args.predictions) if r['dataset']=='wikitq']
lookup={r['id']:r for r in predictions}
if len(lookup)!=len(predictions) or {r['id'] for r in samples}!=set(lookup):
    raise ValueError('Exact WTQ prediction coverage required')
scores=[]
for sample in samples:
    prediction=lookup[sample['id']]; target,canonical=targets[sample['id']]
    answer=prediction['answer'] if prediction['status']=='ok' else []
    answer=answer if isinstance(answer,list) else [answer]
    score=int(match(target,[str(v) for v in answer],canonical))
    scores.append({'dataset':'wikitq','id':sample['id'],'score':score,'status':prediction['status']})
report={'metric':'WTQ official v1.0.2 denotation, Python3 syntax port','n':len(scores),
        'accuracy':sum(r['score'] for r in scores)/len(scores) if scores else None,
        'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'converted_sha256':hashlib.sha256(converted.encode()).hexdigest(),
        'gold_canonical_values':'official CoreNLP targetCanon','self_checks_passed':4,'scores':scores,
        'samples_structured_sha256':digest(sorted(samples,key=lambda r:(r['dataset'],r['id']))),
        'predictions_structured_sha256':digest(sorted(predictions,key=lambda r:(r['dataset'],r['id']))),
        'tagged_files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(args.tagged).glob('*.tagged'))}}
(out/'scores.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='scores'}),flush=True)
