"""CPU-only audit using the exact production tokenizer, sampler and padding rules."""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer
from tableqa.data import read_jsonl,digest
from tableqa.training import prepare_training_data
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('records');p.add_argument('config');p.add_argument('output')
args=p.parse_args();cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
cfg=cfg.get('config',cfg)
tokenizer=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True)
records=read_jsonl(args.records);_,stats=prepare_training_data(cfg,records,tokenizer)
stats['records_sha256']=digest(records)
out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in stats.items() if k!='source_exposure'},ensure_ascii=False,indent=2))
