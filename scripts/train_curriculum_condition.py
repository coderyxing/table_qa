"""Train one frozen curriculum condition on exactly one visible GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from tableqa.cli import metadata
from tableqa.data import read_jsonl, digest
from tableqa.training import train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage'); p.add_argument('condition', choices=['R', 'M'])
    args = p.parse_args(); stage = Path(args.stage).resolve(); name = args.condition
    root = stage.parent / ('training-' + name)
    if root.exists(): raise ValueError('Training output exists; preserve this attempt')
    config = json.loads((stage/f'{name}-config.json').read_text())
    protocol = json.loads((stage/'protocol.json').read_text())
    preparation = json.loads((stage/'run.json').read_text())
    records = read_jsonl(stage/f'{name}-records.jsonl'); current = metadata(config)
    if current['source_hash'] != preparation['source_hash']:
        raise ValueError('Training module code differs from the frozen preflight')
    if digest(records) != config['frozen_records_sha256']:
        raise ValueError('Frozen record order/content changed')
    adapter = Path(config['initial_adapter'])/'adapter_model.safetensors'
    if hashlib.sha256(adapter.read_bytes()).hexdigest() != protocol['training']['initial_adapter_sha256']:
        raise ValueError('Initial checkpoint changed')
    if shutil.disk_usage(stage).free < 8*1024**3:
        raise ValueError('Keep 8 GiB free before saving training artifacts')
    root.mkdir(); (root/'run.json').write_text(json.dumps(current, indent=2))
    try:
        statistics = train(config, records, root/'adapter')
        (root/'statistics.json').write_text(json.dumps(statistics, indent=2))
        if statistics['supervised_tokens_per_epoch'] != config['frozen_supervised_tokens_per_epoch']:
            raise ValueError('Actual training token schedule changed')
        (root/'COMPLETE').write_text('one frozen epoch completed\n')
        print('CURRICULUM_CONDITION_COMPLETE', name, flush=True)
    except BaseException as exc:
        (root/'FAILED.json').write_text(json.dumps({'type':type(exc).__name__,'message':str(exc)}, indent=2))
        raise


if __name__ == '__main__': main()
