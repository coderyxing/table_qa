"""Tokenize and audit the synthetic candidate without loading model weights."""
import argparse
import json
from pathlib import Path
from tableqa.data import read_jsonl, digest
from tableqa.training import prepare_training_data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('pool'); p.add_argument('model'); p.add_argument('initial_adapter'); p.add_argument('output')
    args = p.parse_args(); root = Path(args.output)
    if root.exists(): raise ValueError('Preflight already exists')
    pool = Path(args.pool); records = read_jsonl(pool/'records.jsonl')
    report = json.loads((pool/'report.json').read_text())
    if not report['synthetic_only'] or digest(records) != report['records_sha256']:
        raise ValueError('Synthetic pool identity mismatch')
    config = {'model': str(Path(args.model).resolve()),
              'initial_adapter': str(Path(args.initial_adapter).resolve()), 'seed': 42,
              'sampling_policy': 'coverage', 'max_seq_length': 8192,
              'lora_rank': 32, 'lora_alpha': 64, 'learning_rate': 0.0002,
              'answer_weight': 1, 'micro_batch': 1, 'gradient_accumulation_steps': 16,
              'max_epochs': 3, 'loss_balance': 'none'}
    if not Path(config['initial_adapter'], 'adapter_config.json').is_file():
        raise ValueError('Initial adapter does not exist')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True)
    encoded, stats = prepare_training_data(config, records, tokenizer)
    lengths = [len(r['input_ids']) for r in encoded]
    root.mkdir(parents=True)
    (root/'candidate-training-config.json').write_text(json.dumps(config, indent=2))
    (root/'preflight.json').write_text(json.dumps({'synthetic_only': True, 'training_executed': False,
        'candidate_only_not_adopted': True, 'records_sha256': report['records_sha256'],
        'max_record_tokens': max(lengths), 'min_record_tokens': min(lengths),
        'statistics': stats, 'required_followup': 'Compare matched-prompt old/new students on fresh synthetic instances and representative real development data; control added training budget before attribution.'}, indent=2))
    print(json.dumps({'sources': report['sources'], 'encoded_records': len(encoded),
                      'max_record_tokens': max(lengths),
                      'actual_target_tokens_per_epoch': stats['supervised_tokens_per_epoch'],
                      'statistics_keys': list(stats), 'training_executed': False}))


if __name__ == '__main__': main()
