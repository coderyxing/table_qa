# 脚本用途：只做分词和成本审计，检查合成工具候选的长度与监督分布。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Tokenize and audit the synthetic candidate without loading model weights."""
import argparse
import json
from pathlib import Path
from tableqa.data import read_jsonl, digest
from tableqa.training import prepare_training_data


# 解析当前脚本参数，按模块说明执行本次冻结阶段并保存产物。
# 相关调用：read_jsonl、AutoTokenizer.from_pretrained、prepare_training_data、(root / 'candidate-training-config.json').write_text。
# 异常约束：Preflight already exists
def main():
    # 定义本入口的参数解析器；位置参数、默认值和允许取值由相邻 add_argument 明确声明。
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('pool'); p.add_argument('model'); p.add_argument('initial_adapter'); p.add_argument('output')
    # 解析本次命令的实际参数，后续样本路径和实验开关均由此确定。
    args = p.parse_args(); root = Path(args.output)
    if root.exists(): raise ValueError('Preflight already exists')
    # 读取结构化输入：pool / 'records.jsonl'，保留原有样本身份字段。
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
    # 使用本阶段指定的 tokenizer；预算、提示模板与监督掩码应沿用同一版本。
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
