# 脚本用途：在训练和评测前冻结一轮课程学习记录与开发来源。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Freeze a one-epoch controlled student pilot before any training/evaluation."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from tableqa.cli import metadata
from tableqa.curriculum import matched_schedules
from tableqa.data import read_jsonl, write_jsonl, digest, table_fingerprint
from tableqa.evaluation import label
from tableqa.inference import system_prompt
from tableqa.training import prepare_training_data
from scripts.build_tool_curriculum import make_pool


# 选择满足划分和历史接触限制的开发题。
# 输入：rows；n；seed（随机种子）。
# 返回形式含：result。
# 异常约束：Only the already used validation tuning cohort is allowed
def select_development(rows, n=20, seed=2035):
    if any(r['split'] != 'validation' for r in rows):
        raise ValueError('Only the already used validation tuning cohort is allowed')
    result = []; rng = random.Random(seed)
    for dataset in ['wikitq', 'tabfact']:
        candidates = sorted([r for r in rows if r['dataset'] == dataset], key=lambda r: r['id'])
        rng.shuffle(candidates)
        if len(candidates) < n: raise ValueError('Insufficient development inputs')
        result += candidates[:n]
    rng.shuffle(result)
    return result


# 构造并核对训练记录、监督类型和真实 token 成本。
# 输入：records（规范记录序列）；samples（固定样本集合）。
# 返回形式含：result。
# 异常约束：Training record/source split mismatch
def prepare_records(records, samples):
    refs = {(s['dataset'], s['id']): s for s in samples}; result = []
    for original in records:
        record = deepcopy(original); sample = refs[(record['dataset'], record['id'])]
        if record['split'] != 'train' or sample['split'] != 'train':
            raise ValueError('Training record/source split mismatch')
        if record['dataset'] not in {'wikitq', 'tabfact'}:
            raise ValueError('No unreviewed FeTaQA operation training')
        if len(record['messages']) != 2 or record['messages'][0]['role'] != 'system':
            raise ValueError('Unexpected source prompt shape')
        record['messages'][0]['content'] = system_prompt('B5', True, True, True)
        if record['dataset'] == 'tabfact':
            record['sampling_label'] = label(sample['reference'])
        result.append(record)
    return result


# 解析当前脚本参数，按模块说明执行本次冻结阶段并保存产物。
# 相关调用：read_jsonl、AutoTokenizer.from_pretrained、write_jsonl、(root / 'protocol.json').write_text。
# 异常约束：Stage already exists; preserve protocol and outputs
def main():
    # 定义本入口的参数解析器；位置参数、默认值和允许取值由相邻 add_argument 明确声明。
    p = argparse.ArgumentParser(description=__doc__)
    # 解析本次命令的实际参数，后续样本路径和实验开关均由此确定。
    p.add_argument('output'); args = p.parse_args(); root = Path(args.output)
    if root.exists(): raise ValueError('Stage already exists; preserve protocol and outputs')
    real = Path('runs/accuracy-training-pool'); synthetic = Path('runs/tool-curriculum-v1')
    # 读取结构化输入：real / 'records.jsonl'，保留原有样本身份字段。
    real_records = read_jsonl(real/'records.jsonl'); synthetic_records = read_jsonl(synthetic/'records.jsonl')
    real_report = json.loads((real/'report.json').read_text()); syn_report = json.loads((synthetic/'report.json').read_text())
    if digest(real_records) != real_report['records_sha256'] or digest(synthetic_records) != syn_report['records_sha256']:
        raise ValueError('Candidate pool identity mismatch')
    # 读取结构化输入：real / 'samples.jsonl'，保留原有样本身份字段。
    real_samples = read_jsonl(real/'samples.jsonl'); synthetic_samples = read_jsonl(synthetic/'samples.jsonl')
    real_records = prepare_records(real_records, real_samples)
    synthetic_records = prepare_records(synthetic_records, synthetic_samples)
    model = str(Path('models/Qwen2.5-3B-Instruct').resolve())
    initial = Path('runs/coverage-student-control/results/B5/adapter').resolve()
    adapter_cfg = json.loads((initial/'adapter_config.json').read_text())
    if adapter_cfg['r'] != 32 or adapter_cfg['lora_alpha'] != 64:
        raise ValueError('Unexpected initial adapter shape')
    from transformers import AutoTokenizer
    # 使用本阶段指定的 tokenizer；预算、提示模板与监督掩码应沿用同一版本。
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    control, treatment, matching = matched_schedules(real_records, synthetic_records, tokenizer)
    # 读取结构化输入：'runs/accuracy-development-v1/samples.jsonl'，保留原有样本身份字段。
    tune = read_jsonl('runs/accuracy-development-v1/samples.jsonl')
    development = select_development(tune)
    probe, _ = make_pool(4, seed=2034)
    train_fps = {table_fingerprint(s['table']) for s in real_samples + synthetic_samples}
    if any(table_fingerprint(s['table']) in train_fps for s in development + probe):
        raise ValueError('Evaluation table overlaps this training pool')
    common_keys = {(r['dataset'], r['id']) for r in real_records}
    for rows in [control, treatment]:
        if not common_keys <= {(r['dataset'], r['id']) for r in rows}:
            raise ValueError('Common real training sources were lost')
    configs = {}; preflight = {}
    for name, rows in [('R', control), ('M', treatment)]:
        config = {'model': model, 'initial_adapter': str(initial), 'seed': 42,
                  'sampling_policy': 'frozen', 'frozen_records_sha256': digest(rows),
                  'frozen_supervised_tokens_per_epoch': matching['control']['target_tokens'],
                  'max_seq_length': 8192, 'lora_rank': 32, 'lora_alpha': 64,
                  'learning_rate': .0002, 'answer_weight': 1,
                  'micro_batch': 1, 'gradient_accumulation_steps': 16,
                  'max_epochs': 1, 'loss_balance': 'none'}
        _, stats = prepare_training_data(config, rows, tokenizer)
        configs[name] = config; preflight[name] = stats
    if preflight['R']['supervised_tokens_per_epoch'] != preflight['M']['supervised_tokens_per_epoch']:
        raise ValueError('Actual trainer token budgets differ')
    root.mkdir(parents=True)
    # 保存逐条结构化产物到 root / 'R-records.jsonl'，供后续重放或审核读取。
    write_jsonl(root/'R-records.jsonl', control); write_jsonl(root/'M-records.jsonl', treatment)
    # 保存逐条结构化产物到 root / 'development-samples.jsonl'，供后续重放或审核读取。
    write_jsonl(root/'development-samples.jsonl', development); write_jsonl(root/'synthetic-probe.jsonl', probe)
    for name, config in configs.items():
        (root/f'{name}-config.json').write_text(json.dumps(config, indent=2))
    protocol = {'development_only': True, 'formal_test_not_read': True,
                'already_evaluated_confirmation_not_read': True,
                'P': 'Original 149-source 3B adapter, unchanged, same new tool prompt',
                'R': 'Common 172 real sources plus target-length-matched real replay',
                'M': 'Same common real records plus all 96 synthetic curriculum sources',
                'training': {'epochs': 1, 'seed': 42, 'effective_batch': 16,
                             'initial_adapter_sha256': hashlib.sha256((initial/'adapter_model.safetensors').read_bytes()).hexdigest()},
                'matching': matching, 'same_system_prompt': system_prompt('B5', True, True, True),
                'inference_settings': {'mode': 'B5', 'typed_quantities': True, 'semantic_extract': True,
                    'tool_demonstrations': True, 'max_block': 4, 'max_operations': 8, 'max_repairs': 2},
                'model_config': {'backend': 'local', 'model': model, 'max_input_tokens': 8192,
                                 'max_new_tokens': 512, 'device_map': 'auto', 'seed': 42},
                'development': {'n': len(development), 'n_per_task': 20, 'selection_seed': 2035,
                                'cohort': 'subset of already-used tuning cohort, not independent confirmation',
                                'sha256': digest(development)},
                'synthetic_probe': {'n': len(probe), 'seed': 2034, 'same_template_new_instances': True,
                                    'not_out_of_domain_or_benchmark_result': True, 'sha256': digest(probe)},
                'automatic_adoption': False, 'three_training_seeds_completed': False,
                'limits': ['Exact target tokens/steps do not imply identical input tokens or GPU compute.',
                           'Data-composition and control replay distributions differ; not an innovation-module ablation.',
                           'One epoch and one seed are a bounded development test, not the three-task main experiment.']}
    (root/'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    (root/'preflight.json').write_text(json.dumps(preflight, indent=2))
    (root/'run.json').write_text(json.dumps(metadata(protocol), indent=2))
    print(json.dumps({'prepared': True, 'matching_level': matching['matching_level'],
                      'records_per_epoch': len(control), 'target_tokens_per_epoch': matching['control']['target_tokens'],
                      'source_counts': [matching['control']['source_ids'], matching['treatment']['source_ids']],
                      'input_tokens': [preflight[n]['input_tokens_per_epoch'] for n in ['R', 'M']]}), flush=True)


if __name__ == '__main__': main()
