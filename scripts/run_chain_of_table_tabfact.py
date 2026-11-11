# 脚本用途：在固定上游源码上运行 TabFact/Qwen 迁移，支持精确声明映射。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""TabFact-only pinned-source/Qwen migration with optional Dater mapping.

Requires a verified source audit directory. Does NOT run an official test split,
extend prompts to other tasks, or claim reproduction of the paper's old model.
"""
import argparse
import hashlib
import json
from pathlib import Path

from tableqa.chain_of_table import load_upstream, load_dater_mapping, LocalTextModel, predict_tabfact
from tableqa.cli import metadata
from tableqa.data import read_jsonl, write_jsonl, digest
from tableqa.evaluation import diagnostic_score


# 解析当前脚本参数，按模块说明执行本次冻结阶段并保存产物。
# 相关调用：read_jsonl、write_jsonl、(root / 'protocol.json').write_text、(root / 'run.json').write_text。
# 异常约束：Output exists; preserve this run
def main():
    # 定义本入口的参数解析器；位置参数、默认值和允许取值由相邻 add_argument 明确声明。
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('data'); p.add_argument('source_audit'); p.add_argument('model'); p.add_argument('output')
    p.add_argument('--limit', type=int, default=4)
    p.add_argument('--max-generations', type=int, default=256)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dater-map', type=Path)
    # 解析本次命令的实际参数，后续样本路径和实验开关均由此确定。
    args = p.parse_args(); root = Path(args.output)
    if root.exists(): raise ValueError('Output exists; preserve this run')
    if args.limit < 1: raise ValueError('Positive diagnostic limit required')
    # 读取结构化输入：args.data，保留原有样本身份字段。
    all_samples = read_jsonl(args.data)
    if any(r['split'] != 'train' for r in all_samples):
        raise ValueError('This first migration diagnostic accepts training inputs only')
    samples = [r for r in all_samples if r['dataset'] == 'tabfact'][:args.limit]
    if len(samples) != args.limit: raise ValueError('Insufficient TabFact inputs')
    dater_mapping = load_dater_mapping(args.dater_map) if args.dater_map is not None else None
    chain, manifest = load_upstream(args.source_audit)
    model_config = {'model': str(Path(args.model).resolve()), 'device_map': 'balanced',
                    'max_input_tokens': 8192, 'seed': args.seed}
    protocol = {'diagnostic_only': True, 'split': 'train', 'n': len(samples),
                'upstream_commit': manifest['commit'], 'source_manifest_sha256': digest(manifest),
                'sample_sha256': digest(samples), 'model_config': model_config,
                'variant': ('pinned_code_defaults_raw_statement' if dater_mapping is None else
                            'pinned_code_defaults_dater_add_column'), 'max_generations': args.max_generations,
                'dater_mapping_sha256': (hashlib.sha256(args.dater_map.read_bytes()).hexdigest()
                                         if args.dater_map is not None else None),
                'dater_mapping_hits': (sum(s['question'] in dater_mapping for s in samples)
                                       if dater_mapping is not None else 0),
                'sampling': 'upstream defaults; eight row/column candidates at temperature 0.5',
                'confidence': 'upstream return-order proxy; NOT a model likelihood',
                'runtime_changes': ['verified source with eval replaced by literal parsing/static lookup',
                    'local Qwen text generation; candidate samples executed sequentially',
                    'all runtime failures retained; no response cache or API retry sleeping'],
                'unaligned': ['Dater statement coverage on the paper cohort', 'paper parameter variant', 'paper test IDs',
                              'WikiTQ and FeTaQA prompts', 'old PaLM2/GPT3.5 model'],
                'formal_speed_benchmark': False, 'may_claim_full_paper_reproduction': False}
    # 保存逐条结构化产物到 root / 'samples.jsonl'，供后续重放或审核读取。
    root.mkdir(parents=True); write_jsonl(root / 'samples.jsonl', samples)
    (root / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    (root / 'run.json').write_text(json.dumps(metadata(protocol), indent=2), encoding='utf-8')
    backend = LocalTextModel(model_config); predictions = []
    for sample in samples:
        result = predict_tabfact(sample, backend, chain, args.max_generations, dater_mapping=dater_mapping)
        # 保存逐条结构化产物到 root / 'predictions.jsonl'，供后续重放或审核读取。
        predictions.append(result); write_jsonl(root / 'predictions.jsonl', predictions)
        print(json.dumps({'id': result['id'], 'status': result['status'], 'answer': result['answer'],
                          'candidate_generations': result['metrics']['model_calls']}), flush=True)
    scores = [diagnostic_score(p, s) for p, s in zip(predictions, samples)]
    report = {'diagnostic_only': True, 'n': len(samples), 'accuracy': sum(scores) / len(scores),
              'dater_mapping_hits': sum(r['dater_map_hit'] for r in predictions),
              'dater_text_changes': sum(r['dater_text_changed'] for r in predictions),
              'failures': sum(r['status'] != 'ok' for r in predictions),
              'upstream_migration_attempted': True,
              'successful_pipelines': sum(r['status'] == 'ok' for r in predictions),
              'full_paper_reproduction': False}
    (root / 'diagnostic.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('COT_TABFACT_MIGRATION_DIAGNOSTIC_COMPLETE', flush=True)


if __name__ == '__main__': main()
