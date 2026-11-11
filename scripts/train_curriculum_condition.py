# 脚本用途：按冻结课程配置在单张可见 GPU 上训练一个组别。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Train one frozen curriculum condition on exactly one visible GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from tableqa.cli import metadata
from tableqa.data import read_jsonl, digest
from tableqa.training import train


# 解析当前脚本参数，按模块说明执行本次冻结阶段并保存产物。
# 相关调用：read_jsonl、(root / 'run.json').write_text、(stage / f'{name}-config.json').read_text、(stage / 'protocol.json').read_text。
# 异常约束：Training output exists; preserve this attempt
def main():
    # 定义本入口的参数解析器；位置参数、默认值和允许取值由相邻 add_argument 明确声明。
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage'); p.add_argument('condition', choices=['R', 'M'])
    # 解析本次命令的实际参数，后续样本路径和实验开关均由此确定。
    args = p.parse_args(); stage = Path(args.stage).resolve(); name = args.condition
    root = stage.parent / ('training-' + name)
    if root.exists(): raise ValueError('Training output exists; preserve this attempt')
    config = json.loads((stage/f'{name}-config.json').read_text())
    protocol = json.loads((stage/'protocol.json').read_text())
    preparation = json.loads((stage/'run.json').read_text())
    # 读取结构化输入：stage / f'{name}-records.jsonl'，保留原有样本身份字段。
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
        # 执行当前配置的训练阶段，输出检查点与实际训练记录统计。
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
