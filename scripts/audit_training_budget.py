# 脚本用途：复用生产 tokenizer、采样和平衡规则，离线核算实际训练 token 和来源曝光。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""CPU-only audit using the exact production tokenizer, sampler and padding rules."""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer
from tableqa.data import read_jsonl,digest
from tableqa.training import prepare_training_data
# 定义本入口的参数解析器；位置参数、默认值和允许取值由相邻 add_argument 明确声明。
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('records');p.add_argument('config');p.add_argument('output')
# 解析本次命令的实际参数，后续样本路径和实验开关均由此确定。
args=p.parse_args();cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
cfg=cfg.get('config',cfg)
# 使用本阶段指定的 tokenizer；预算、提示模板与监督掩码应沿用同一版本。
tokenizer=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True)
# 读取结构化输入：args.records，保留原有样本身份字段。
records=read_jsonl(args.records);_,stats=prepare_training_data(cfg,records,tokenizer)
stats['records_sha256']=digest(records)
out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in stats.items() if k!='source_exposure'},ensure_ascii=False,indent=2))
