# 脚本用途：校验两组预测与样本的完整对应，导入官方分数并输出配对分析。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Compare two completed runs or explicitly inspect an incomplete run."""
import argparse
import json
from pathlib import Path
from tableqa.data import read_jsonl
from tableqa.analysis import summarize_run,compare_runs,write_analysis,read_completed_prefix
# 定义本入口的参数解析器；位置参数、默认值和允许取值由相邻 add_argument 明确声明。
parser=argparse.ArgumentParser(); parser.add_argument('samples'); parser.add_argument('predictions_a'); parser.add_argument('output')
parser.add_argument('--predictions-b'); parser.add_argument('--official-a'); parser.add_argument('--official-b')
# 解析本次命令的实际参数，后续样本路径和实验开关均由此确定。
parser.add_argument('--allow-partial',action='store_true'); args=parser.parse_args()
# 读取结构化输入：args.samples，保留原有样本身份字段。
samples=read_jsonl(args.samples); a=read_completed_prefix(args.predictions_a) if args.allow_partial else read_jsonl(args.predictions_a)
load=lambda p:json.loads(Path(p).read_text()) if p else None
oa,ob=load(args.official_a),load(args.official_b)
summaries={'A':summarize_run(samples,a,oa,args.allow_partial)}; comparisons=[]
if args.predictions_b:
    if args.allow_partial: raise ValueError('Paired comparisons require complete coverage')
    # 读取结构化输入：args.predictions_b，保留原有样本身份字段。
    b=read_jsonl(args.predictions_b); summaries['B']=summarize_run(samples,b,ob)
    comparisons=compare_runs(samples,a,b,oa,ob)
write_analysis(args.output,summaries,comparisons)
print(json.dumps({'output':args.output,'partial':summaries['A']['partial']}),flush=True)
