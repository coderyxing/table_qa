# 脚本用途：核对 FeTaQA 迁移产物的哈希、覆盖、配置与自动指标一致性。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Post-run artifact consistency audit, not a semantic/human review.

Run from the project root with PYTHONPATH pointing at the frozen inference
snapshot. Never rewrites predictions, protocols, checkpoints or report.json.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from tableqa.data import digest, read_jsonl, table_fingerprint
from tableqa.reporting import freeform_metrics, compare_freeform, freeform_texts
from scripts.fetaqa_transfer_pilot import select_samples


# 验证明确的不变量，失败时立即报告对应的审计原因。
# 输入：condition；message。
def require(condition,message):
    if not condition:raise ValueError(message)


# 计算文件或产物内容摘要，供模型、输入和协议绑定。
# 输入：path（输入或输出路径）。
# 相关调用：hashlib.sha256(Path(path).read_bytes()).hexdigest。
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
# 读取指定结构化产物，交由当前阶段核验来源和配置。
# 输入：path（输入或输出路径）。
# 相关调用：Path(path).read_text。
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


# 解析当前脚本参数，按模块说明执行本次冻结阶段并保存产物。
# 相关调用：digest、read_jsonl、(root / 'completion-audit.json').write_text、p.read_text。
def main():
    # 定义本入口的参数解析器；位置参数、默认值和允许取值由相邻 add_argument 明确声明。
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('results');args=parser.parse_args()
    root=Path(args.results);protocol=load(root/'protocol.json');run=load(root/'run.json');report=load(root/'report.json')
    require((root/'COMPLETE').is_file(),'Inference did not complete')
    require(run['config']==protocol,'Run/protocol mismatch')
    source=Path(__import__('tableqa').__file__).parent
    source_hash=digest({p.name:p.read_text(encoding='utf-8') for p in sorted(source.glob('*.py'))})
    require(source_hash==run['source_hash'],'Audit must use the frozen inference modules')
    # 读取结构化输入：root / 'samples.jsonl'，保留原有样本身份字段。
    samples=read_jsonl(root/'samples.jsonl')
    require(len(samples)==12==protocol['n'],'Unexpected pilot size')
    require(digest(samples)==protocol['samples_sha256'],'Frozen samples changed')
    # 读取结构化输入：'data/normalized/joint_validation_clean.jsonl'，保留原有样本身份字段。
    require(samples==select_samples(read_jsonl('data/normalized/joint_validation_clean.jsonl')),'Sampling protocol mismatch')
    # 读取结构化输入：'runs/accuracy-training-pool/samples.jsonl'，保留原有样本身份字段。
    training=read_jsonl('runs/accuracy-training-pool/samples.jsonl')+read_jsonl('runs/tool-curriculum-v1/samples.jsonl')
    train_fps={table_fingerprint(s['table']) for s in training}
    require(not any(table_fingerprint(s['table']) in train_fps for s in samples),'Training table overlap')
    previous=Path('runs/curriculum-student-v1')
    original=load(previous/'prepared/R-config.json')['initial_adapter']
    adapters={'P':Path(original),'R':previous/'training-R/adapter','M':previous/'training-M/adapter'}
    for name,path in adapters.items():
        require(sha(path/'adapter_model.safetensors')==protocol['adapter_sha256'][name],'Checkpoint changed')
        exposure=load(path/'training_exposure.json')
        datasets=sorted({s['dataset'] for s in exposure['per_source']})
        require('fetaqa' not in datasets and datasets==protocol['training_dataset_exposure'][name],'Target training exposure')
    outputs={};summaries={};files=['protocol.json','run.json','samples.jsonl','report.json','report.md','COMPLETE']
    metric_keys=['sacrebleu','rouge1_f1','rouge2_f1','rougeL_f1']
    require(set(protocol['conditions'])==set(report['conditions'])=={'PD','PC','RC','MC'},'Unexpected conditions')
    for name,condition in protocol['conditions'].items():
        # 读取结构化输入：root / filename，保留原有样本身份字段。
        filename=f'{name}-predictions.jsonl';files.append(filename);rows=read_jsonl(root/filename);outputs[name]=rows
        aligned=freeform_texts(rows,samples)
        require(len(rows)==12,'Incomplete outputs')
        require(all(p['mode']==condition['mode'] for p in rows),'Prediction mode mismatch')
        require(all(p['metrics']['latency_comparable'] is False and p['metrics']['cache_hits']==0 for p in rows),'Unexpected timing/cache protocol')
        metrics=freeform_metrics(rows,samples);stored=report['conditions'][name]
        for key in metric_keys:require(math.isclose(metrics[key],stored[key],abs_tol=1e-10),'Metric recomputation mismatch')
        for key in ['n','failures','invalid_answer_text','sacrebleu_signature','metric_packages']:
            require(metrics[key]==stored[key],'Metric metadata mismatch')
        summaries[name]={**metrics,'empty_hypotheses':sum(not h.strip() for h in aligned['hypotheses'])}
    for a,b in [('PD','PC'),('PC','RC'),('RC','MC')]:
        expected=compare_freeform(outputs[a],outputs[b],samples,repeats=protocol['bootstrap_repeats'])
        require(expected==report['comparisons'][a+'-vs-'+b],'Paired interval recomputation mismatch')
    audit={'passed':True,'scope':'artifact consistency, no semantic or human approval',
           'prediction_count':48,'condition_count':4,'source_hash':source_hash,
           'no_project_fetaqa_training_exposure':True,'no_exact_training_table_overlap':True,
           'all_failures_and_invalid_answer_types_scored_as_empty':True,
           'human_factual_consistency':None,'automatic_adoption':False,'conditions':summaries,
           'artifact_sha256':{f:sha(root/f) for f in files}}
    (root/'completion-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FETAQA_TRANSFER_ARTIFACT_AUDIT_COMPLETE',json.dumps({'predictions':48,'passed':True}))


if __name__=='__main__':main()
