# 脚本用途：核对已完成联合对照的产物并输出结果摘要。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Audit and format completed joint controls; no training or model calls."""
import argparse
import hashlib
import json
from pathlib import Path


# 读取指定结构化产物，交由当前阶段核验来源和配置。
# 输入：p。
# 相关调用：Path(p).read_text。
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
# 计算文件或产物内容摘要，供模型、输入和协议绑定。
# 输入：p。
# 相关调用：hashlib.sha256(Path(p).read_bytes()).hexdigest。
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# 解析当前脚本参数，按模块说明执行本次冻结阶段并保存产物。
# 相关调用：read_jsonl、(root / 'report.md').write_text、(root / 'completion-audit.json').write_text、digest。
# 异常约束：Wait for all conditions
def main():
    # 定义本入口的参数解析器；位置参数、默认值和允许取值由相邻 add_argument 明确声明。
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage');args=parser.parse_args();root=Path(args.stage)
    if not (root/'COMPLETE').exists():raise ValueError('Wait for all conditions')
    if (root/'completion-audit.json').exists():raise ValueError('Preserve completed audit')
    protocol=load(root/'protocol.json');report=load(root/'report.json');preflight=load(root/'preflight.json')
    from tableqa.data import read_jsonl,digest
    # 读取结构化输入：root / 'sources.jsonl'，保留原有样本身份字段。
    sources=read_jsonl(root/'sources.jsonl');samples=read_jsonl(root/'samples.jsonl')
    if digest(sources)!=protocol['sources_sha256'] or digest(samples)!=protocol['validation_sha256']:raise ValueError('Frozen sources/evaluation changed')
    expected_ids={(s['dataset'],s['id']) for s in sources};sample_ids={(s['dataset'],s['id']) for s in samples}
    audit={'scope':'actual schedule/source/coverage audit, not semantic approval','conditions':{},'equal_target_tokens':False,'automatic_adoption':False}
    lines=['# 修复版多任务B0/B2/B5开发对照','',
        '同一基础3B、新建LoRA、相同684来源：172条弱检查短答/标签轨迹＋512条官方FeTaQA答案。',
        'FeTaQA三组均为B0；不声称其操作蒸馏或人工审核完成。来源相同，训练token不同。','',
        '|配置|WTQ（20题）|TabFact（20题）|FeTaQA BLEU（100题）|R1|R2|RL|训练目标token|实际步数|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for mode in ['B0','B2','B5']:
        out=root/mode;statistics=load(out/'statistics.json');exposure=load(out/'adapter/training_exposure.json')['per_source']
        # 读取结构化输入：out / 'predictions.jsonl'，保留原有样本身份字段。
        history=load(out/'adapter/training_history.json');predictions=read_jsonl(out/'predictions.jsonl');loaded=load(out/'loaded-model.json')
        if {(r['dataset'],r['id']) for r in exposure}!=expected_ids:raise ValueError('Actual source exposure mismatch')
        if len(predictions)!=len(samples) or {(r['dataset'],r['id']) for r in predictions}!=sample_ids:raise ValueError('Prediction coverage mismatch')
        if not loaded['active_adapters'] or loaded['adapter_sha256']!=sha(out/'adapter/adapter_model.safetensors'):raise ValueError('Loaded adapter binding mismatch')
        steps=max(row.get('step',0) for row in history)
        if steps!=statistics['records_after_alignment']//16:raise ValueError('Actual optimizer steps mismatch')
        for key in ['sampling_sequence_sha256','supervised_tokens_per_epoch','source_samples_after_alignment','records_after_alignment']:
            if statistics[key]!=preflight[mode][key]:raise ValueError('Preflight/actual mismatch')
        if statistics['gradient_normalization']!='explicit-microbatch-mean-v1':raise ValueError('Wrong gradient normalization')
        values=report['conditions'][mode];task={r['dataset']:r['observed_score'] for r in values['task_summary']['rows']};ft=values['freeform']
        lines.append(f"|{mode}|{task['wikitq']:.3f}|{task['tabfact']:.3f}|{ft['sacrebleu']:.3f}|{ft['rouge1_f1']:.3f}|{ft['rouge2_f1']:.3f}|{ft['rougeL_f1']:.3f}|{statistics['supervised_tokens_per_epoch']}|{steps}|")
        audit['conditions'][mode]={'actual_steps':steps,'actual_sources':len(expected_ids),'predictions':len(predictions),'adapter_sha256':loaded['adapter_sha256'],
            'actual_target_tokens':statistics['supervised_tokens_per_epoch'],'padding_records':statistics['padding_records']}
    lines+=['','单种子、旧开发输入，无正式测试/速度比较或自动采用。B2/B5的目标token不同，不能据此单独认定分块的因果收益。',
            '配对区间、失败、修复及调用/token原始记录均保留；保留零收益与负向结果。']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    audit['passed']=True;audit['artifact_sha256']={n:sha(root/n) for n in ['protocol.json','run.json','sources.jsonl','samples.jsonl','preflight.json','report.json','report.md']}
    (root/'completion-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print('JOINT_CORE_FINAL_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':main()
