# 模块职责：导出官方判分输入，计算 FeTaQA 语料级自动指标，并生成离线 HTML 报告。
# 接口约定：样本、操作和结果采用本项目的结构化字典；细节见各入口及字段检查。
"""Export official-evaluator inputs and render honest run summaries."""
import csv
import html
import json
from pathlib import Path


# 按任务格式导出当前完整预测，保留供官方工具关联的 ID。
# 输入：predictions（完整预测集合）；samples（固定样本集合）；directory（数据或模型目录）。
# 相关调用：write_jsonl。
def export_official(predictions, samples, directory):
    from .data import write_jsonl
    lookup={(r["dataset"],r["id"]):r for r in samples}
    root=Path(directory); root.mkdir(parents=True,exist_ok=True)
    # WTQ: id followed by predicted denotation values (tab-separated).
    with (root/"wikitq_predictions.tsv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.writer(f,delimiter="\t")
        for p in predictions:
            if p["dataset"]!="wikitq": continue
            answer=p["answer"] if p["status"]=="ok" else []
            answer=answer if isinstance(answer,list) else [answer]
            writer.writerow([p["id"]]+[str(v) for v in answer])
    records=[]
    for p in predictions:
        r=lookup[(p["dataset"],p["id"])]
        records.append({"dataset":p["dataset"],"id":p["id"],"prediction":p["answer"],
                        "reference":r["reference"],"status":p["status"]})
    # 保存逐条结构化产物到 root / 'aligned_references.jsonl'，供后续重放或审核读取。
    write_jsonl(root/"aligned_references.jsonl",records)


# 严格对齐 FeTaQA 参考与预测，把失败或无效文本按协议计入。
# 输入：predictions（完整预测集合）；samples（固定样本集合）。
# 返回形式含：字段 keys、hypotheses、references、failures、invalid_answer_text。
# 异常约束：Freeform metrics require exact coverage, including failures
def freeform_texts(predictions,samples):
    """Exact FeTaQA ID alignment. Labels/highlights remain offline-only data."""
    from .analysis import index_records
    refs=index_records([r for r in samples if r['dataset']=='fetaqa'],'freeform sample')
    preds=index_records([r for r in predictions if r['dataset']=='fetaqa'],'freeform prediction')
    if set(refs)!=set(preds):raise ValueError('Freeform metrics require exact coverage, including failures')
    hypotheses=[];references=[];keys=sorted(refs);failed=0;invalid=0
    for key in keys:
        reference=refs[key]['reference'];prediction=preds[key]
        if not isinstance(reference,str) or not reference.strip():raise ValueError('Nonempty freeform reference text required')
        if prediction['status'] not in {'ok','failed'}:raise ValueError('Invalid freeform prediction status')
        answer=prediction.get('answer') if prediction['status']=='ok' else ''
        failed+=prediction['status']=='failed'
        if isinstance(answer,list):answer=' '.join(map(str,answer))
        elif not isinstance(answer,str):invalid+=prediction['status']=='ok';answer=''
        hypotheses.append(answer);references.append(reference)
    return {'keys':keys,'hypotheses':hypotheses,'references':references,'failures':failed,'invalid_answer_text':invalid}


# 计算固定版本的语料 BLEU 与宏平均 ROUGE，保存签名和尺度。
# 输入：predictions（完整预测集合）；samples（固定样本集合）。
# 返回形式含：字段 sacrebleu、sacrebleu_signature、n、failures、invalid_answer_text、metric_protocol、metric_scales。
def freeform_metrics(predictions,samples):
    """Versioned automatic metrics; no human factuality or trajectory approval."""
    aligned=freeform_texts(predictions,samples)
    if not aligned['keys']:return {}
    import importlib.metadata
    import sacrebleu
    from rouge_score import rouge_scorer
    hypotheses=aligned['hypotheses'];references=aligned['references']
    bleu=sacrebleu.metrics.BLEU(); score=bleu.corpus_score(hypotheses,[references])
    rouge=rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'],use_stemmer=True)
    per_example=[rouge.score(r,h) for r,h in zip(references,hypotheses)]
    values={key+'_f1':sum(r[key].fmeasure for r in per_example)/len(per_example) for key in ['rouge1','rouge2','rougeL']}
    return {"sacrebleu":score.score,"sacrebleu_signature":str(bleu.get_signature()),**values,
            'n':len(hypotheses),'failures':aligned['failures'],'invalid_answer_text':aligned['invalid_answer_text'],
            'metric_protocol':'freeform-auto-v2; corpus BLEU; macro ROUGE F1; failure-inclusive',
            'metric_scales':{'sacrebleu':'0..100','rouge1_f1':'0..1','rouge2_f1':'0..1','rougeL_f1':'0..1'},
            'metric_packages':{'sacrebleu':sacrebleu.__version__,'rouge-score':importlib.metadata.version('rouge-score')},
            'human_factual_consistency':None,'semantic_correctness_proven':False,
            "rouge_configuration":"rouge-score; stemming=True; default tokenizer",
            "comparability":"Check paper/official metric settings before direct comparison."}


# 对两组相同样本做配对重采样，每次重新计算语料级指标。
# 输入：predictions_a；predictions_b；samples（固定样本集合）；repeats（重采样次数）；seed（随机种子）。
# 返回形式含：字段 n、repeats、seed、uncertainty_scope、metrics。
# 异常约束：Nonempty paired freeform inputs required
def compare_freeform(predictions_a,predictions_b,samples,repeats=500,seed=42):
    """Paired example bootstrap: corpus BLEU is recomputed on each resample."""
    a=freeform_texts(predictions_a,samples);b=freeform_texts(predictions_b,samples)
    if not a['keys'] or a['keys']!=b['keys']:raise ValueError('Nonempty paired freeform inputs required')
    if type(repeats) is not int or repeats<1:raise ValueError('Positive bootstrap repeats required')
    import random
    import sacrebleu
    from rouge_score import rouge_scorer
    from .evaluation import quantile
    bleu=sacrebleu.metrics.BLEU();rouge=rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'],use_stemmer=True)
    refs=a['references'];ha=a['hypotheses'];hb=b['hypotheses'];n=len(refs)
    ra=[rouge.score(r,h) for r,h in zip(refs,ha)];rb=[rouge.score(r,h) for r,h in zip(refs,hb)]
    metric_keys=['sacrebleu','rouge1_f1','rouge2_f1','rougeL_f1'];boot={k:[] for k in metric_keys}
    rng=random.Random(seed)
    for _ in range(repeats):
        ids=[rng.randrange(n) for _ in range(n)];rr=[refs[i] for i in ids]
        boot['sacrebleu'].append(bleu.corpus_score([hb[i] for i in ids],[rr]).score-bleu.corpus_score([ha[i] for i in ids],[rr]).score)
        for key in ['rouge1','rouge2','rougeL']:
            boot[key+'_f1'].append(sum(rb[i][key].fmeasure-ra[i][key].fmeasure for i in ids)/n)
    full_a=freeform_metrics(predictions_a,samples);full_b=freeform_metrics(predictions_b,samples)
    return {'n':n,'repeats':repeats,'seed':seed,'uncertainty_scope':'paired examples, not training seeds or human factuality',
            'metrics':{key:{'a':full_a[key],'b':full_b[key],'difference_b_minus_a':full_b[key]-full_a[key],
                            'ci95':[quantile(boot[key],.025),quantile(boot[key],.975)]} for key in metric_keys}}


# 把已有统计渲染成离线 HTML，转义外部文本后展示。
# 输入：evaluation；mechanisms；output（输出位置）。
# 相关调用：Path(output).write_text。
def html_report(evaluation,mechanisms,output):
    rows=[]
    for dataset,values in evaluation["metrics"].items():
        rows.append(f"<h2>{html.escape(dataset)}</h2><pre>{html.escape(json.dumps(values,indent=2,ensure_ascii=False))}</pre>")
    body='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>表格问答实验报告</title>
<style>body{max-width:960px;margin:40px auto;font:16px/1.6 system-ui;padding:0 20px}pre{background:#f4f5f7;padding:16px;overflow:auto}</style>
<h1>表格问答实验报告</h1><p>诊断指标不能替代WikiTQ官方评测；fixture结果不能作为真实实验。失败样本计入统计。</p>'''
    body+=''.join(rows)+"<h2>机制与失败分布</h2><pre>"+html.escape(json.dumps(mechanisms,indent=2,ensure_ascii=False))+"</pre></html>"
    Path(output).write_text(body,encoding="utf-8")
