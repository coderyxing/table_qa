"""Paired, failure-inclusive analysis. Standard library only; no GPU calls."""
from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics
from .evaluation import diagnostic_score,label,paired_bootstrap,quantile
from .data import digest

DIRECT={'A0','A1','B0','B1'}


def read_completed_prefix(path):
    """A live writer may be midway through its last JSONL record. Never edit it."""
    records=[]
    with open(path,encoding='utf-8') as f:
        for line in f:
            if not line.endswith('\n'): break
            if line.strip(): records.append(json.loads(line))
    return records


def index_records(records,description):
    result={}
    for record in records:
        key=(record['dataset'],record['id'])
        if key in result: raise ValueError(f'Duplicate {description}: {key}')
        result[key]=record
    return result


def failure_category(prediction):
    failure=prediction.get('failure') or {}
    text=f"{failure.get('type','')} {failure.get('message','')}".casefold()
    for phrases,category in [
        (['context','configured input','exceeds configured'],'context_limit'),
        (['jsondecode','expected json','extra data'],'invalid_json'),
        (['projection removes future','dependency'],'dependency_violation'),
        (['unavailable column','missing column'],'invalid_reference'),
        (['ambiguous numeric','not a number','non-finite'],'numeric_parse'),
        (['unknown operation','unknown comparator','unsupported selection'],'unsupported_operation'),
        (['budget','no answer within'],'operation_budget'),
        (['assert'],'assertion_contract'),
        (['out of memory'],'gpu_memory')]:
        if any(phrase in text for phrase in phrases): return category
    return 'other_failure'


def validate_run(samples,predictions,allow_partial=False):
    refs=index_records(samples,'sample'); preds=index_records(predictions,'prediction')
    if set(preds)-set(refs): raise ValueError('Unknown prediction IDs')
    missing=set(refs)-set(preds)
    if missing and not allow_partial: raise ValueError('Incomplete run; failures must be recorded, not omitted')
    for key,p in preds.items():
        if p.get('status') not in {'ok','failed'}: raise ValueError(f'Invalid status: {key}')
        latency=p['metrics']['total_seconds']
        if isinstance(latency,bool) or not isinstance(latency,(float,int)) or not math.isfinite(latency) or latency<0:
            raise ValueError(f'Invalid latency: {key}')
    return refs,preds,missing


def record_hash(records):
    return digest(sorted(records,key=lambda r:(r['dataset'],r['id'])))


def official_lookup(report,expected_keys,samples,predictions):
    if report is None: return None
    rows=report['scores']; result={}
    for row in rows:
        key=(row['dataset'],row['id'])
        if key in result or row['score'] not in (0,1): raise ValueError('Invalid official per-example scores')
        result[key]=float(row['score'])
    if set(result)!=expected_keys: raise ValueError('Official WTQ score coverage differs from predictions')
    if not report.get('source_sha256') or not report.get('converted_sha256'):
        raise ValueError('Official evaluator provenance missing')
    sample_rows=[r for r in samples if (r['dataset'],r['id']) in expected_keys]
    prediction_rows=[r for r in predictions if (r['dataset'],r['id']) in expected_keys]
    if report.get('samples_structured_sha256')!=record_hash(sample_rows) or report.get('predictions_structured_sha256')!=record_hash(prediction_rows):
        raise ValueError('Official scores are stale or belong to different samples/predictions; regrade explicitly')
    return result


def primary_score(prediction,sample,official=None):
    if prediction['status']!='ok': return 0.
    if sample['dataset']=='wikitq' and official is not None:
        return official[(sample['dataset'],sample['id'])]
    return diagnostic_score(prediction,sample)


def metric_name(dataset,official):
    if dataset=='wikitq': return 'official_denotation_accuracy' if official is not None else 'DIAGNOSTIC_normalized_set_match'
    if dataset=='tabfact': return 'label_accuracy'
    return 'DIAGNOSTIC_token_F1_NOT_factual_consistency'


def summarize_run(samples,predictions,official_report=None,allow_partial=False):
    refs,preds,missing=validate_run(samples,predictions,allow_partial)
    official=official_lookup(official_report,{k for k in preds if k[0]=='wikitq'},samples,predictions)
    rows=[]
    for dataset in sorted({k[0] for k in refs}):
        keys=sorted(k for k in preds if k[0]==dataset); planned=sum(k[0]==dataset for k in refs)
        if not keys:
            rows.append({'dataset':dataset,'expected':planned,'completed':0,'complete':False}); continue
        ps=[preds[k] for k in keys]; success=[p for p in ps if p['status']=='ok']
        executable=[p for p in success if p.get('answer_source')!='direct_fallback' and p.get('executable_answer',True)]
        modes={p['mode'] for p in ps}
        if len(modes)!=1: raise ValueError('One run cannot mix inference modes within a task')
        mode=next(iter(modes))
        latencies=[p['metrics']['total_seconds'] for p in ps]
        success_latencies=[p['metrics']['total_seconds'] for p in success]
        errors=Counter(failure_category(p) for p in ps if p['status']!='ok')
        input_tokens=[p['metrics'].get('input_tokens') for p in ps]
        output_tokens=[p['metrics'].get('output_tokens') for p in ps]
        row={'dataset':dataset,'split':sorted({refs[k]['split'] for k in keys}),
             'mode':mode,'expected':planned,'completed':len(keys),'complete':len(keys)==planned,
             'metric':metric_name(dataset,official),'observed_score':statistics.mean(primary_score(preds[k],refs[k],official) for k in keys),
             'failed':len(ps)-len(success),'pipeline_success_rate':len(success)/len(ps),
             'executable_chain_rate':None if mode in DIRECT else len(executable)/len(ps),
             'fallback_attempts':sum(bool(p.get('routing',{}).get('fallback_attempted')) for p in ps),
             'fallback_answers':sum(p.get('answer_source')=='direct_fallback' for p in ps),
             'response_format_rate':len(success)/len(ps) if mode in DIRECT else None,
             'all_case_p50_seconds':quantile(latencies,.5),'all_case_p95_seconds':quantile(latencies,.95),
             'success_only_p50_seconds':quantile(success_latencies,.5),
             'mean_calls':statistics.mean(p['metrics']['model_calls'] for p in ps),
             'mean_input_tokens':statistics.mean(input_tokens) if all(t is not None for t in input_tokens) else None,
             'mean_output_tokens':statistics.mean(output_tokens) if all(t is not None for t in output_tokens) else None,
             'cache_hits':sum(p['metrics'].get('cache_hits',0) for p in ps),
             'failure_categories':dict(errors),'human_factual_consistency':None}
        row['valid_output_reference_mismatch']=None if dataset=='fetaqa' else sum(
            preds[k]['status']=='ok' and primary_score(preds[k],refs[k],official)==0 for k in keys)
        if dataset=='tabfact':
            matrix=Counter()
            for k in keys:
                p=preds[k]; truth=label(refs[k]['reference']); guessed=label(p['answer']) if p['status']=='ok' else None
                if truth is None: raise ValueError('Invalid reference verification label')
                matrix[f'true_{truth}_pred_{guessed if guessed is not None else "invalid"}']+=1
            row['confusion_matrix']=dict(matrix)
        rows.append(row)
    return {'partial':bool(missing),'missing_count':len(missing),'rows':rows,
            'samples_structured_sha256':record_hash(samples),'predictions_structured_sha256':record_hash(predictions),
            'interpretation':'Development results when split=train; partial observed scores are not whole-run scores.'}


def compare_runs(samples,predictions_a,predictions_b,official_a=None,official_b=None,repeats=2000):
    refs,a,_=validate_run(samples,predictions_a); _,b,_=validate_run(samples,predictions_b)
    wtq={k for k in refs if k[0]=='wikitq'}
    oa=official_lookup(official_a,wtq,samples,predictions_a); ob=official_lookup(official_b,wtq,samples,predictions_b)
    if (oa is None)!=(ob is None): raise ValueError('Both methods must use the same WTQ scoring protocol')
    output=[]
    for dataset in sorted({k[0] for k in refs}):
        keys=sorted(k for k in refs if k[0]==dataset)
        sa=[primary_score(a[k],refs[k],oa) for k in keys]; sb=[primary_score(b[k],refs[k],ob) for k in keys]
        uncertainty=paired_bootstrap(sa,sb,repeats=repeats)
        all_predictions=list(a.values())+list(b.values())
        caches=sum(p['metrics'].get('cache_hits',0) for p in all_predictions)
        unmatched_timing=any(p['metrics'].get('latency_comparable') is False for p in all_predictions)
        output.append({'dataset':dataset,'n':len(keys),'metric':metric_name(dataset,oa),
            'quality_a':statistics.mean(sa),'quality_b':statistics.mean(sb),**uncertainty,
            'correct_both':sum(x==1 and y==1 for x,y in zip(sa,sb)) if dataset!='fetaqa' else None,
            'a_only_correct':sum(x==1 and y==0 for x,y in zip(sa,sb)) if dataset!='fetaqa' else None,
            'b_only_correct':sum(x==0 and y==1 for x,y in zip(sa,sb)) if dataset!='fetaqa' else None,
            'latency_ratio_a_over_b':None if caches or unmatched_timing else (quantile([a[k]['metrics']['total_seconds'] for k in keys],.5)/
                max(quantile([b[k]['metrics']['total_seconds'] for k in keys],.5),1e-12)),
            'latency_comparability':'Observed ratio only; requires matched hardware/backend, warm-up and quality/failure accounting.',
            'uncertainty_scope':'Paired example bootstrap, NOT training-seed variability.'})
    return output


def write_analysis(directory,summaries,comparisons=None):
    root=Path(directory); root.mkdir(parents=True,exist_ok=True)
    payload={'runs':summaries,'paired_comparisons':comparisons or []}
    (root/'analysis.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    flat=[]
    for name,summary in summaries.items():
        for row in summary['rows']:
            flat.append({'run':name,**{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()}})
    fields=sorted(set().union(*(r.keys() for r in flat))) if flat else ['run']
    with (root/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(flat)
    text=['# 开发实验分析','', '本报告区分任务质量、运行成功率与轨迹可执行率。训练集开发样本不作为正式测试成绩；自由文本自动分数不证明事实一致性。','',
          '|运行|任务|完成/计划|指标|分数|失败|全样本P50秒|','|---|---|---|---|---:|---:|---:|']
    for name,summary in summaries.items():
        for r in summary['rows']:
            text.append(f"|{name} ({r.get('mode','pending')})|{r['dataset']}|{r['completed']}/{r['expected']}|{r.get('metric','pending')}|{r.get('observed_score','—')}|{r.get('failed','—')}|{r.get('all_case_p50_seconds','—')}|")
    text+=['','配对差值方向为B−A。置信区间按样本重采样，不能代替多随机种子实验。延迟包含失败及修复；成功样本延迟另存JSON，不能用失败快速退出证明加速。']
    if comparisons:
        text+=['','|任务|配对数量|B−A|样本bootstrap 95%区间|','|---|---:|---:|---|']
        for r in comparisons:text.append(f"|{r['dataset']}|{r['n']}|{r['difference_b_minus_a']}|{r['ci95']}|")
    (root/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
