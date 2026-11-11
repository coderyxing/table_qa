"""Failure-only answer fallback. A fallback is never an executable evidence chain."""
from copy import deepcopy
import time
from .experiments import settings as inference_settings
from .inference import predict


def combine_recovery(primary, fallback=None, elapsed=None):
    if primary['mode'] in {'A0','A1','B0','B1'}:raise ValueError('Primary must be an operation model')
    if fallback is not None:
        if primary['status']!='failed':raise ValueError('Successful primary answers cannot be replaced')
        if any(primary[k]!=fallback[k] for k in ['id','dataset','task']):raise ValueError('Fallback source differs')
        if fallback['mode'] not in {'A0','B0'}:raise ValueError('Fallback must be an explicit answer-only condition')
    result=deepcopy(primary if fallback is None else fallback)
    result['mode']=primary['mode']+'+answer_fallback'
    result['answer_source']='operations' if primary['status']=='ok' else ('direct_fallback' if fallback and fallback['status']=='ok' else 'failed')
    result['executable_answer']=primary['status']=='ok'
    result['routing']={'fallback_attempted':fallback is not None,'primary_status':primary['status'],
                       'fallback_status':fallback['status'] if fallback else None,'policy':'failure_only_no_reference'}
    result['semantic_verification']='not_proven'
    result['primary_attempt']=deepcopy(primary)
    if fallback is not None:
        result['fallback_attempt']=deepcopy(fallback)
        result['operations']=[];result['trace']=[];result['evidence']=[]
        result['responses']=deepcopy(primary['responses']+fallback['responses'])
        # Actual primary failures and their costs remain in the nested audit trail.
        result['errors']=deepcopy(primary['errors']+fallback['errors'])
        metrics={}
        for key in set(primary['metrics'])|set(fallback['metrics']):
            a=primary['metrics'].get(key);b=fallback['metrics'].get(key)
            if key=='peak_allocated_bytes':metrics[key]=max(a or 0,b or 0)
            elif isinstance(a,(int,float)) and isinstance(b,(int,float)):metrics[key]=a+b
            else:metrics[key]=None
        result['metrics']=metrics
    if elapsed is not None:result['metrics']['total_seconds']=elapsed
    result['metrics']['latency_comparable']=False  # Accuracy protocol, not a matched latency benchmark.
    return result


def predict_with_recovery(sample,primary_model,fallback_model,settings):
    start=time.perf_counter()
    primary=predict(sample,primary_model,settings)
    backup=None
    if primary['status']=='failed':backup=predict(sample,fallback_model,inference_settings('B0'))
    combined=combine_recovery(primary,backup,time.perf_counter()-start)
    return primary,combined
