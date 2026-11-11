"""Offline trajectory checks. References are used here, NEVER in prediction."""
from .core import TableState,run_plan,ExecutionError
from .evaluation import diagnostic_score
from .review import review_approved


def audit_trajectory(sample,prediction,reference_score=None):
    if reference_score is not None and (sample['dataset']!='wikitq' or type(reference_score) not in (int,float) or reference_score not in (0,1)):
        raise ValueError('Official overrides must be binary WikiTQ scores from a bound report')
    reasons=[]
    if prediction.get('status')!='ok': return {'decision':'rejected','reasons':['prediction_failed']}
    ops=prediction.get('operations',[])
    if not ops or ops[-1].get('op')!='answer':
        return {'decision':'rejected','reasons':['no_terminal_answer']}
    terminal=ops[-1]
    if sample['task']=='verification' and ('assert' not in terminal or 'value' in terminal):
        reasons.append('literal_verification_label')
    if sample['task']=='short' and 'value' in terminal:
        reasons.append('short_answer_not_executed')
    try:
        final,_=run_plan(TableState.from_table(sample['table']),ops)
        if final.answer!=prediction.get('answer'): reasons.append('replay_answer_mismatch')
        if set(final.evidence)!=set(prediction.get('evidence',[])): reasons.append('replay_evidence_mismatch')
    except (ExecutionError,KeyError,TypeError,ValueError): reasons.append('replay_failed')
    if sample['task']!='freeform' and (reference_score if reference_score is not None else diagnostic_score(prediction,sample))!=1:
        reasons.append('reference_mismatch_official' if reference_score is not None else 'reference_mismatch_weak_metric')
    if reasons: return {'decision':'rejected','reasons':reasons}
    if sample['task']=='freeform' and not review_approved(sample,prediction):
        return {'decision':'review_required','reasons':['freeform_semantics_needs_human_review']}
    return {'decision':'accepted_reviewed' if sample['task']=='freeform' else ('accepted_official' if reference_score is not None else 'accepted_weak'),
            'reasons':[],'semantic_correctness_proven':False}


def audit_run_trajectories(samples,predictions,official_report=None):
    from .analysis import validate_run,official_lookup
    refs,preds,_=validate_run(samples,predictions)
    official=official_lookup(official_report,{k for k in refs if k[0]=='wikitq'},samples,predictions)
    return [{'dataset':key[0],'id':key[1],**audit_trajectory(refs[key],pred,
            official[key] if official is not None and key[0]=='wikitq' else None)} for key,pred in preds.items()]
