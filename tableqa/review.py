"""Human review packets and hash-bound training approvals. No automatic reviewer."""
from collections import Counter
from copy import deepcopy
import html
import json
from .analysis import index_records
from .data import digest

SCHEMA = 'tableqa-human-review-v1'
CHECKS = ('answer_faithful', 'answer_complete', 'evidence_valid',
          'evidence_complete', 'operations_semantically_valid', 'source_annotation_conflict')
REVIEW_FIELDS = {'trajectory_review_approved', 'trajectory_review_approval'}
FILTER_SPLITS = {'train_filter': 'train', 'validation_filter': 'validation'}


def unreviewed(sample):
    return {k: v for k, v in sample.items() if k not in REVIEW_FIELDS}


def make_packet(samples, predictions, purpose='train_filter'):
    if purpose not in {*FILTER_SPLITS, 'evaluation'}:
        raise ValueError('Explicit train_filter, validation_filter or evaluation purpose required')
    refs = index_records(samples, 'sample'); preds = index_records(predictions, 'prediction')
    if set(refs) != set(preds): raise ValueError('Exact source/prediction coverage required')
    items = []
    for key in sorted(refs):
        sample = unreviewed(refs[key]); prediction = preds[key]
        if sample['task'] != 'freeform': continue
        if purpose in FILTER_SPLITS and sample['split'] != FILTER_SPLITS[purpose]:
            raise ValueError('Review filtering purpose does not match sample split')
        payload = {'sample': sample, 'prediction': deepcopy(prediction)}
        items.append({**payload, 'item_sha256': digest(payload)})
    if not items: raise ValueError('No freeform examples to review')
    return {'schema': SCHEMA, 'purpose': purpose, 'items': items,
            'packet_sha256': digest({'purpose': purpose, 'items': items})}


def validate_packet(packet):
    if packet.get('schema') != SCHEMA or packet.get('purpose') not in {*FILTER_SPLITS, 'evaluation'}:
        raise ValueError('Unknown review packet schema/purpose')
    if packet.get('packet_sha256') != digest({'purpose': packet['purpose'], 'items': packet['items']}):
        raise ValueError('Review packet changed; export a new packet')
    keys = set()
    for item in packet['items']:
        sample, prediction = item['sample'], item['prediction']
        key = (sample['dataset'], sample['id'])
        if key in keys or key != (prediction['dataset'], prediction['id']):
            raise ValueError('Duplicate or mismatched review item')
        keys.add(key)
        if sample['task'] != 'freeform' or (packet['purpose'] in FILTER_SPLITS and sample['split'] != FILTER_SPLITS[packet['purpose']]):
            raise ValueError('Review task/split violates purpose')
        if item['item_sha256'] != digest({'sample': sample, 'prediction': prediction}):
            raise ValueError('Review item changed')
    if not keys: raise ValueError('Empty review packet')


def review_template(packet):
    validate_packet(packet)
    return [{'dataset': item['sample']['dataset'], 'id': item['sample']['id'],
             'item_sha256': item['item_sha256'], 'packet_sha256': packet['packet_sha256'],
             'reviewer': None, 'decision': None, 'checks': dict.fromkeys(CHECKS), 'notes': ''}
            for item in packet['items']]


def _validated_reviews(packet, rows):
    expected = index_records(review_template(packet), 'template')
    actual = index_records(rows, 'review')
    if set(expected) != set(actual): raise ValueError('Review coverage must match the complete packet')
    names = set()
    for key, row in actual.items():
        if any(row.get(k) != expected[key][k] for k in ('item_sha256', 'packet_sha256')):
            raise ValueError('Stale review: packet or item binding differs')
        name = row.get('reviewer')
        if not isinstance(name, str) or not name.strip(): raise ValueError('Named human reviewer required')
        names.add(name.strip().casefold())
        if row.get('decision') not in {'approve', 'reject', 'uncertain'}: raise ValueError('Review is incomplete')
        checks = row.get('checks', {})
        if set(checks) != set(CHECKS) or any(v is not None and type(v) is not bool for v in checks.values()):
            raise ValueError('Review checks must be booleans or null')
        if row['decision'] != 'uncertain' and any(v is None for v in checks.values()):
            raise ValueError('Decisive reviews require every check')
        acceptable = all(checks[k] is True for k in CHECKS[:-1]) and checks[CHECKS[-1]] is False
        if row['decision'] == 'approve' and not acceptable: raise ValueError('Approval conflicts with review checks')
        if row['decision'] == 'reject' and acceptable: raise ValueError('Rejection conflicts with review checks')
        if not isinstance(row.get('notes'), str) or (row['decision'] != 'approve' and not row['notes'].strip()):
            raise ValueError('Rejection/uncertainty requires notes')
    if len(names) != 1: raise ValueError('Each review file must identify one reviewer')
    return actual, next(iter(names))


def review_approved(sample, prediction=None, operations=None):
    """Audit binding, not proof of reviewer identity or semantic truth."""
    approval = sample.get('trajectory_review_approval') or {}
    if sample.get('trajectory_review_approved') is not True: return False
    names = approval.get('reviewers', [])
    if (approval.get('schema') != SCHEMA or approval.get('purpose') not in FILTER_SPLITS
            or sample.get('split') != FILTER_SPLITS.get(approval.get('purpose')) or len(names) != 2
            or any(not isinstance(n, str) or not n.strip() for n in names)
            or len({n.strip().casefold() for n in names}) != 2
            or approval.get('sample_sha256') != digest(unreviewed(sample))): return False
    if prediction is not None and approval.get('prediction_sha256') != digest(prediction): return False
    if operations is not None and approval.get('operations_sha256') != digest(operations): return False
    return bool(approval.get('packet_sha256') and approval.get('item_sha256'))


def merge_reviews(packet, reviewer1, reviewer2):
    validate_packet(packet)
    a, name1 = _validated_reviews(packet, reviewer1); b, name2 = _validated_reviews(packet, reviewer2)
    if name1 == name2: raise ValueError('Two distinct reviewer identities required')
    decisions = []; approved = []; counts = Counter(); disagreements = 0
    for item in packet['items']:
        sample, pred = item['sample'], item['prediction']; key = (sample['dataset'], sample['id'])
        pair = (a[key]['decision'], b[key]['decision'])
        decision = pair[0] if pair[0] == pair[1] else 'disagreement'
        counts[decision] += 1; disagreements += pair[0] != pair[1]
        decisions.append({'dataset': key[0], 'id': key[1], 'decision': decision,
                          'item_sha256': item['item_sha256'], 'reviews': [a[key], b[key]]})
        if decision != 'approve' or packet['purpose'] not in FILTER_SPLITS: continue
        reviewed = deepcopy(sample)
        reviewed['trajectory_review_approved'] = True
        reviewed['trajectory_review_approval'] = {
            'schema': SCHEMA, 'purpose': packet['purpose'], 'packet_sha256': packet['packet_sha256'],
            'item_sha256': item['item_sha256'], 'reviewers': [a[key]['reviewer'].strip(), b[key]['reviewer'].strip()],
            'sample_sha256': digest(sample), 'prediction_sha256': digest(pred),
            'operations_sha256': digest(pred.get('operations', []))}
        # Human agreement does not override a failed/non-replayable execution.
        from .quality import audit_trajectory
        if audit_trajectory(reviewed, pred)['decision'] != 'accepted_reviewed':
            raise ValueError(f'Cannot approve a non-replayable/failed training trajectory: {key}')
        approved.append(reviewed)
    n = len(decisions)
    return {'schema': SCHEMA, 'purpose': packet['purpose'], 'packet_sha256': packet['packet_sha256'],
            'n': n, 'decision_counts': dict(counts), 'decision_agreement': (n-disagreements)/n,
            'identity_limit': 'Names are validated; the tool cannot authenticate independent human reviewers.',
            'semantic_correctness_proven': False, 'decisions': decisions}, approved


def render_packet(packet):
    """Static offline HTML. Annotate separate JSONL templates; no external assets."""
    validate_packet(packet)
    esc = lambda x: html.escape(str(x), quote=True)
    parts = ['<!doctype html><meta charset="utf-8"><title>FeTaQA人工审核</title>',
             '<style>body{max-width:1200px;margin:30px auto;font:16px/1.6 sans-serif;padding:0 16px}'
             'table{border-collapse:collapse}th,td{border:1px solid #aaa;padding:5px;white-space:pre-wrap}'
             '.table{overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f4f6;padding:14px}'
             'article{border-top:3px solid #356;margin-top:40px}.evidence{background:#fff0ad}</style>',
             '<h1>FeTaQA独立人工审核材料</h1><p>两位审核人分别填写各自JSONL，完成前不互看决定。'
             '本工具不自动判断事实正确性。完整表格与原始坐标供核验；黄色仅表示执行器追踪的来源，'
             '不表示证据已由人工确认。来源标注与参考答案也可能存在遗漏。</p>',
             '<p>检查项：答案有依据、答案覆盖充分、证据正确、证据覆盖充分、操作语义正确、来源标注冲突。'
             'approve要求前五项true且冲突false；reject/uncertain需写备注。分歧保留待复核，不自动批准。</p>',
             f'<p>用途：{esc(packet["purpose"])}；packet SHA256：{esc(packet["packet_sha256"])}</p>']
    for i, item in enumerate(packet['items'], 1):
        s, p = item['sample'], item['prediction']; evidence = set(p.get('evidence', []))
        parts += [f'<article><h2>{i}. {esc(s["dataset"])} / {esc(s["id"])}</h2>',
                  f'<p>页面/表格标题：{esc(s.get("title", ""))}</p>',
                  f'<p>问题：{esc(s["question"])}</p><p>参考答案：{esc(s["reference"])}</p>',
                  f'<p>教师答案：{esc(p.get("answer"))}；运行状态：{esc(p["status"])}</p>',
                  '<div class="table"><table><tr><th>row ID</th>']
        parts += [f'<th>c{j}: {esc(h)}</th>' for j, h in enumerate(s['table']['headers'])]
        parts += ['</tr>']
        for ri, row in enumerate(s['table']['rows']):
            parts += [f'<tr><th>r{ri}</th>']
            for ci, value in enumerate(row):
                css = ' class="evidence"' if f'r{ri}:c{ci}' in evidence else ''
                parts += [f'<td{css}>{esc(value)}</td>']
            parts += ['</tr>']
        parts += ['</table></div><h3>操作与来源</h3><pre>',
                  esc(json.dumps({'operations': p.get('operations', []), 'evidence': p.get('evidence', []),
                                  'source_highlighted_cells': s.get('highlighted_cells', []),
                                  'failure': p.get('failure')}, ensure_ascii=False, indent=2)),
                  '</pre></article>']
    return '\n'.join(parts)
