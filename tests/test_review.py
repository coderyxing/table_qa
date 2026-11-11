# 测试范围：生成独立人工审核材料，以样本和轨迹哈希绑定双人裁决，汇总一致审核结果。
# 运行方式：python -m unittest discover -s tests -p test_review.py -v
# 阅读顺序：先看样本或替身模型的构造，再看每个用例前的预期约束。
from copy import deepcopy
import unittest
from tableqa.data import example
from tableqa.distillation import build_records
from tableqa.experiments import settings
from tableqa.inference import predict, context
from tableqa.core import TableState
from tableqa.models import ReplayModel
from tableqa.quality import audit_trajectory
from tableqa.review import make_packet, review_template, merge_reviews, review_approved, render_packet


# 构造本组测试所需的表格、样本或轨迹组合。
# 输入：split（数据划分）。
# 返回形式含：2 项组成的结果。
def fixture(split='train'):
    s = example('fetaqa', split, 'x', 't', ['Team'], [['<script>A</script>']], 'Which team?', 'A')
    p = predict(s, ReplayModel([{'operations': [{'op': 'answer', 'columns': ['c0'], 'value': 'A'}]}]), settings('A4'))
    return s, p


# 为给定审核材料构造测试裁决，用于一致性和哈希绑定检查。
# 输入：packet；name；decision。
# 返回形式含：rows。
def reviews(packet, name, decision='approve'):
    rows = review_template(packet)
    for r in rows:
        r.update(reviewer=name, decision=decision, notes='' if decision == 'approve' else 'Missing fact.')
        r['checks'] = {k: k != 'source_annotation_conflict' for k in r['checks']}
        if decision == 'reject': r['checks']['answer_complete'] = False
    return rows


# HumanReviewTests：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
class HumanReviewTests(unittest.TestCase):
    # 回归用例：test_export_does_not_approve_and_escapes_table_content
    # 应为 None：rows[0]['decision']
    # 应为 None：rows[0]['reviewer']
    # 应排除：'<script>A</script>'，render_packet(packet)
    def test_export_does_not_approve_and_escapes_table_content(self):
        s, p = fixture(); packet = make_packet([s], [p]); rows = review_template(packet)
        self.assertIsNone(rows[0]['decision']); self.assertIsNone(rows[0]['reviewer'])
        self.assertNotIn('<script>A</script>', render_packet(packet))
        self.assertIn('&lt;script&gt;A&lt;/script&gt;', render_packet(packet))
        self.assertNotIn('trajectory_review_approved', s)

    # 回归用例：test_two_positive_reviews_allow_exact_replayed_training_chain
    # 应相等：report['decision_agreement']，1.0
    # 应相等：audit_trajectory(accepted[0], p)['decision']，'accepted_reviewed'
    # 应成立：review_approved(accepted[0], p, p['operations'])
    def test_two_positive_reviews_allow_exact_replayed_training_chain(self):
        s, p = fixture(); packet = make_packet([s], [p])
        report, accepted = merge_reviews(packet, reviews(packet, 'Reviewer A'), reviews(packet, 'Reviewer B'))
        self.assertEqual(report['decision_agreement'], 1.)
        self.assertEqual(audit_trajectory(accepted[0], p)['decision'], 'accepted_reviewed')
        self.assertTrue(review_approved(accepted[0], p, p['operations']))
        self.assertTrue(build_records(accepted[0], p['operations']))
        public = context(accepted[0], TableState.from_table(s['table']), [], 'A4')
        self.assertNotIn('Reviewer A', public); self.assertNotIn('review_approval', public)

    # 回归用例：test_boolean_alone_cannot_approve
    # 应相等：audit_trajectory(s, p)['decision']，'review_required'
    # 应抛出预期异常：ValueError
    def test_boolean_alone_cannot_approve(self):
        s, p = fixture(); s['trajectory_review_approved'] = True
        self.assertEqual(audit_trajectory(s, p)['decision'], 'review_required')
        with self.assertRaises(ValueError): build_records(s, p['operations'])

    # 回归用例：test_disagreement_and_uncertainty_never_auto_approve
    # 应相等：report['decision_counts']，{'disagreement': 1}
    # 应相等：accepted，[]
    # 应相等：report['decision_counts']，{'uncertain': 1}
    def test_disagreement_and_uncertainty_never_auto_approve(self):
        s, p = fixture(); packet = make_packet([s], [p])
        report, accepted = merge_reviews(packet, reviews(packet, 'A'), reviews(packet, 'B', 'reject'))
        self.assertEqual(report['decision_counts'], {'disagreement': 1}); self.assertEqual(accepted, [])
        report, accepted = merge_reviews(packet, reviews(packet, 'A', 'uncertain'), reviews(packet, 'B', 'uncertain'))
        self.assertEqual(report['decision_counts'], {'uncertain': 1}); self.assertEqual(accepted, [])

    # 回归用例：test_incomplete_same_reviewer_or_bad_checks_are_rejected
    # 应抛出预期异常：ValueError
    def test_incomplete_same_reviewer_or_bad_checks_are_rejected(self):
        s, p = fixture(); packet = make_packet([s], [p]); a = reviews(packet, 'A')
        for b in [review_template(packet), reviews(packet, ' a '), [], a+a]:
            with self.assertRaises(ValueError): merge_reviews(packet, a, b)
        b = reviews(packet, 'B'); b[0]['checks']['answer_faithful'] = False
        with self.assertRaises(ValueError): merge_reviews(packet, a, b)
        b = reviews(packet, 'B', 'reject'); b[0]['notes'] = ''
        with self.assertRaises(ValueError): merge_reviews(packet, a, b)

    # 回归用例：test_stale_reviews_and_changed_packet_are_rejected
    # 应抛出预期异常：ValueError
    def test_stale_reviews_and_changed_packet_are_rejected(self):
        s, p = fixture(); packet = make_packet([s], [p]); a = reviews(packet, 'A'); b = reviews(packet, 'B')
        b[0]['item_sha256'] = 'old'
        with self.assertRaises(ValueError): merge_reviews(packet, a, b)
        changed = deepcopy(packet); changed['items'][0]['prediction']['answer'] = 'new'
        with self.assertRaises(ValueError): review_template(changed)

    # 回归用例：test_approval_does_not_survive_changed_table_prediction_or_operations
    # 应不成立：review_approved(changed, p)
    # 应不成立：review_approved(s, {**p, 'answer': 'new'})
    # 应不成立：review_approved(s, operations=[{'op': 'answer', 'value': 'new'}])
    def test_approval_does_not_survive_changed_table_prediction_or_operations(self):
        s, p = fixture(); packet = make_packet([s], [p])
        _, accepted = merge_reviews(packet, reviews(packet, 'A'), reviews(packet, 'B'))
        s = accepted[0]; changed = deepcopy(s); changed['table']['rows'][0][0] = 'new'
        self.assertFalse(review_approved(changed, p))
        self.assertFalse(review_approved(s, {**p, 'answer': 'new'}))
        self.assertFalse(review_approved(s, operations=[{'op': 'answer', 'value': 'new'}]))

    # 回归用例：test_human_reviews_cannot_override_execution_failure
    # 应抛出预期异常：ValueError
    def test_human_reviews_cannot_override_execution_failure(self):
        s, p = fixture(); p['status'] = 'failed'; packet = make_packet([s], [p])
        with self.assertRaises(ValueError): merge_reviews(packet, reviews(packet, 'A'), reviews(packet, 'B'))

    # 回归用例：test_evaluation_reviews_never_produce_training_approvals
    # 应相等：report['decision_counts']，{'approve': 1}
    # 应相等：accepted，[]
    # 应抛出预期异常：ValueError
    def test_evaluation_reviews_never_produce_training_approvals(self):
        s, p = fixture('test')
        with self.assertRaises(ValueError): make_packet([s], [p], 'train_filter')
        packet = make_packet([s], [p], 'evaluation')
        report, accepted = merge_reviews(packet, reviews(packet, 'A'), reviews(packet, 'B'))
        self.assertEqual(report['decision_counts'], {'approve': 1}); self.assertEqual(accepted, [])

    # 回归用例：test_validation_review_stays_validation_and_requires_explicit_record_permission
    # 应相等：accepted[0]['split']，'validation'
    # 应成立：build_records(accepted[0], p['operations'], allow_validation=True)
    # 应抛出预期异常：ValueError
    def test_validation_review_stays_validation_and_requires_explicit_record_permission(self):
        s, p = fixture('validation'); packet = make_packet([s], [p], 'validation_filter')
        _, accepted = merge_reviews(packet, reviews(packet, 'A'), reviews(packet, 'B'))
        self.assertEqual(accepted[0]['split'], 'validation')
        with self.assertRaises(ValueError): build_records(accepted[0], p['operations'])
        self.assertTrue(build_records(accepted[0], p['operations'], allow_validation=True))

    # 回归用例：test_incomplete_source_or_duplicate_predictions_are_rejected
    # 应抛出预期异常：ValueError
    def test_incomplete_source_or_duplicate_predictions_are_rejected(self):
        s, p = fixture()
        with self.assertRaises(ValueError): make_packet([s], [])
        with self.assertRaises(ValueError): make_packet([s], [p, p])


if __name__ == '__main__': unittest.main()
