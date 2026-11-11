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


def fixture(split='train'):
    s = example('fetaqa', split, 'x', 't', ['Team'], [['<script>A</script>']], 'Which team?', 'A')
    p = predict(s, ReplayModel([{'operations': [{'op': 'answer', 'columns': ['c0'], 'value': 'A'}]}]), settings('A4'))
    return s, p


def reviews(packet, name, decision='approve'):
    rows = review_template(packet)
    for r in rows:
        r.update(reviewer=name, decision=decision, notes='' if decision == 'approve' else 'Missing fact.')
        r['checks'] = {k: k != 'source_annotation_conflict' for k in r['checks']}
        if decision == 'reject': r['checks']['answer_complete'] = False
    return rows


class HumanReviewTests(unittest.TestCase):
    def test_export_does_not_approve_and_escapes_table_content(self):
        s, p = fixture(); packet = make_packet([s], [p]); rows = review_template(packet)
        self.assertIsNone(rows[0]['decision']); self.assertIsNone(rows[0]['reviewer'])
        self.assertNotIn('<script>A</script>', render_packet(packet))
        self.assertIn('&lt;script&gt;A&lt;/script&gt;', render_packet(packet))
        self.assertNotIn('trajectory_review_approved', s)

    def test_two_positive_reviews_allow_exact_replayed_training_chain(self):
        s, p = fixture(); packet = make_packet([s], [p])
        report, accepted = merge_reviews(packet, reviews(packet, 'Reviewer A'), reviews(packet, 'Reviewer B'))
        self.assertEqual(report['decision_agreement'], 1.)
        self.assertEqual(audit_trajectory(accepted[0], p)['decision'], 'accepted_reviewed')
        self.assertTrue(review_approved(accepted[0], p, p['operations']))
        self.assertTrue(build_records(accepted[0], p['operations']))
        public = context(accepted[0], TableState.from_table(s['table']), [], 'A4')
        self.assertNotIn('Reviewer A', public); self.assertNotIn('review_approval', public)

    def test_boolean_alone_cannot_approve(self):
        s, p = fixture(); s['trajectory_review_approved'] = True
        self.assertEqual(audit_trajectory(s, p)['decision'], 'review_required')
        with self.assertRaises(ValueError): build_records(s, p['operations'])

    def test_disagreement_and_uncertainty_never_auto_approve(self):
        s, p = fixture(); packet = make_packet([s], [p])
        report, accepted = merge_reviews(packet, reviews(packet, 'A'), reviews(packet, 'B', 'reject'))
        self.assertEqual(report['decision_counts'], {'disagreement': 1}); self.assertEqual(accepted, [])
        report, accepted = merge_reviews(packet, reviews(packet, 'A', 'uncertain'), reviews(packet, 'B', 'uncertain'))
        self.assertEqual(report['decision_counts'], {'uncertain': 1}); self.assertEqual(accepted, [])

    def test_incomplete_same_reviewer_or_bad_checks_are_rejected(self):
        s, p = fixture(); packet = make_packet([s], [p]); a = reviews(packet, 'A')
        for b in [review_template(packet), reviews(packet, ' a '), [], a+a]:
            with self.assertRaises(ValueError): merge_reviews(packet, a, b)
        b = reviews(packet, 'B'); b[0]['checks']['answer_faithful'] = False
        with self.assertRaises(ValueError): merge_reviews(packet, a, b)
        b = reviews(packet, 'B', 'reject'); b[0]['notes'] = ''
        with self.assertRaises(ValueError): merge_reviews(packet, a, b)

    def test_stale_reviews_and_changed_packet_are_rejected(self):
        s, p = fixture(); packet = make_packet([s], [p]); a = reviews(packet, 'A'); b = reviews(packet, 'B')
        b[0]['item_sha256'] = 'old'
        with self.assertRaises(ValueError): merge_reviews(packet, a, b)
        changed = deepcopy(packet); changed['items'][0]['prediction']['answer'] = 'new'
        with self.assertRaises(ValueError): review_template(changed)

    def test_approval_does_not_survive_changed_table_prediction_or_operations(self):
        s, p = fixture(); packet = make_packet([s], [p])
        _, accepted = merge_reviews(packet, reviews(packet, 'A'), reviews(packet, 'B'))
        s = accepted[0]; changed = deepcopy(s); changed['table']['rows'][0][0] = 'new'
        self.assertFalse(review_approved(changed, p))
        self.assertFalse(review_approved(s, {**p, 'answer': 'new'}))
        self.assertFalse(review_approved(s, operations=[{'op': 'answer', 'value': 'new'}]))

    def test_human_reviews_cannot_override_execution_failure(self):
        s, p = fixture(); p['status'] = 'failed'; packet = make_packet([s], [p])
        with self.assertRaises(ValueError): merge_reviews(packet, reviews(packet, 'A'), reviews(packet, 'B'))

    def test_evaluation_reviews_never_produce_training_approvals(self):
        s, p = fixture('test')
        with self.assertRaises(ValueError): make_packet([s], [p], 'train_filter')
        packet = make_packet([s], [p], 'evaluation')
        report, accepted = merge_reviews(packet, reviews(packet, 'A'), reviews(packet, 'B'))
        self.assertEqual(report['decision_counts'], {'approve': 1}); self.assertEqual(accepted, [])

    def test_validation_review_stays_validation_and_requires_explicit_record_permission(self):
        s, p = fixture('validation'); packet = make_packet([s], [p], 'validation_filter')
        _, accepted = merge_reviews(packet, reviews(packet, 'A'), reviews(packet, 'B'))
        self.assertEqual(accepted[0]['split'], 'validation')
        with self.assertRaises(ValueError): build_records(accepted[0], p['operations'])
        self.assertTrue(build_records(accepted[0], p['operations'], allow_validation=True))

    def test_incomplete_source_or_duplicate_predictions_are_rejected(self):
        s, p = fixture()
        with self.assertRaises(ValueError): make_packet([s], [])
        with self.assertRaises(ValueError): make_packet([s], [p, p])


if __name__ == '__main__': unittest.main()
