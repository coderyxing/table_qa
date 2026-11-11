import importlib.util
import math
import unittest
from tableqa.reporting import freeform_texts,freeform_metrics,compare_freeform


AVAILABLE=importlib.util.find_spec('sacrebleu') is not None and importlib.util.find_spec('rouge_score') is not None


def data():return [{'dataset':'fetaqa','id':uid,'reference':'the red team won six games'} for uid in ['a','b']]
def predictions():return [{'dataset':'fetaqa','id':uid,'status':'ok','answer':'the red team won six games'} for uid in ['b','a']]


class FreeformAlignmentTests(unittest.TestCase):
    def test_missing_duplicate_and_unknown_outputs_cannot_inflate_scores(self):
        for rows in [predictions()[:1],predictions()+predictions()[:1],[{**p,'id':'x'} for p in predictions()]]:
            with self.assertRaises(ValueError):freeform_texts(rows,data())
        aligned=freeform_texts(predictions(),data())
        self.assertEqual(aligned['keys'],[('fetaqa','a'),('fetaqa','b')])

    def test_failures_are_empty_even_when_answer_field_contains_gold(self):
        rows=predictions();rows[0]['status']='failed'
        result=freeform_texts(rows,data())
        self.assertEqual(result['hypotheses'][1],'');self.assertEqual(result['failures'],1)
        rows=predictions();rows[0]['answer']={'text':'the red team won six games'}
        self.assertEqual(freeform_texts(rows,data())['invalid_answer_text'],1)

    def test_transfer_selection_reads_no_gold_or_highlights_and_rejects_test(self):
        from copy import deepcopy
        from scripts.fetaqa_transfer_pilot import select_samples
        rows=[{'dataset':'fetaqa','split':'validation','id':str(i),'reference':'PRIVATE',
               'table':{'headers':['x'],'rows':[[str(i)]]},'highlighted_cells':[[0,0]]} for i in range(16)]
        first=select_samples(rows);changed=deepcopy(rows)
        for r in changed:r['reference']='changed';r['highlighted_cells']=[]
        self.assertEqual([r['id'] for r in first],[r['id'] for r in select_samples(changed)])
        changed[0]['split']='test'
        with self.assertRaises(ValueError):select_samples(changed)


@unittest.skipUnless(AVAILABLE,'Optional automatic-metric dependencies are not installed')
class FreeformActualMetricTests(unittest.TestCase):
    def test_four_metrics_and_failed_case_denominator(self):
        rows=predictions();full=freeform_metrics(rows,data())
        self.assertAlmostEqual(full['sacrebleu'],100)
        for key in ['rouge1_f1','rouge2_f1','rougeL_f1']:self.assertAlmostEqual(full[key],1)
        rows[0]['status']='failed';half=freeform_metrics(rows,data())
        self.assertAlmostEqual(half['sacrebleu'],100/math.e)
        for key in ['rouge1_f1','rouge2_f1','rougeL_f1']:self.assertAlmostEqual(half[key],.5)
        self.assertIsNone(half['human_factual_consistency']);self.assertFalse(half['semantic_correctness_proven'])

    def test_identical_paired_outputs_have_zero_differences_and_intervals(self):
        result=compare_freeform(predictions(),predictions(),data(),repeats=10)
        for metric in result['metrics'].values():
            self.assertEqual(metric['difference_b_minus_a'],0);self.assertEqual(metric['ci95'],[0,0])


if __name__=='__main__':unittest.main()
