import unittest
from tableqa.data import example,table_fingerprint
from tableqa.tuning import accuracy_cohorts


class TuningTests(unittest.TestCase):
    def records(self):
        return [example(d,'validation',f'{d}-{i}-{q}',str(i),['v'],[[f'{d}-{i}']],f'q{q}',q)
                for d in ['wikitq','tabfact'] for i in range(12) for q in range(2)]
    def test_whole_table_independence_previous_exclusion_and_determinism(self):
        rows=self.records();prior=[rows[0]];groups=accuracy_cohorts(rows,prior,3)
        self.assertEqual(groups,accuracy_cohorts(list(reversed(rows)),prior,3))
        a={table_fingerprint(r['table']) for r in groups['tune']}
        b={table_fingerprint(r['table']) for r in groups['confirmation']}
        self.assertEqual(len(a),6);self.assertEqual(len(b),6);self.assertFalse(a&b)
        self.assertNotIn(table_fingerprint(prior[0]['table']),a|b)
    def test_labels_do_not_choose_inputs_and_insufficient_data_fails(self):
        rows=self.records();changed=[{**r,'reference':'different'} for r in rows]
        a=accuracy_cohorts(rows,[],3);b=accuracy_cohorts(changed,[],3)
        self.assertEqual([r['id'] for r in a['tune']],[r['id'] for r in b['tune']])
        with self.assertRaises(ValueError):accuracy_cohorts(rows,[],7)
        with self.assertRaises(ValueError):accuracy_cohorts([{**rows[0],'split':'test'}],[],1)
