from copy import deepcopy
import json
import math
import unittest
from pathlib import Path
import tempfile
from tableqa.core import TableState,run_plan
from tableqa.data import example
from tableqa.distillation import build_records,compress
from tableqa.optimization import optimize_operations,state_cost_partition,effective_fixed_blocks
from tableqa.optimization import verify_subsequence_rewrite


class CharacterTokenizer:
    def apply_chat_template(self,messages,**kwargs):
        text=json.dumps(messages,ensure_ascii=False)
        if 'REFERENCE_SENTINEL' in text:raise AssertionError('Reference leaked into partition cost')
        return list(text)
    def encode(self,text,**kwargs):return list(text)


def fixture():
    return example('wikitq','train','x','t',['Name','Long'],
                   [[f'A{i:02}', 'padding'*40] for i in range(20)],'Which is first?',['REFERENCE_SENTINEL'])


def operations():
    return [{'op':'derive','as':'unused','expression':{'op':'add','args':[1,2]}},
            {'op':'project','columns':['c0']},{'op':'sort','column':'c0','type':'text'},
            {'op':'select','mode':'first','n':1},{'op':'answer','columns':['c0']}]


class OptimizationTests(unittest.TestCase):
    def test_search_removes_nonlocal_dead_derive_and_projection(self):
        sample=fixture();state=TableState.from_table(sample['table'])
        ops=[operations()[0],{'op':'project','columns':['c0','c1']},{'op':'answer','columns':['c0','c1']}]
        self.assertEqual(len(compress(state,ops)),3)
        shorter,report=optimize_operations(state,ops)
        self.assertEqual(shorter,[ops[-1]]);self.assertTrue(report['optimal_in_deletion_family'])
        self.assertEqual(run_plan(state,ops)[0].signature(),run_plan(state,shorter)[0].signature())
        self.assertEqual(len(state.columns),2)

    def test_answer_equality_cannot_override_evidence_loss(self):
        state=TableState.from_table({'headers':['name','score'],'rows':[['A',1],['B',2]]})
        ops=[{'op':'sort','column':'c1','type':'number'},{'op':'answer','columns':['c0']}]
        shorter,report=optimize_operations(state,ops)
        self.assertEqual(shorter,ops);self.assertTrue(report['signature_equal'])
        self.assertEqual(run_plan(state,ops)[0].answer,run_plan(state,ops[-1:])[0].answer)
        self.assertNotEqual(run_plan(state,ops)[0].evidence,run_plan(state,ops[-1:])[0].evidence)

    def test_feedback_markers_are_protected_and_budget_is_honest(self):
        state=TableState.from_table({'headers':['v'],'rows':[[1],[2]]})
        ops=[{'op':'project','columns':['c0'],'requires_feedback':True},{'op':'answer','columns':['c0']}]
        self.assertEqual(optimize_operations(state,ops)[0],ops)
        self.assertEqual(compress(state,ops),ops)
        ops=[{'op':'filter','column':'c0','cmp':'gt','value':1,'type':'number'},ops[-1]]
        result,report=optimize_operations(state,ops,max_candidates=1)
        self.assertEqual(result,ops);self.assertTrue(report['candidate_budget_exhausted'])
        self.assertFalse(report['optimal_in_deletion_family'])
        with self.assertRaises(ValueError):optimize_operations(state,ops,max_candidates=True)

    def test_state_cost_changes_boundaries_and_preserves_flat_plan(self):
        sample=fixture();ops=operations()
        aware,a=state_cost_partition(sample,ops,CharacterTokenizer(),3,100000)
        blind,b=state_cost_partition(sample,ops,CharacterTokenizer(),3,100000,state_sensitive=False)
        self.assertEqual([o for block in aware for o in block],ops)
        self.assertNotEqual(a['boundaries'],b['boundaries'])
        self.assertLessEqual(a['cost'],a['fixed_control_cost'])
        self.assertEqual(a['boundaries'],[0,2,4,5])
        self.assertTrue(all(len(block)<=3 for block in aware))
        self.assertEqual(aware[-1],[ops[-1]])

    def test_dp_matches_exhaustive_search_over_feasible_partitions(self):
        _,report=state_cost_partition(fixture(),operations(),CharacterTokenizer(),3,100000)
        edges={(e['start'],e['end']):e['cost'] for e in report['feasible_edges']}
        n=len(operations())
        def enumerate_costs(start):
            if start==n:return [0.]
            return [cost+tail for (i,j),cost in edges.items() if i==start for tail in enumerate_costs(j)]
        self.assertAlmostEqual(report['cost'],min(enumerate_costs(0)))

    def test_feedback_is_not_crossed_even_when_more_calls_are_required(self):
        ops=operations();ops[1]['requires_feedback']=True
        groups,report=state_cost_partition(fixture(),ops,CharacterTokenizer(),4,100000)
        self.assertIn(1,report['boundaries'])
        self.assertFalse(any(o.get('requires_feedback') for group in groups for o in group[1:]))

    def test_budget_and_test_split_are_explicit_constraints(self):
        with self.assertRaises(ValueError):state_cost_partition(fixture(),operations(),CharacterTokenizer(),max_length=1)
        for split in ['validation','test']:
            s={**fixture(),'split':split}
            with self.assertRaises(ValueError):state_cost_partition(s,operations(),CharacterTokenizer(),max_length=100000)
        state_cost_partition({**fixture(),'split':'validation'},operations(),CharacterTokenizer(),max_length=100000,allow_validation=True)
        for weight in [-1,float('nan'),True]:
            with self.assertRaises(ValueError):state_cost_partition(fixture(),operations(),CharacterTokenizer(),input_weight=weight)

    def test_output_budget_is_separate_from_total_training_context(self):
        with self.assertRaises(ValueError):state_cost_partition(fixture(),operations(),CharacterTokenizer(),
            max_length=100000,max_output_tokens=1)
        with self.assertRaises(ValueError):state_cost_partition(fixture(),operations(),CharacterTokenizer(),max_output_tokens=True)

    def test_record_builder_uses_exact_partition_without_reference_or_oracle_inputs(self):
        sample=fixture();ops=operations()
        groups,_=state_cost_partition(sample,ops,CharacterTokenizer(),3,100000)
        rows=build_records(sample,ops,'B5',3,partition=groups)
        # Legacy compression leaves this nontrivial plan unchanged.
        self.assertEqual([o for r in rows for o in json.loads(r['target'])['operations']],ops)
        self.assertNotIn('REFERENCE_SENTINEL',str(rows))
        self.assertEqual(len(rows),len(groups))
        with self.assertRaises(ValueError):build_records(sample,ops,'B5',3,partition=[ops])
        with self.assertRaises(ValueError):build_records(sample,ops,'B5',3,partition=[[ops[-1]]])

    def test_search_supervision_replays_exactly_and_default_records_stay_fixed(self):
        sample=fixture();state=TableState.from_table(sample['table'])
        ops=[operations()[0],{'op':'project','columns':['c0','c1']},{'op':'answer','columns':['c0','c1']}]
        rows=build_records(sample,ops,'B5',compression_policy='search')
        self.assertEqual(len(rows),1)
        self.assertEqual(json.loads(rows[0]['target'])['operations'],[ops[-1]])
        with self.assertRaises(ValueError):build_records(sample,ops,compression_policy='unknown')

    def test_fixed_control_matches_existing_answer_separation(self):
        ops=operations();s=fixture()
        old=build_records(s,ops,'B5',3)
        fixed=build_records(s,ops,'B5',3,partition=effective_fixed_blocks(ops,3))
        self.assertEqual(old,fixed)

    def test_prepared_rewrite_must_preserve_source_order_state_and_feedback(self):
        sample=fixture();state=TableState.from_table(sample['table']);ops=operations()
        for candidate in [ops[-1:],[{'op':'answer','value':'made up'}],list(reversed(ops))]:
            with self.assertRaises(ValueError):verify_subsequence_rewrite(state,ops,candidate)
        no_op=[{'op':'project','columns':['c0','c1'],'requires_feedback':True},ops[-1]]
        with self.assertRaises(ValueError):verify_subsequence_rewrite(state,no_op,no_op[-1:])

    def test_prepared_search_records_match_direct_search_and_keep_original_review_gate(self):
        sample=fixture();state=TableState.from_table(sample['table'])
        ops=[operations()[0],{'op':'project','columns':['c0','c1']},{'op':'answer','columns':['c0','c1']}]
        optimized,_=optimize_operations(state,ops)
        self.assertEqual(build_records(sample,ops,'B5',compression_policy='search'),
                         build_records(sample,ops,'B5',prepared_operations=optimized))
        with self.assertRaises(ValueError):build_records(sample,ops,'B2',prepared_operations=optimized)

    def test_search_cli_writes_real_records_and_audit_metadata(self):
        from tableqa.cli import main
        from tableqa.data import write_jsonl,read_jsonl
        from tableqa.inference import predict
        from tableqa.models import ReplayModel
        from tableqa.experiments import settings
        sample=example('wikitq','train','x','t',['name','score'],[['A',1],['B',2]],'Which names?',['A','B'])
        ops=[operations()[0],{'op':'project','columns':['c0','c1']},{'op':'answer','columns':['c0']}]
        pred=predict(sample,ReplayModel([{'operations':ops}]),settings('A4'))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write_jsonl(root/'samples.jsonl',[sample]);write_jsonl(root/'predictions.jsonl',[pred])
            out=root/'records.jsonl'
            main(['build-records',str(root/'samples.jsonl'),str(root/'predictions.jsonl'),str(out),
                  '--mode','B5','--compression-policy','search'])
            self.assertEqual(len(read_jsonl(out)),1)
            self.assertEqual(json.loads(Path(str(out)+'.filter.json').read_text())['compression_policy'],'search')
            self.assertTrue(read_jsonl(str(out)+'.optimization.jsonl')[0]['compression']['signature_equal'])


if __name__=='__main__':unittest.main()
