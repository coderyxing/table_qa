import json
from pathlib import Path
import tempfile
import unittest
from tableqa.cli import main
from tableqa.data import adapt,example,write_jsonl,read_jsonl,wtq_unescape
from tableqa.training import encode_record,balance_and_budget
from tableqa.models import CachedModel,ReplayModel
from tableqa.protocols import perturb,budget_records,replacement_diagnostic
from tableqa.inference import predict
from tableqa.distillation import build_records
from tableqa.experiments import settings
from tableqa.evaluation import evaluate


class FakeTokenizer:
    eos_token_id=999
    def apply_chat_template(self,messages,**kwargs): return 'PROMPT'
    def encode(self,text,**kwargs): return list(range(len(text.split()) or 1))


def fixture(split='train'):
    return example('wikitq',split,'x','t',['Name','Score'],[['A',2],['B',3]],'Who has max?',['B'])


class PipelineTests(unittest.TestCase):
    def test_stepwise_control_uses_same_state_representation(self):
        self.assertTrue(settings('B2').compact)
        self.assertTrue(settings('B5').compact)
        self.assertFalse(settings('B4').compact)
    def test_direct_baseline_uses_task_prompt_and_matches_training(self):
        from tableqa.inference import system_prompt
        record=build_records(fixture(),[],'B0')[0]
        self.assertEqual(record['messages'][0]['content'],system_prompt('B0'))
        self.assertIn('1 if the statement is entailed',system_prompt('B0'))
        self.assertNotIn('Operation schema',system_prompt('B0'))
    def test_initial_table_is_not_duplicated_in_full_context(self):
        from tableqa.inference import context
        from tableqa.core import TableState
        sample=fixture(); state=TableState.from_table(sample['table'])
        initial=json.loads(context(sample,state,[],mode='B4'))
        later=json.loads(context(sample,state,[{'op':'sort'}],mode='B4'))
        self.assertNotIn('original_table',initial)
        self.assertIn('original_table',later)
    def test_readable_evidence_recovers_removed_values_without_restoring_columns(self):
        from tableqa.inference import context
        from tableqa.core import TableState, run_plan
        sample = fixture()
        initial = TableState.from_table(sample['table'])
        self.assertEqual(context(sample, initial, []), context(sample, initial, [], readable_evidence=True))
        operations = [{'op': 'select', 'mode': 'extreme', 'column': 'c1', 'which': 'max'},
                      {'op': 'project', 'columns': ['c0']}]
        state, _ = run_plan(initial, operations)
        original = json.loads(context(sample, state, operations))
        enriched = json.loads(context(sample, state, operations, readable_evidence=True))
        self.assertEqual(enriched.pop('original_evidence_cells'), [
            {'cell': 'r0:c1', 'header': 'Score', 'value': 2},
            {'cell': 'r1:c1', 'header': 'Score', 'value': 3},
        ])
        self.assertEqual(enriched, original)
    def test_wtq_escape_sequences(self):
        self.assertEqual(wtq_unescape(r'A\pB'),'A|B')
        self.assertEqual(wtq_unescape(r'A\\nB'),r'A\nB')
        self.assertEqual(wtq_unescape(r'A\nB'),'A\nB')
    def test_prompt_and_padding_mask_boundary(self):
        record={'id':'x','messages':[],'target':'one two three','kind':'block'}
        encoded=encode_record(record,FakeTokenizer(),20)
        self.assertEqual(encoded['labels'],[-100,0,1,2,999])
        with self.assertRaises(ValueError): encode_record(record,FakeTokenizer(),2)
    def test_budget_applied_after_balancing(self):
        rows=[{'kind_id':0,'labels':[-100,1,2]},{'kind_id':1,'labels':[-100,3,4,5]}]
        chosen,total=balance_and_budget(rows,42,3)
        self.assertLessEqual(total,3); self.assertTrue(chosen)
    def test_no_target_slicing(self):
        records=[{'split':'train','target':'one two three'},{'split':'train','target':'one'}]
        chosen,stats=budget_records(records,FakeTokenizer(),3)
        self.assertEqual(len(chosen),1); self.assertEqual(stats['supervised_target_tokens'],2)
    def test_semantic_review_required(self):
        with self.assertRaises(ValueError): perturb(fixture(),'row_shuffle')
        changed=perturb(fixture(),'column_shuffle',approved=True)
        self.assertEqual(changed['reference'],fixture()['reference'])
    def test_replacement_is_diagnostic_only(self):
        ops=[{'op':'answer','columns':['c0']}]
        with self.assertRaises(ValueError): replacement_diagnostic(fixture(),ops,0,ops[0])
        result=replacement_diagnostic(fixture(),ops,0,ops[0],True)
        self.assertFalse(result['eligible_for_main_metrics'])
    def test_cached_generation_flag(self):
        with tempfile.TemporaryDirectory() as d:
            model=CachedModel(ReplayModel([{'answer':'A'}]),d,{'model':'fixture','revision':'1'})
            self.assertFalse(model.generate([])['cache_hit'])
            self.assertTrue(model.generate([])['cache_hit'])
    def test_adapters(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/'t.csv').write_text('Name,Score\nA,2\n',encoding='utf-8')
            (p/'wtq.tsv').write_text('id\tutterance\tcontext\ttargetValue\nx\tWho?\tt.csv\tA\n',encoding='utf-8')
            self.assertEqual(adapt('wikitq',p/'wtq.tsv','train',p)[0]['reference'],['A'])
            (p/'tab.csv').write_text('Name#Score\nA#2\n',encoding='utf-8')
            (p/'tab.json').write_text(json.dumps({'tab.csv':[['A has 2'],[1],'caption']}),encoding='utf-8')
            self.assertEqual(adapt('tabfact',p/'tab.json','test',p)[0]['reference'],1)
            (p/'feta.json').write_text(json.dumps([{'feta_id':1,'table_array':[['N'],['A']], 'question':'Who?','answer':'A'}]),encoding='utf-8')
            self.assertEqual(adapt('fetaqa',p/'feta.json','validation')[0]['table']['rows'],[['A']])
    def test_training_inference_full_history_identical(self):
        ops=[{'op':'select','column':'c1','which':'max'},{'op':'answer','columns':['c0']}]
        records=build_records(fixture(),ops,'B2')
        class Spy(ReplayModel):
            def __init__(self): super().__init__([{'operations':[ops[0]]},{'operations':[ops[1]]}]); self.messages=[]
            def generate(self,messages): self.messages.append(messages); return super().generate(messages)
        model=Spy(); predict(fixture(),model,settings('B2'))
        self.assertEqual(records[0]['messages'],model.messages[0])
        self.assertEqual(records[1]['messages'],model.messages[1])
    def test_feedback_violation_fails(self):
        ops=[{'op':'select','column':'c1'},{'op':'answer','columns':['c0'],'requires_feedback':True}]
        p=predict(fixture(),ReplayModel([{'operations':ops}]),settings('B5',max_repairs=0))
        self.assertEqual(p['status'],'failed')
    def test_demo_report_and_manifest_cli(self):
        with tempfile.TemporaryDirectory() as d:
            main(['demo','--output',d])
            main(['report',str(Path(d)/'examples.jsonl'),str(Path(d)/'predictions.jsonl'),str(Path(d)/'report')])
            self.assertTrue((Path(d)/'report/report.html').exists())
            main(['manifest',str(Path(d)/'manifest.jsonl')])
            self.assertGreater(len(read_jsonl(Path(d)/'manifest.jsonl')),100)
    def test_resume_identity_guard(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); write_jsonl(p/'data.jsonl',[fixture()])
            config={'mode':'B0','model_config':{'backend':'replay','responses':[{'answer':['B']}]}}
            (p/'cfg.json').write_text(json.dumps(config),encoding='utf-8')
            cmd=['predict',str(p/'data.jsonl'),str(p/'cfg.json'),str(p/'out.jsonl')]
            main(cmd); main(cmd+['--resume'])
            self.assertEqual(len(read_jsonl(p/'out.jsonl')),1)
            config['mode']='B1'; (p/'cfg.json').write_text(json.dumps(config),encoding='utf-8')
            with self.assertRaises(ValueError): main(cmd+['--resume'])
    def test_failed_freeform_scored_zero(self):
        row={**fixture('test'),'dataset':'fetaqa','task':'freeform','reference':'A'}
        p=predict(row,ReplayModel(['bad']),settings('B0'))
        report=evaluate([p],[row]); self.assertEqual(report['fetaqa']['diagnostic_score'],0)
    def test_validation_records_keep_their_split(self):
        records=build_records(fixture('validation'),[{'op':'answer','columns':['c0']}],allow_validation=True)
        self.assertEqual(records[0]['split'],'validation')
        with self.assertRaises(ValueError):
            build_records(fixture('test'),[],allow_validation=True)
    def test_failed_operation_index_is_preserved(self):
        ops=[{'op':'project','columns':['c0','c1']},
             {'op':'derive','as':'bad','expression':{'op':'div','args':[1,0]}}]
        result=predict(fixture(),ReplayModel([{'operations':ops}]),settings('B5',max_repairs=0))
        self.assertEqual(result['errors'][0]['step'],1)
        self.assertEqual(result['operations'],[])


if __name__=='__main__': unittest.main()
