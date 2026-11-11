# 测试范围：验证对应模块的数据隔离、输入约束与历史回归行为。
# 运行方式：python -m unittest discover -s tests -p test_pipeline.py -v
# 阅读顺序：先看样本或替身模型的构造，再看每个用例前的预期约束。
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


# FakeTokenizer：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
class FakeTokenizer:
    eos_token_id=999
    # 用确定性的测试聊天模板表示消息，便于核对提示边界和成本。
    # 输入：messages（模型可见消息）；**kwargs。
    # 返回形式含：'PROMPT'。
    def apply_chat_template(self,messages,**kwargs): return 'PROMPT'
    # 用确定性的测试编码规则返回 token 序列，避免依赖实际模型权重。
    # 输入：text（待解析文本）；**kwargs。
    def encode(self,text,**kwargs): return list(range(len(text.split()) or 1))


# 构造本组测试所需的表格、样本或轨迹组合。
# 输入：split（数据划分）。
def fixture(split='train'):
    return example('wikitq',split,'x','t',['Name','Score'],[['A',2],['B',3]],'Who has max?',['B'])


# PipelineTests：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
class PipelineTests(unittest.TestCase):
    # 回归用例：test_stepwise_control_uses_same_state_representation
    # 应成立：settings('B2').compact
    # 应成立：settings('B5').compact
    # 应不成立：settings('B4').compact
    def test_stepwise_control_uses_same_state_representation(self):
        self.assertTrue(settings('B2').compact)
        self.assertTrue(settings('B5').compact)
        self.assertFalse(settings('B4').compact)
    # 回归用例：test_direct_baseline_uses_task_prompt_and_matches_training
    # 应相等：record['messages'][0]['content']，system_prompt('B0')
    # 应包含：'1 if the statement is entailed'，system_prompt('B0')
    # 应排除：'Operation schema'，system_prompt('B0')
    def test_direct_baseline_uses_task_prompt_and_matches_training(self):
        from tableqa.inference import system_prompt
        record=build_records(fixture(),[],'B0')[0]
        self.assertEqual(record['messages'][0]['content'],system_prompt('B0'))
        self.assertIn('1 if the statement is entailed',system_prompt('B0'))
        self.assertNotIn('Operation schema',system_prompt('B0'))
    # 回归用例：test_initial_table_is_not_duplicated_in_full_context
    # 应排除：'original_table'，initial
    # 应包含：'original_table'，later
    def test_initial_table_is_not_duplicated_in_full_context(self):
        from tableqa.inference import context
        from tableqa.core import TableState
        sample=fixture(); state=TableState.from_table(sample['table'])
        initial=json.loads(context(sample,state,[],mode='B4'))
        later=json.loads(context(sample,state,[{'op':'sort'}],mode='B4'))
        self.assertNotIn('original_table',initial)
        self.assertIn('original_table',later)
    # 回归用例：test_readable_evidence_recovers_removed_values_without_restoring_columns
    # 应相等：context(sample, initial, [])，context(sample, initial, [], readable_evidence=True)
    # 应相等：enriched.pop('original_evidence_cells')，[{'cell': 'r0:c1', 'header': 'Score', 'value': 2}, {'cell': 'r1:c1', 'header': 'Score', 'value': 3}]
    # 应相等：enriched，original
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
    # 回归用例：test_wtq_escape_sequences
    # 应相等：wtq_unescape('A\\pB')，'A|B'
    # 应相等：wtq_unescape('A\\\\nB')，'A\\nB'
    # 应相等：wtq_unescape('A\\nB')，'A\nB'
    def test_wtq_escape_sequences(self):
        self.assertEqual(wtq_unescape(r'A\pB'),'A|B')
        self.assertEqual(wtq_unescape(r'A\\nB'),r'A\nB')
        self.assertEqual(wtq_unescape(r'A\nB'),'A\nB')
    # 回归用例：test_prompt_and_padding_mask_boundary
    # 应相等：encoded['labels']，[-100, 0, 1, 2, 999]
    # 应抛出预期异常：ValueError
    def test_prompt_and_padding_mask_boundary(self):
        record={'id':'x','messages':[],'target':'one two three','kind':'block'}
        encoded=encode_record(record,FakeTokenizer(),20)
        self.assertEqual(encoded['labels'],[-100,0,1,2,999])
        with self.assertRaises(ValueError): encode_record(record,FakeTokenizer(),2)
    # 回归用例：test_budget_applied_after_balancing
    # 核验约束：total，3
    # 应成立：chosen
    def test_budget_applied_after_balancing(self):
        rows=[{'kind_id':0,'labels':[-100,1,2]},{'kind_id':1,'labels':[-100,3,4,5]}]
        chosen,total=balance_and_budget(rows,42,3)
        self.assertLessEqual(total,3); self.assertTrue(chosen)
    # 回归用例：test_no_target_slicing
    # 应相等：len(chosen)，1
    # 应相等：stats['supervised_target_tokens']，2
    def test_no_target_slicing(self):
        records=[{'split':'train','target':'one two three'},{'split':'train','target':'one'}]
        chosen,stats=budget_records(records,FakeTokenizer(),3)
        self.assertEqual(len(chosen),1); self.assertEqual(stats['supervised_target_tokens'],2)
    # 回归用例：test_semantic_review_required
    # 应相等：changed['reference']，fixture()['reference']
    # 应抛出预期异常：ValueError
    def test_semantic_review_required(self):
        with self.assertRaises(ValueError): perturb(fixture(),'row_shuffle')
        changed=perturb(fixture(),'column_shuffle',approved=True)
        self.assertEqual(changed['reference'],fixture()['reference'])
    # 回归用例：test_replacement_is_diagnostic_only
    # 应不成立：result['eligible_for_main_metrics']
    # 应抛出预期异常：ValueError
    def test_replacement_is_diagnostic_only(self):
        ops=[{'op':'answer','columns':['c0']}]
        with self.assertRaises(ValueError): replacement_diagnostic(fixture(),ops,0,ops[0])
        result=replacement_diagnostic(fixture(),ops,0,ops[0],True)
        self.assertFalse(result['eligible_for_main_metrics'])
    # 回归用例：test_cached_generation_flag
    # 应不成立：model.generate([])['cache_hit']
    # 应成立：model.generate([])['cache_hit']
    def test_cached_generation_flag(self):
        with tempfile.TemporaryDirectory() as d:
            model=CachedModel(ReplayModel([{'answer':'A'}]),d,{'model':'fixture','revision':'1'})
            self.assertFalse(model.generate([])['cache_hit'])
            self.assertTrue(model.generate([])['cache_hit'])
    # 回归用例：test_adapters
    # 应相等：adapt('wikitq', p / 'wtq.tsv', 'train', p)[0]['reference']，['A']
    # 应相等：adapt('tabfact', p / 'tab.json', 'test', p)[0]['reference']，1
    # 应相等：adapt('fetaqa', p / 'feta.json', 'validation')[0]['table']['rows']，[['A']]
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
    # 回归用例：test_training_inference_full_history_identical
    # 应相等：records[0]['messages']，model.messages[0]
    # 应相等：records[1]['messages']，model.messages[1]
    def test_training_inference_full_history_identical(self):
        ops=[{'op':'select','column':'c1','which':'max'},{'op':'answer','columns':['c0']}]
        records=build_records(fixture(),ops,'B2')
        # Spy：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
        class Spy(ReplayModel):
            # 初始化 Spy 的依赖及持久状态；具体预算和身份字段见赋值。
            def __init__(self): super().__init__([{'operations':[ops[0]]},{'operations':[ops[1]]}]); self.messages=[]
            # 按当前适配接口生成或重放响应，同时保留调用所需的观测状态。
            # 输入：messages（模型可见消息）。
            # 相关调用：super().generate。
            def generate(self,messages): self.messages.append(messages); return super().generate(messages)
        model=Spy(); predict(fixture(),model,settings('B2'))
        self.assertEqual(records[0]['messages'],model.messages[0])
        self.assertEqual(records[1]['messages'],model.messages[1])
    # 回归用例：test_feedback_violation_fails
    # 应相等：p['status']，'failed'
    def test_feedback_violation_fails(self):
        ops=[{'op':'select','column':'c1'},{'op':'answer','columns':['c0'],'requires_feedback':True}]
        p=predict(fixture(),ReplayModel([{'operations':ops}]),settings('B5',max_repairs=0))
        self.assertEqual(p['status'],'failed')
    # 回归用例：test_demo_report_and_manifest_cli
    # 应成立：(Path(d) / 'report/report.html').exists()
    # 左值应更大：len(read_jsonl(Path(d) / 'manifest.jsonl'))，100
    def test_demo_report_and_manifest_cli(self):
        with tempfile.TemporaryDirectory() as d:
            main(['demo','--output',d])
            main(['report',str(Path(d)/'examples.jsonl'),str(Path(d)/'predictions.jsonl'),str(Path(d)/'report')])
            self.assertTrue((Path(d)/'report/report.html').exists())
            main(['manifest',str(Path(d)/'manifest.jsonl')])
            self.assertGreater(len(read_jsonl(Path(d)/'manifest.jsonl')),100)
    # 回归用例：test_resume_identity_guard
    # 应相等：len(read_jsonl(p / 'out.jsonl'))，1
    # 应抛出预期异常：ValueError
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
    # 回归用例：test_failed_freeform_scored_zero
    # 应相等：report['fetaqa']['diagnostic_score']，0
    def test_failed_freeform_scored_zero(self):
        row={**fixture('test'),'dataset':'fetaqa','task':'freeform','reference':'A'}
        p=predict(row,ReplayModel(['bad']),settings('B0'))
        report=evaluate([p],[row]); self.assertEqual(report['fetaqa']['diagnostic_score'],0)
    # 回归用例：test_validation_records_keep_their_split
    # 应相等：records[0]['split']，'validation'
    # 应抛出预期异常：ValueError
    def test_validation_records_keep_their_split(self):
        records=build_records(fixture('validation'),[{'op':'answer','columns':['c0']}],allow_validation=True)
        self.assertEqual(records[0]['split'],'validation')
        with self.assertRaises(ValueError):
            build_records(fixture('test'),[],allow_validation=True)
    # 回归用例：test_failed_operation_index_is_preserved
    # 应相等：result['errors'][0]['step']，1
    # 应相等：result['operations']，[]
    def test_failed_operation_index_is_preserved(self):
        ops=[{'op':'project','columns':['c0','c1']},
             {'op':'derive','as':'bad','expression':{'op':'div','args':[1,0]}}]
        result=predict(fixture(),ReplayModel([{'operations':ops}]),settings('B5',max_repairs=0))
        self.assertEqual(result['errors'][0]['step'],1)
        self.assertEqual(result['operations'],[])


if __name__=='__main__': unittest.main()
