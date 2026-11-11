# 测试范围：在重放等价和反馈边界约束下搜索教师链子序列，再用状态成本规划离线分块。
# 运行方式：python -m unittest discover -s tests -p test_optimization.py -v
# 阅读顺序：先看样本或替身模型的构造，再看每个用例前的预期约束。
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


# CharacterTokenizer：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
class CharacterTokenizer:
    # 用确定性的测试聊天模板表示消息，便于核对提示边界和成本。
    # 输入：messages（模型可见消息）；**kwargs。
    # 异常约束：Reference leaked into partition cost
    def apply_chat_template(self,messages,**kwargs):
        text=json.dumps(messages,ensure_ascii=False)
        if 'REFERENCE_SENTINEL' in text:raise AssertionError('Reference leaked into partition cost')
        return list(text)
    # 用确定性的测试编码规则返回 token 序列，避免依赖实际模型权重。
    # 输入：text（待解析文本）；**kwargs。
    def encode(self,text,**kwargs):return list(text)


# 构造本组测试所需的表格、样本或轨迹组合。
def fixture():
    return example('wikitq','train','x','t',['Name','Long'],
                   [[f'A{i:02}', 'padding'*40] for i in range(20)],'Which is first?',['REFERENCE_SENTINEL'])


# 返回本组重放与压缩测试使用的明确操作序列。
# 返回形式含：列表记录。
def operations():
    return [{'op':'derive','as':'unused','expression':{'op':'add','args':[1,2]}},
            {'op':'project','columns':['c0']},{'op':'sort','column':'c0','type':'text'},
            {'op':'select','mode':'first','n':1},{'op':'answer','columns':['c0']}]


# OptimizationTests：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
class OptimizationTests(unittest.TestCase):
    # 回归用例：test_search_removes_nonlocal_dead_derive_and_projection
    # 应相等：len(compress(state, ops))，3
    # 应相等：shorter，[ops[-1]]
    # 应成立：report['optimal_in_deletion_family']
    def test_search_removes_nonlocal_dead_derive_and_projection(self):
        sample=fixture();state=TableState.from_table(sample['table'])
        ops=[operations()[0],{'op':'project','columns':['c0','c1']},{'op':'answer','columns':['c0','c1']}]
        self.assertEqual(len(compress(state,ops)),3)
        shorter,report=optimize_operations(state,ops)
        self.assertEqual(shorter,[ops[-1]]);self.assertTrue(report['optimal_in_deletion_family'])
        self.assertEqual(run_plan(state,ops)[0].signature(),run_plan(state,shorter)[0].signature())
        self.assertEqual(len(state.columns),2)

    # 回归用例：test_answer_equality_cannot_override_evidence_loss
    # 应相等：shorter，ops
    # 应成立：report['signature_equal']
    # 应相等：run_plan(state, ops)[0].answer，run_plan(state, ops[-1:])[0].answer
    def test_answer_equality_cannot_override_evidence_loss(self):
        state=TableState.from_table({'headers':['name','score'],'rows':[['A',1],['B',2]]})
        ops=[{'op':'sort','column':'c1','type':'number'},{'op':'answer','columns':['c0']}]
        shorter,report=optimize_operations(state,ops)
        self.assertEqual(shorter,ops);self.assertTrue(report['signature_equal'])
        self.assertEqual(run_plan(state,ops)[0].answer,run_plan(state,ops[-1:])[0].answer)
        self.assertNotEqual(run_plan(state,ops)[0].evidence,run_plan(state,ops[-1:])[0].evidence)

    # 回归用例：test_feedback_markers_are_protected_and_budget_is_honest
    # 应相等：optimize_operations(state, ops)[0]，ops
    # 应相等：compress(state, ops)，ops
    # 应相等：result，ops
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

    # 回归用例：test_state_cost_changes_boundaries_and_preserves_flat_plan
    # 应相等：[o for block in aware for o in block]，ops
    # 应不同：a['boundaries']，b['boundaries']
    # 核验约束：a['cost']，a['fixed_control_cost']
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

    # 回归用例：test_dp_matches_exhaustive_search_over_feasible_partitions
    # 应在浮点容差内相等：report['cost']，min(enumerate_costs(0))
    def test_dp_matches_exhaustive_search_over_feasible_partitions(self):
        _,report=state_cost_partition(fixture(),operations(),CharacterTokenizer(),3,100000)
        edges={(e['start'],e['end']):e['cost'] for e in report['feasible_edges']}
        n=len(operations())
        # 在小规模测试空间枚举分块成本，独立验证动态规划选择。
        # 输入：start。
        # 返回形式含：列表记录。
        def enumerate_costs(start):
            if start==n:return [0.]
            return [cost+tail for (i,j),cost in edges.items() if i==start for tail in enumerate_costs(j)]
        self.assertAlmostEqual(report['cost'],min(enumerate_costs(0)))

    # 回归用例：test_feedback_is_not_crossed_even_when_more_calls_are_required
    # 应包含：1，report['boundaries']
    # 应不成立：any((o.get('requires_feedback') for group in groups for o in group[1:]))
    def test_feedback_is_not_crossed_even_when_more_calls_are_required(self):
        ops=operations();ops[1]['requires_feedback']=True
        groups,report=state_cost_partition(fixture(),ops,CharacterTokenizer(),4,100000)
        self.assertIn(1,report['boundaries'])
        self.assertFalse(any(o.get('requires_feedback') for group in groups for o in group[1:]))

    # 回归用例：test_budget_and_test_split_are_explicit_constraints
    # 应抛出预期异常：ValueError
    def test_budget_and_test_split_are_explicit_constraints(self):
        with self.assertRaises(ValueError):state_cost_partition(fixture(),operations(),CharacterTokenizer(),max_length=1)
        for split in ['validation','test']:
            s={**fixture(),'split':split}
            with self.assertRaises(ValueError):state_cost_partition(s,operations(),CharacterTokenizer(),max_length=100000)
        state_cost_partition({**fixture(),'split':'validation'},operations(),CharacterTokenizer(),max_length=100000,allow_validation=True)
        for weight in [-1,float('nan'),True]:
            with self.assertRaises(ValueError):state_cost_partition(fixture(),operations(),CharacterTokenizer(),input_weight=weight)

    # 回归用例：test_output_budget_is_separate_from_total_training_context
    # 应抛出预期异常：ValueError
    def test_output_budget_is_separate_from_total_training_context(self):
        with self.assertRaises(ValueError):state_cost_partition(fixture(),operations(),CharacterTokenizer(),
            max_length=100000,max_output_tokens=1)
        with self.assertRaises(ValueError):state_cost_partition(fixture(),operations(),CharacterTokenizer(),max_output_tokens=True)

    # 回归用例：test_record_builder_uses_exact_partition_without_reference_or_oracle_inputs
    # 应相等：[o for r in rows for o in json.loads(r['target'])['operations']]，ops
    # 应排除：'REFERENCE_SENTINEL'，str(rows)
    # 应相等：len(rows)，len(groups)
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

    # 回归用例：test_search_supervision_replays_exactly_and_default_records_stay_fixed
    # 应相等：len(rows)，1
    # 应相等：json.loads(rows[0]['target'])['operations']，[ops[-1]]
    # 应抛出预期异常：ValueError
    def test_search_supervision_replays_exactly_and_default_records_stay_fixed(self):
        sample=fixture();state=TableState.from_table(sample['table'])
        ops=[operations()[0],{'op':'project','columns':['c0','c1']},{'op':'answer','columns':['c0','c1']}]
        rows=build_records(sample,ops,'B5',compression_policy='search')
        self.assertEqual(len(rows),1)
        self.assertEqual(json.loads(rows[0]['target'])['operations'],[ops[-1]])
        with self.assertRaises(ValueError):build_records(sample,ops,compression_policy='unknown')

    # 回归用例：test_fixed_control_matches_existing_answer_separation
    # 应相等：old，fixed
    def test_fixed_control_matches_existing_answer_separation(self):
        ops=operations();s=fixture()
        old=build_records(s,ops,'B5',3)
        fixed=build_records(s,ops,'B5',3,partition=effective_fixed_blocks(ops,3))
        self.assertEqual(old,fixed)

    # 回归用例：test_prepared_rewrite_must_preserve_source_order_state_and_feedback
    # 应抛出预期异常：ValueError
    def test_prepared_rewrite_must_preserve_source_order_state_and_feedback(self):
        sample=fixture();state=TableState.from_table(sample['table']);ops=operations()
        for candidate in [ops[-1:],[{'op':'answer','value':'made up'}],list(reversed(ops))]:
            with self.assertRaises(ValueError):verify_subsequence_rewrite(state,ops,candidate)
        no_op=[{'op':'project','columns':['c0','c1'],'requires_feedback':True},ops[-1]]
        with self.assertRaises(ValueError):verify_subsequence_rewrite(state,no_op,no_op[-1:])

    # 回归用例：test_prepared_search_records_match_direct_search_and_keep_original_review_gate
    # 应相等：build_records(sample, ops, 'B5', compression_policy='search')，build_records(sample, ops, 'B5', prepared_operations=optimized)
    # 应抛出预期异常：ValueError
    def test_prepared_search_records_match_direct_search_and_keep_original_review_gate(self):
        sample=fixture();state=TableState.from_table(sample['table'])
        ops=[operations()[0],{'op':'project','columns':['c0','c1']},{'op':'answer','columns':['c0','c1']}]
        optimized,_=optimize_operations(state,ops)
        self.assertEqual(build_records(sample,ops,'B5',compression_policy='search'),
                         build_records(sample,ops,'B5',prepared_operations=optimized))
        with self.assertRaises(ValueError):build_records(sample,ops,'B2',prepared_operations=optimized)

    # 回归用例：test_search_cli_writes_real_records_and_audit_metadata
    # 应相等：len(read_jsonl(out))，1
    # 应相等：json.loads(Path(str(out) + '.filter.json').read_text())['compression_policy']，'search'
    # 应成立：read_jsonl(str(out) + '.optimization.jsonl')[0]['compression']['signature_equal']
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
