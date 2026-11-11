# 测试范围：验证对应模块的数据隔离、输入约束与历史回归行为。
# 运行方式：python -m unittest discover -s tests -p test_training_alignment.py -v
# 阅读顺序：先看样本或替身模型的构造，再看每个用例前的预期约束。
import unittest
from tableqa.training import align_accumulation_batches
from tableqa.training import prepare_training_data,encode_record,balance_and_budget


# Tokenizer：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
class Tokenizer:
    eos_token_id=99
    # 用确定性的测试聊天模板表示消息，便于核对提示边界和成本。
    # 输入：messages（模型可见消息）；**kwargs。
    # 返回形式含：'PROMPT'。
    def apply_chat_template(self,messages,**kwargs):return 'PROMPT'
    # 用确定性的测试编码规则返回 token 序列，避免依赖实际模型权重。
    # 输入：text（待解析文本）；**kwargs。
    def encode(self,text,**kwargs):return list(range(len(text.split())))

# AlignmentTests：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
class AlignmentTests(unittest.TestCase):
    # 回归用例：test_partial_accumulation_is_explicitly_padded
    # 应相等：len(result)，8
    # 应相等：stats['padding_records']，2
    # 应相等：stats['supervised_tokens_per_epoch']，8
    def test_partial_accumulation_is_explicitly_padded(self):
        rows=[{'labels':[-100,i]} for i in range(6)]
        result,stats=align_accumulation_batches(rows,4)
        self.assertEqual(len(result),8)
        self.assertEqual(stats['padding_records'],2)
        self.assertEqual(stats['supervised_tokens_per_epoch'],8)
    # 回归用例：test_budget_is_respected_after_padding
    # 应相等：len(result)，4
    # 应相等：stats['supervised_tokens_per_epoch']，4
    # 应抛出预期异常：ValueError
    def test_budget_is_respected_after_padding(self):
        rows=[{'labels':[-100,i]} for i in range(6)]
        result,stats=align_accumulation_batches(rows,4,6)
        self.assertEqual(len(result),4)
        self.assertEqual(stats['supervised_tokens_per_epoch'],4)
        with self.assertRaises(ValueError): align_accumulation_batches(rows,4,3)

    # 回归用例：test_audit_preserves_legacy_sampling_and_records_actual_source_coverage
    # 应相等：actual，legacy
    # 应相等：stats['source_samples_before_balancing']，10
    # 应相等：stats['source_samples_after_alignment']，len({r['source_key'] for r in actual})
    def test_audit_preserves_legacy_sampling_and_records_actual_source_coverage(self):
        records=[{'id':str(i),'dataset':'wikitq','kind':'block' if i<8 else 'answer',
                  'messages':[],'target':'one two'} for i in range(10)]
        config={'seed':42,'micro_batch':1,'gradient_accumulation_steps':4,'target_token_budget':30}
        encoded=[encode_record(r,Tokenizer(),4096) for r in records]
        legacy,_=balance_and_budget(encoded,42,30); legacy,_=align_accumulation_batches(legacy,4,30)
        actual,stats=prepare_training_data(config,records,Tokenizer())
        self.assertEqual(actual,legacy)
        self.assertEqual(stats['source_samples_before_balancing'],10)
        self.assertEqual(stats['source_samples_after_alignment'],len({r['source_key'] for r in actual}))
        self.assertEqual(sum(stats['target_tokens_by_kind'].values()),stats['supervised_tokens_per_epoch'])
        self.assertEqual(sum(r['target_tokens'] for r in stats['source_exposure']),stats['supervised_tokens_per_epoch'])
        self.assertEqual(sum(r['record_exposures'] for r in stats['source_exposure']),len(actual))

    # 回归用例：test_answer_only_padding_is_explicit_in_token_and_exposure_totals
    # 应相等：stats['padding_records']，1
    # 应相等：stats['source_samples_after_alignment']，3
    # 应相等：stats['target_tokens_by_kind']，{'answer': 8}
    def test_answer_only_padding_is_explicit_in_token_and_exposure_totals(self):
        records=[{'id':str(i),'dataset':'tabfact','kind':'answer','messages':[],'target':'one'} for i in range(3)]
        _,stats=prepare_training_data({'gradient_accumulation_steps':4},records,Tokenizer())
        self.assertEqual(stats['padding_records'],1)
        self.assertEqual(stats['source_samples_after_alignment'],3)
        self.assertEqual(stats['target_tokens_by_kind'],{'answer':8})

    # 回归用例：test_coverage_policy_retains_every_original_record_before_budget
    # 应相等：{r['labels'][-1] for r in actual}，set(range(23))
    # 应相等：sum((r['kind_id'] == 0 for r in actual))，20
    # 应相等：sum((r['kind_id'] == 1 for r in actual))，20
    def test_coverage_policy_retains_every_original_record_before_budget(self):
        rows=[{'kind_id':0 if i<20 else 1,'labels':[-100,i]} for i in range(23)]
        actual,total=balance_and_budget(rows,42,sampling_policy='coverage')
        self.assertEqual({r['labels'][-1] for r in actual},set(range(23)))
        self.assertEqual(sum(r['kind_id']==0 for r in actual),20)
        self.assertEqual(sum(r['kind_id']==1 for r in actual),20)
        self.assertEqual(total,40)
        self.assertEqual(balance_and_budget(rows,42,sampling_policy='coverage'),(actual,total))

    # 回归用例：test_coverage_policy_still_reports_budget_drops_honestly
    # 核验约束：stats['supervised_tokens_per_epoch']，12
    # 左值应更小：stats['source_samples_after_alignment']，stats['source_samples_before_balancing']
    def test_coverage_policy_still_reports_budget_drops_honestly(self):
        records=[{'id':str(i),'dataset':'wikitq','kind':'block' if i<8 else 'answer',
                  'messages':[],'target':'one two'} for i in range(10)]
        _,stats=prepare_training_data({'sampling_policy':'coverage','target_token_budget':12,
            'gradient_accumulation_steps':4},records,Tokenizer())
        self.assertLessEqual(stats['supervised_tokens_per_epoch'],12)
        self.assertLess(stats['source_samples_after_alignment'],stats['source_samples_before_balancing'])

    # 回归用例：test_single_kind_order_is_unchanged_between_policies
    # 应相等：balance_and_budget(rows, 42, sampling_policy='replacement')，balance_and_budget(rows, 42, sampling_policy='coverage')
    def test_single_kind_order_is_unchanged_between_policies(self):
        rows=[{'kind_id':1,'labels':[-100,i]} for i in range(10)]
        self.assertEqual(balance_and_budget(rows,42,sampling_policy='replacement'),
                         balance_and_budget(rows,42,sampling_policy='coverage'))
