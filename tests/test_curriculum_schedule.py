# 测试范围：验证对应模块的数据隔离、输入约束与历史回归行为。
# 运行方式：python -m unittest discover -s tests -p test_curriculum_schedule.py -v
# 阅读顺序：先看样本或替身模型的构造，再看每个用例前的预期约束。
import unittest
from tableqa.curriculum import match_record_costs, matched_schedules
from tableqa.data import digest
from tableqa.training import prepare_training_data


# Tokenizer：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
class Tokenizer:
    eos_token_id = 999
    # 用确定性的测试聊天模板表示消息，便于核对提示边界和成本。
    # 输入：messages（模型可见消息）；**kwargs。
    # 返回形式含：'prompt'。
    def apply_chat_template(self, messages, **kwargs): return 'prompt'
    # 用确定性的测试编码规则返回 token 序列，避免依赖实际模型权重。
    # 输入：value；**kwargs。
    def encode(self, value, **kwargs): return list(range(len(value)))


# 构造带消息、目标和监督类型的训练记录夹具。
# 输入：uid；kind；cost。
# 返回形式含：字段 id、dataset、split、kind、messages、target。
def record(uid, kind, cost):
    return {'id': uid, 'dataset': 'wikitq', 'split': 'train', 'kind': kind,
            'messages': [], 'target': 'x' * (cost - 1)}


# FrozenCurriculumTests：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
class FrozenCurriculumTests(unittest.TestCase):
    # 回归用例：test_exact_matching_uses_complete_records_and_is_reproducible
    # 应相等：sum((len(r['target']) + 1 for r in matched))，17
    # 应相等：digest(matched)，digest(match_record_costs(pool, [4, 6, 7], Tokenizer(), seed=1))
    # 应成立：all((r in pool for r in matched))
    def test_exact_matching_uses_complete_records_and_is_reproducible(self):
        pool = [record('a', 'block', 3), record('b', 'block', 7)]
        matched = match_record_costs(pool, [4, 6, 7], Tokenizer(), seed=1)
        self.assertEqual(sum(len(r['target']) + 1 for r in matched), 17)
        self.assertEqual(digest(matched), digest(match_record_costs(pool, [4, 6, 7], Tokenizer(), seed=1)))
        self.assertTrue(all(r in pool for r in matched))
        with self.assertRaises(ValueError): match_record_costs(pool, [4, 4], Tokenizer())

    # 回归用例：test_schedule_keeps_all_sources_and_matches_target_budget_and_steps
    # 应相等：len(a)，len(b)
    # 应相等：len(a) % 4，0
    # 应相等：report['control']['target_tokens']，report['treatment']['target_tokens']
    def test_schedule_keeps_all_sources_and_matches_target_budget_and_steps(self):
        real = [record('a', 'block', 3), record('b', 'block', 7), record('c', 'answer', 3)]
        synthetic = [record('s1', 'block', 5), record('s2', 'answer', 3)]
        a, b, report = matched_schedules(real, synthetic, Tokenizer(), multiple=4)
        self.assertEqual(len(a), len(b)); self.assertEqual(len(a) % 4, 0)
        self.assertEqual(report['control']['target_tokens'], report['treatment']['target_tokens'])
        self.assertTrue({'a', 'b', 'c'} <= {r['id'] for r in a})
        self.assertTrue({'a', 'b', 'c', 's1', 's2'} <= {r['id'] for r in b})
        for aa, bb in zip(a, b):
            if aa['schedule_part'] == 'common': self.assertEqual(aa['target'], bb['target'])

    # 回归用例：test_frozen_training_never_resamples_crops_or_pads_schedule
    # 应相等：stats['supervised_tokens_per_epoch']，16
    # 应相等：stats['padding_records']，0
    # 应相等：stats['dropped_budget_records']，0
    def test_frozen_training_never_resamples_crops_or_pads_schedule(self):
        rows = [record('a', 'block', 3), record('b', 'answer', 5)] * 2
        cfg = {'sampling_policy': 'frozen', 'frozen_records_sha256': digest(rows),
               'frozen_supervised_tokens_per_epoch': 16, 'gradient_accumulation_steps': 4}
        _, stats = prepare_training_data(cfg, rows, Tokenizer())
        self.assertEqual(stats['supervised_tokens_per_epoch'], 16)
        self.assertEqual(stats['padding_records'], 0); self.assertEqual(stats['dropped_budget_records'], 0)
        for change in [{'frozen_records_sha256': 'wrong'}, {'frozen_supervised_tokens_per_epoch': 17},
                       {'target_token_budget': 16}, {'gradient_accumulation_steps': 3}]:
            with self.assertRaises(ValueError): prepare_training_data({**cfg, **change}, rows, Tokenizer())


if __name__ == '__main__': unittest.main()
