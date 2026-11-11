# 测试范围：验证对应模块的数据隔离、输入约束与历史回归行为。
# 运行方式：python -m unittest discover -s tests -p test_curriculum_stage.py -v
# 阅读顺序：先看样本或替身模型的构造，再看每个用例前的预期约束。
from copy import deepcopy
import unittest
from tableqa.data import example
from tableqa.inference import system_prompt
from scripts.prepare_curriculum_stage import prepare_records, select_development


# CurriculumStageTests：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
class CurriculumStageTests(unittest.TestCase):
    # 回归用例：test_development_selection_is_label_blind_and_rejects_test
    # 应相等：len(chosen)，40
    # 应相等：[(r['dataset'], r['id']) for r in chosen]，[(r['dataset'], r['id']) for r in select_development(changed)]
    # 应抛出预期异常：ValueError
    def test_development_selection_is_label_blind_and_rejects_test(self):
        rows = [{'dataset':d,'id':str(i),'split':'validation','reference':i%2}
                for d in ['wikitq','tabfact'] for i in range(25)]
        chosen = select_development(rows)
        self.assertEqual(len(chosen),40)
        changed = deepcopy(rows)
        for row in changed: row['reference'] = 'REPLACED'
        self.assertEqual([(r['dataset'],r['id']) for r in chosen],
                         [(r['dataset'],r['id']) for r in select_development(changed)])
        changed[0]['split']='test'
        with self.assertRaises(ValueError):select_development(changed)

    # 回归用例：test_common_prompt_changes_no_targets_or_public_input_and_is_training_only
    # 应相等：result['messages'][0]['content']，system_prompt('B5', True, True, True)
    # 应相等：result['messages'][1]，record['messages'][1]
    # 应相等：result['target']，record['target']
    def test_common_prompt_changes_no_targets_or_public_input_and_is_training_only(self):
        sample=example('tabfact','train','x','t',['a'],[[1]],'There is a row.',1)
        record={'dataset':'tabfact','id':'x','split':'train','kind':'answer',
                'messages':[{'role':'system','content':'old'},{'role':'user','content':'PUBLIC_INPUT'}],
                'target':'UNCHANGED_TARGET'}
        result=prepare_records([record],[sample])[0]
        self.assertEqual(result['messages'][0]['content'],system_prompt('B5',True,True,True))
        self.assertEqual(result['messages'][1],record['messages'][1])
        self.assertEqual(result['target'],record['target'])
        self.assertEqual(record['messages'][0]['content'],'old')
        self.assertEqual(result['sampling_label'],1)
        self.assertNotIn('sampling_label',str(result['messages']))
        with self.assertRaises(ValueError):prepare_records([{**record,'split':'test'}],[sample])


if __name__=='__main__':unittest.main()
