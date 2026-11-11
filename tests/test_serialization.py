# 测试范围：在保留表格信息的前提下转换输入表示，重建状态并适配模型调用。
# 运行方式：python -m unittest discover -s tests -p test_serialization.py -v
# 阅读顺序：先看样本或替身模型的构造，再看每个用例前的预期约束。
import unittest
import json
from copy import deepcopy
from tableqa.core import TableState,execute
from tableqa.serialization import array_view,restore_view,ArrayInputModel

# SerializationTests：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
class SerializationTests(unittest.TestCase):
    # 回归用例：test_round_trip_preserves_duplicate_header_ids_and_values
    # 应相等：restore_view(array_view(original))，original
    # 核验约束：original['rows'][0]，dict
    def test_round_trip_preserves_duplicate_header_ids_and_values(self):
        state=TableState.from_table({'headers':['X','X'],'rows':[[None,'a|b'],['line\nbreak','2,000']]})
        original=state.view(['X','X'])
        self.assertEqual(restore_view(array_view(original)),original)
        self.assertIsInstance(original['rows'][0],dict)
    # 回归用例：test_sorted_ids_and_empty_state_are_preserved
    # 应相等：restore_view(array_view(original))，original
    def test_sorted_ids_and_empty_state_are_preserved(self):
        state=TableState.from_table({'headers':['n'],'rows':[[2],[1]]})
        state=execute(state,{'op':'sort','column':'c0','type':'number'})
        original=state.view(['n']); self.assertEqual(restore_view(array_view(original)),original)
        state=execute(state,{'op':'filter','column':'c0','cmp':'gt','value':5,'type':'number'})
        original=state.view(['n']); self.assertEqual(restore_view(array_view(original)),original)
    # 回归用例：test_ragged_candidate_is_rejected
    # 应抛出预期异常：ValueError
    def test_ragged_candidate_is_rejected(self):
        with self.assertRaises(ValueError):restore_view({'columns':['c0'],'rows':[[1,2]],'row_ids':['r0']})
    # 回归用例：test_model_adapter_preserves_question_and_original_messages
    # 应相等：ArrayInputModel(Spy()).generate(messages)，{'text': 'ok'}
    # 应相等：messages，original
    def test_model_adapter_preserves_question_and_original_messages(self):
        state=TableState.from_table({'headers':['x'],'rows':[[1]]}).view(['x'])
        messages=[{'role':'system','content':'system'},{'role':'user','content':json.dumps({'question':'q','state':state})},
                  {'role':'user','content':'Repair failed operation'}]
        original=deepcopy(messages)
        # Spy：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
        class Spy:
            # 按当前适配接口生成或重放响应，同时保留调用所需的观测状态。
            # 输入：converted。
            # 返回形式含：字段 text。
            def generate(self,converted):
                payload=json.loads(converted[1]['content'])
                assert payload['question']=='q'
                assert payload['state']['rows']==[[1]]
                assert restore_view(payload['state'])==state
                assert converted[0]==messages[0] and converted[2]==messages[2]
                return {'text':'ok'}
        self.assertEqual(ArrayInputModel(Spy()).generate(messages),{'text':'ok'})
        self.assertEqual(messages,original)
