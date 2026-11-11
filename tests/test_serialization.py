import unittest
import json
from copy import deepcopy
from tableqa.core import TableState,execute
from tableqa.serialization import array_view,restore_view,ArrayInputModel

class SerializationTests(unittest.TestCase):
    def test_round_trip_preserves_duplicate_header_ids_and_values(self):
        state=TableState.from_table({'headers':['X','X'],'rows':[[None,'a|b'],['line\nbreak','2,000']]})
        original=state.view(['X','X'])
        self.assertEqual(restore_view(array_view(original)),original)
        self.assertIsInstance(original['rows'][0],dict)
    def test_sorted_ids_and_empty_state_are_preserved(self):
        state=TableState.from_table({'headers':['n'],'rows':[[2],[1]]})
        state=execute(state,{'op':'sort','column':'c0','type':'number'})
        original=state.view(['n']); self.assertEqual(restore_view(array_view(original)),original)
        state=execute(state,{'op':'filter','column':'c0','cmp':'gt','value':5,'type':'number'})
        original=state.view(['n']); self.assertEqual(restore_view(array_view(original)),original)
    def test_ragged_candidate_is_rejected(self):
        with self.assertRaises(ValueError):restore_view({'columns':['c0'],'rows':[[1,2]],'row_ids':['r0']})
    def test_model_adapter_preserves_question_and_original_messages(self):
        state=TableState.from_table({'headers':['x'],'rows':[[1]]}).view(['x'])
        messages=[{'role':'system','content':'system'},{'role':'user','content':json.dumps({'question':'q','state':state})},
                  {'role':'user','content':'Repair failed operation'}]
        original=deepcopy(messages)
        class Spy:
            def generate(self,converted):
                payload=json.loads(converted[1]['content'])
                assert payload['question']=='q'
                assert payload['state']['rows']==[[1]]
                assert restore_view(payload['state'])==state
                assert converted[0]==messages[0] and converted[2]==messages[2]
                return {'text':'ok'}
        self.assertEqual(ArrayInputModel(Spy()).generate(messages),{'text':'ok'})
        self.assertEqual(messages,original)
