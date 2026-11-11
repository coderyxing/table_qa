# 测试范围：验证对应模块的数据隔离、输入约束与历史回归行为。
# 运行方式：python -m unittest discover -s tests -p test_json_stop.py -v
# 阅读顺序：先看样本或替身模型的构造，再看每个用例前的预期约束。
import unittest
from tableqa.models import json_object_complete

# JsonStoppingTests：集中验证本模块的协议与回归场景；各用例独立构造输入并核对预期结果。
class JsonStoppingTests(unittest.TestCase):
    # 回归用例：test_balanced_json_only
    # 应成立：json_object_complete('{"answer":"a } brace"}')
    # 应不成立：json_object_complete('{"answer":"not done')
    # 应不成立：json_object_complete('{"a":1}{"b":2}')
    def test_balanced_json_only(self):
        self.assertTrue(json_object_complete('{"answer":"a } brace"}'))
        self.assertFalse(json_object_complete('{"answer":"not done'))
        self.assertFalse(json_object_complete('{"a":1}{"b":2}'))
        self.assertFalse(json_object_complete('[1,2]'))
        self.assertFalse(json_object_complete('```json\n{"a":1}'))
