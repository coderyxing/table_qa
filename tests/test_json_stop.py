import unittest
from tableqa.models import json_object_complete

class JsonStoppingTests(unittest.TestCase):
    def test_balanced_json_only(self):
        self.assertTrue(json_object_complete('{"answer":"a } brace"}'))
        self.assertFalse(json_object_complete('{"answer":"not done'))
        self.assertFalse(json_object_complete('{"a":1}{"b":2}'))
        self.assertFalse(json_object_complete('[1,2]'))
        self.assertFalse(json_object_complete('```json\n{"a":1}'))
