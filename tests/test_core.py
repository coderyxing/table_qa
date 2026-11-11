import unittest
from tableqa.core import TableState,ExecutionError,execute,run_plan,validate_plan,number
from tableqa.distillation import compress,blocks,build_records
from tableqa.data import example,nested_samples,protocol_split,audit
from tableqa.inference import predict
from tableqa.models import ReplayModel
from tableqa.experiments import settings,manifest
from tableqa.evaluation import evaluate,paired_bootstrap


def sample(split="train"):
    return example("wikitq",split,"a","t",["Team","Score","Year"],
                   [["A","10",2022],["B","20",2023],["C","20",2024]],"Who has the highest score?",["B","C"])


class ExecutorTests(unittest.TestCase):
    def setUp(self): self.state=TableState.from_table(sample()["table"])
    def test_ties_preserved(self):
        s,trace=run_plan(self.state,[{"op":"select","column":"c1","which":"max"},
                                    {"op":"answer","columns":["c0"]}])
        self.assertEqual(s.answer,["B","C"])
        self.assertIn("r0:c1",s.evidence)
        self.assertIn("r1:c0",s.evidence)
        self.assertEqual(len(self.state.rows),3)
    def test_dependency_prevents_early_delete(self):
        with self.assertRaisesRegex(ExecutionError,"future dependency"):
            validate_plan(self.state,[{"op":"project","columns":["c0"]},
                                     {"op":"filter","column":"c1","cmp":"gt","value":5}])
    def test_derived_dependency_allowed(self):
        ops=[{"op":"project","columns":["c1"]},
             {"op":"derive","as":"double","expression":{"op":"mul","args":[{"col":"c1"},2]}},
             {"op":"answer","columns":["double"]}]
        s,_=run_plan(self.state,ops); self.assertEqual(s.answer,[20,40,40])
    def test_sort_lineage_stable(self):
        s=execute(self.state,{"op":"sort","column":"c1","type":"number","descending":True})
        self.assertEqual(s.row_ids,["r1","r2","r0"])
        self.assertEqual(s.lineage["r1"]["c0"],{"r1:c0"})
    def test_state_header_metadata_only_contains_active_columns(self):
        s=execute(self.state,{"op":"project","columns":["c0"]})
        self.assertEqual(s.view(["Team","Score","Year"])["original_headers"],{"c0":"Team"})
    def test_group_sum_and_sources(self):
        s=execute(self.state,{"op":"group_aggregate","by":[],"aggregations":[{"fn":"sum","column":"c1","as":"sum"}]})
        self.assertEqual(s.rows,[{"sum":50}]); self.assertEqual(len(s.lineage["g0"]["sum"]),3)
    def test_empty_count(self):
        s=execute(self.state,{"op":"filter","column":"c1","cmp":"gt","value":100,"type":"number"})
        s=execute(s,{"op":"group_aggregate","by":[],"aggregations":[{"fn":"count","as":"n"}]})
        self.assertEqual(s.rows,[{"n":0}]); self.assertTrue(s.evidence)
    def test_number_strict(self):
        self.assertEqual(float(number("12.5%")),.125)
        self.assertEqual(number("1,200"),1200)
        for v in ["unknown",None,"NaN",True]:
            with self.assertRaises(ExecutionError): number(v)
    def test_atomic_failure(self):
        before=self.state.signature()
        with self.assertRaises(ExecutionError):
            execute(self.state,{"op":"derive","as":"bad","expression":{"op":"div","args":[1,0]}})
        self.assertEqual(self.state.signature(),before)
    def test_no_arbitrary_code(self):
        with self.assertRaises(ExecutionError): execute(self.state,{"op":"eval","code":"print(1)"})
    def test_duplicate_headers(self):
        s=TableState.from_table({"headers":["X","X"],"rows":[[1,2]]})
        self.assertEqual(s.rows,[{"c0":1,"c1":2}])
    def test_cannot_continue_after_answer(self):
        s=execute(self.state,{"op":"answer","columns":["c0"]})
        with self.assertRaises(ExecutionError): execute(s,{"op":"project","columns":["c0"]})
    def test_whitelist_expression(self):
        with self.assertRaises(ExecutionError):
            execute(self.state,{"op":"derive","as":"x","expression":{"op":"exec","args":[1,2]}})


class DistillationTests(unittest.TestCase):
    def test_compression_replay(self):
        s=TableState.from_table(sample()["table"])
        ops=[{"op":"project","columns":["c0","c1","c2"]},{"op":"project","columns":["c0","c1"]},
             {"op":"answer","columns":["c0"]}]
        result=compress(s,ops); self.assertLess(len(result),len(ops))
        self.assertEqual(run_plan(s,ops)[0].signature(),run_plan(s,result)[0].signature())
    def test_feedback_boundary(self):
        ops=[{"op":"sort"},{"op":"filter","requires_feedback":True},{"op":"answer"}]
        self.assertEqual([len(b) for b in blocks(ops,4)],[1,2])
        self.assertEqual([len(b) for b in blocks(ops,1)],[1,1,1])
    def test_no_test_training(self):
        with self.assertRaises(ValueError): build_records(sample("test"),[],"B0")
    def test_b1_needs_actual_rationale(self):
        with self.assertRaises(ValueError): build_records(sample(),[],"B1")
    def test_records_have_no_reference_in_prompt(self):
        s=sample(); s["reference"]=["SECRET_SENTINEL"]
        records=build_records(s,[{"op":"answer","columns":["c0"]}],"B2")
        self.assertNotIn("SECRET_SENTINEL",str(records[0]["messages"]))


class InferenceTests(unittest.TestCase):
    def test_reference_isolation(self):
        class Spy(ReplayModel):
            def generate(self,messages):
                assert "SECRET_SENTINEL" not in str(messages)
                return super().generate(messages)
        s=sample(); s["reference"]="SECRET_SENTINEL"
        result=predict(s,Spy([{"answer":["B","C"]}]),settings("B0"))
        self.assertEqual(result["status"],"ok")
        self.assertIsNone(result["metrics"]["input_tokens"])
    def test_repair_does_not_commit_partial_block(self):
        bad=[{"op":"filter","column":"c1","cmp":"gt","value":10,"type":"number"},
             {"op":"derive","as":"x","expression":{"op":"div","args":[1,0]}}]
        good=[{"op":"answer","columns":["c0"]}]
        p=predict(sample(),ReplayModel([{"operations":bad},{"operations":good}]),settings("B5"))
        self.assertEqual(p["answer"],["A","B","C"])
        self.assertEqual(p["metrics"]["repairs"],1)
    def test_failure_retained(self):
        p=predict(sample(),ReplayModel(["bad"]),settings("B5",max_repairs=0))
        self.assertEqual(p["status"],"failed"); self.assertTrue(p["failure"])
    def test_answer_in_middle_rejected(self):
        p=predict(sample(),ReplayModel([{"operations":[{"op":"answer","columns":["c0"]},
                   {"op":"project","columns":["c0"]}]}]),settings("B5",max_repairs=0))
        self.assertEqual(p["status"],"failed")


class ProtocolTests(unittest.TestCase):
    def test_nested_fewshot(self):
        records=[{**sample(),"id":str(i),"table_id":str(i//2)} for i in range(10)]
        sets=nested_samples(records,[2,4,8])
        self.assertEqual(sets["2"],sets["4"][:2]); self.assertEqual(sets["4"],sets["8"][:4])
        self.assertEqual(len({r["table_id"] for r in sets["4"]}),4)
    def test_cross_task_no_target_validation(self):
        records=[{**sample(),"split":"train"}, {**sample(),"dataset":"tabfact","split":"validation"},
                 {**sample(),"dataset":"tabfact","split":"test"}]
        parts=protocol_split(records,target="tabfact")
        self.assertFalse(parts["validation"])
        self.assertFalse(parts["train"]) # table-content duplicate removed
    def test_topic_requires_reviews(self):
        with self.assertRaises(ValueError): protocol_split([sample()],topic="sports",annotations={})
    def test_audit(self):
        report=audit([sample(),sample("test")]); self.assertTrue(report["table_overlaps"])
    def test_evaluation_requires_failed_predictions_too(self):
        with self.assertRaises(ValueError): evaluate([], [sample()])
    def test_bootstrap(self):
        result=paired_bootstrap([0,0],[1,1],100)
        self.assertEqual(result["ci95"],[1,1])
    def test_manifest_covers_seven_families(self):
        runs=manifest(); self.assertEqual(len({r["run_id"] for r in runs}),len(runs))
        self.assertTrue({"teacher_main","student_main","teacher_ablation","task_transfer","topic_transfer","parameter","robustness"} <= {r["family"] for r in runs})


if __name__=="__main__": unittest.main()
