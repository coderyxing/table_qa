import unittest
from tableqa.contracts import ObligationTracker, ContractError, run_contracted_block, verify_condition_rewrite
from tableqa.core import TableState, run_plan, ExecutionError


QUESTION='After 2020, which team has the highest score?'
FILTER={'op':'filter','column':'c0','cmp':'gt','value':2020,'type':'number'}
SELECT={'op':'select','mode':'extreme','column':'c2','which':'max'}


def condition(key,quote,operation,after=()):
    start=QUESTION.index(quote)
    return {'id':key,'quote':quote,'question_span':[start,start+len(quote)],'operation':operation,'after':list(after)}


def tracker():
    return ObligationTracker.from_question(QUESTION,[condition('year','After 2020',FILTER),
        condition('maximum','highest score',SELECT,['year'])])


def state():return TableState.from_table({'headers':['Year','Team','Score'],'rows':[[2019,'A',99],[2021,'B',20],[2022,'C',30]]})


class ContractTests(unittest.TestCase):
    def test_valid_blocks_keep_obligations_across_calls(self):
        current,t,trace=run_contracted_block(state(),[FILTER],tracker())
        self.assertEqual(t.report()['pending_ids'],['maximum'])
        final,t,_=run_contracted_block(current,[SELECT,{'op':'answer','columns':['c1']}],t)
        self.assertEqual(final.answer,['C']);self.assertFalse(t.report()['pending_ids'])
        self.assertEqual(trace[0]['condition_obligations']['satisfied_here'],['year'])
        self.assertEqual(t.report()['semantic_correctness'],'not_proven')

    def test_source_spans_extra_gold_fields_and_cycles_are_rejected(self):
        good=condition('year','After 2020',FILTER)
        for item in [{**good,'quote':'After 2019'},{**good,'question_span':[True,10]},
                     {**good,'reference':'gold'},{**good,'after':['year']},{**good,'after':['missing']}]:
            with self.assertRaises(ContractError):ObligationTracker.from_question(QUESTION,[item])
        with self.assertRaises(ContractError):ObligationTracker.from_question(QUESTION,[good,good])

    def test_future_required_columns_are_protected_beyond_current_block(self):
        with self.assertRaises(ContractError):
            run_contracted_block(state(),[{'op':'project','columns':['c1','c2']}],tracker())
        current,t,_=run_contracted_block(state(),[FILTER],tracker())
        current,t,_=run_contracted_block(current,[{'op':'project','columns':['c1','c2']}],t)
        self.assertEqual(current.columns,['c1','c2'])

    def test_presence_without_declared_order_is_insufficient(self):
        with self.assertRaises(ContractError):run_contracted_block(state(),[SELECT,FILTER],tracker())
        with self.assertRaises(ContractError):run_contracted_block(state(),[{'op':'answer','columns':['c1']}],tracker())

    def test_failed_execution_does_not_commit_a_satisfied_obligation(self):
        initial=state();t=tracker();before=initial.signature()
        with self.assertRaises(ExecutionError):
            run_contracted_block(initial,[FILTER,{'op':'derive','as':'bad','expression':{'op':'div','args':[1,0]}}],t)
        self.assertEqual(initial.signature(),before);self.assertEqual(t.fulfilled,frozenset())

    def test_shadowed_aggregate_column_does_not_satisfy_future_original_reference(self):
        aggregate={'op':'group_aggregate','by':[],'aggregations':[{'fn':'sum','column':'c2','as':'c0'}]}
        with self.assertRaises(ContractError):run_contracted_block(state(),[aggregate],tracker())

    def test_equal_current_output_cannot_excuse_dropping_a_condition(self):
        initial=TableState.from_table({'headers':['Score'],'rows':[[10],[20]]})
        predicate={'op':'filter','column':'c0','cmp':'gt','value':5,'type':'number'}
        source=[{'op':'sort','column':'c0','type':'number'},predicate,{'op':'answer','columns':['c0']}]
        shorter=[source[0],source[-1]]
        self.assertEqual(run_plan(initial,source)[0].signature(),run_plan(initial,shorter)[0].signature())
        t=ObligationTracker.from_question('Scores above 5',[{'id':'threshold','quote':'above 5',
            'question_span':[7,14],'operation':predicate}])
        with self.assertRaises(ContractError):verify_condition_rewrite(initial,source,shorter,t)
        self.assertTrue(verify_condition_rewrite(initial,source,source,t)['declared_conditions_preserved'])

    def test_incomplete_natural_language_extraction_is_never_declared_proven(self):
        t=ObligationTracker.from_question(QUESTION,[condition('year','After 2020',FILTER)])
        _,finished,_=run_contracted_block(state(),[FILTER,{'op':'answer','columns':['c1']}],t)
        self.assertEqual(finished.report()['declared_condition_execution_coverage'],1)
        self.assertEqual(finished.report()['question_condition_completeness'],'not_proven')

    def test_literal_final_answers_are_not_condition_obligations(self):
        with self.assertRaises(ContractError):
            ObligationTracker.from_question(QUESTION,[condition('answer','team',{'op':'answer','value':'gold'})])

    def test_rewrite_cannot_remove_explicit_feedback_or_forge_completed_ids(self):
        initial=state();t=tracker();source=[{**FILTER,'requires_feedback':True},SELECT,{'op':'answer','columns':['c1']}]
        candidate=[FILTER,SELECT,source[-1]]
        with self.assertRaises(ContractError):verify_condition_rewrite(initial,source,candidate,t)
        with self.assertRaises(ContractError):ObligationTracker(t.conditions,frozenset({'unknown'}))


if __name__=='__main__':unittest.main()
