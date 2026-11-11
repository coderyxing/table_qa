"""Synthetic diagnostic only: question obligations survive coincidental equality."""
import json
from tableqa.core import TableState, run_plan
from tableqa.optimization import optimize_operations
from tableqa.contracts import ObligationTracker, ContractError, verify_condition_rewrite


def main():
    question='Which teams scored above 5?';quote='above 5';start=question.index(quote)
    predicate={'op':'filter','column':'c1','cmp':'gt','value':5,'type':'number'}
    source=[{'op':'sort','column':'c1','type':'number'},predicate,{'op':'answer','columns':['c0']}]
    state=TableState.from_table({'headers':['Team','Score'],'rows':[['A',10],['B',20]]})
    reduced,_=optimize_operations(state,source)
    tracker=ObligationTracker.from_question(question,[{'id':'threshold','quote':quote,
        'question_span':[start,start+len(quote)],'operation':predicate}])
    rejected=False
    try:verify_condition_rewrite(state,source,reduced,tracker)
    except ContractError:rejected=True
    changed=TableState.from_table({'headers':['Team','Score'],'rows':[['A',0],['B',20]]})
    output={'synthetic_fixture_only':True,'not_a_benchmark_or_model_result':True,
            'source_operations':len(source),'reduced_operations':len(reduced),
            'same_signature_on_initial_table':run_plan(state,source)[0].signature()==run_plan(state,reduced)[0].signature(),
            'declared_condition_checker_rejected_rewrite':rejected,
            'counterexample_source_answer':run_plan(changed,source)[0].answer,
            'counterexample_reduced_answer':run_plan(changed,reduced)[0].answer,
            'semantic_completeness_proven':False}
    if not rejected:raise ValueError('The omitted condition was not detected')
    print(json.dumps(output))


if __name__=='__main__':main()
