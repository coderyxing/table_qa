# 脚本用途：用合成反例检查条件义务能否拦截依赖偶然答案一致性的删步。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""Synthetic diagnostic only: question obligations survive coincidental equality."""
import json
from tableqa.core import TableState, run_plan
from tableqa.optimization import optimize_operations
from tableqa.contracts import ObligationTracker, ContractError, verify_condition_rewrite


# 解析当前脚本参数，按模块说明执行本次冻结阶段并保存产物。
# 相关调用：run_plan(state, source)[0].signature、run_plan(state, reduced)[0].signature、run_plan。
# 异常约束：The omitted condition was not detected
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
    # 在明确给出的表状态上重放操作链，核对执行结果和来源追踪。
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
