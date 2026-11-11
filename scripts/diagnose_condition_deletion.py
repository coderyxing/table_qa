# 脚本用途：构造合成改表反例，观察删除题意条件后的程序行为。
# 使用约定：先核对命令参数、输入划分和工作目录，再执行当前阶段；产物路径在正文中显式给出。
"""A synthetic counterexample, never a benchmark or training augmentation."""
import json
from pathlib import Path
from tableqa.core import TableState,run_plan
from tableqa.optimization import optimize_operations

table={'headers':['Name','Score'],'rows':[['A',10],['B',20]]}
changed={'headers':['Name','Score'],'rows':[['A',0],['B',20]]}
ops=[{'op':'sort','column':'c1','type':'number','descending':True},
     {'op':'filter','column':'c1','type':'number','cmp':'gt','value':5},
     {'op':'answer','columns':['c0']}]
state=TableState.from_table(table)
loose,proof=optimize_operations(state,ops)
guarded,_=optimize_operations(state,ops,preserve_kinds=('filter','sort','select','group_aggregate'))
# 在明确给出的表状态上重放操作链，核对执行结果和来源追踪。
original_results={name:run_plan(state,chain)[0] for name,chain in [('full',ops),('search',loose),('guarded',guarded)]}
modified_results={name:run_plan(TableState.from_table(changed),chain)[0] for name,chain in [('full',ops),('search',loose),('guarded',guarded)]}
assert original_results['full'].signature()==original_results['search'].signature()
assert modified_results['full'].answer!=modified_results['search'].answer
assert modified_results['full'].signature()==modified_results['guarded'].signature()
out=Path('runs/diagnostics/condition-deletion');out.mkdir(parents=True,exist_ok=True)
report={'synthetic_only':True,'diagnostic_only':True,'used_for_training':False,
    'question':'Which entries have scores greater than 5, listed from highest score to lowest?',
    'original_table':table,'modified_table':changed,
    'plans':{'full':ops,'search':loose,'guarded':guarded},
    'original_answers':{k:s.answer for k,s in original_results.items()},
    'modified_answers':{k:s.answer for k,s in modified_results.items()},
    'original_table_search_signature_equal':True,'modified_table_search_answer_equal':False,
    'limits':'Illustrates fixed-table equivalence versus condition preservation; no real-task accuracy or general guarded-safety proof.'}
(out/'counterexample.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
(out/'case.md').write_text('''# 合成诊断：原表等价不等于保留题目条件

问题：哪些条目的分数大于5，并按分数从高到低列出？

|输入|完整链：排序→筛选>5→回答|删除筛选的链|关键条件保留的链|
|---|---|---|---|
|A=10，B=20|B、A|B、A|B、A|
|仅将A改为0，B仍为20|B|B、A|B|

原表上，删除筛选的答案、表格终态、证据和来源签名全部相同；修改一个值后，两条程序行为不同。因此原表重放相等不能证明任意输入上的逻辑等价。

这是预定义合成反例，不是语义不变的问答鲁棒性测试、不属于训练数据，也不计入任何真实任务成绩。它解释关键操作保护的动机，不能证明保护后所有教师链均语义正确。
''',encoding='utf-8')
print(json.dumps(report,indent=2))
