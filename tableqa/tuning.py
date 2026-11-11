# 模块职责：按固定规则选择准确率开发和确认队列，记录历史已接触来源。
# 接口约定：样本、操作和结果采用本项目的结构化字典；细节见各入口及字段检查。
"""Validation-only cohorts, disjoint by whole-table content and selected without labels."""
import random
from .data import table_fingerprint


# 从允许划分中按规则抽取开发与确认队列，排除历史接触来源。
# 输入：records（规范记录序列）；previous；per_task；seed（随机种子）。
# 返回形式含：out。
# 异常约束：Positive cohort size required
def accuracy_cohorts(records,previous,per_task=100,seed=2030):
    if type(per_task) is not int or per_task<1:raise ValueError('Positive cohort size required')
    if any(r['split']!='validation' for r in records):raise ValueError('Accuracy tuning must use validation only')
    excluded={table_fingerprint(r['table']) for r in previous}
    pools={}
    for d in ['wikitq','tabfact']:
        grouped={}
        for r in records:
            if r['dataset']!=d:continue
            fp=table_fingerprint(r['table'])
            if fp not in excluded:grouped.setdefault(fp,[]).append(r)
        keys=sorted(grouped);rng=random.Random(seed);rng.shuffle(keys)
        pools[d]=[]
        for fp in keys:
            candidates=sorted(grouped[fp],key=lambda r:r['id'])
            pools[d].append((fp,candidates[rng.randrange(len(candidates))]))
    used=set(excluded);out={}
    for name in ['tune','confirmation']:
        chosen={}
        for d in ['wikitq','tabfact']:
            selected=[]
            for fp,r in pools[d]:
                if fp in used:continue
                selected.append(r);used.add(fp)
                if len(selected)==per_task:break
            if len(selected)!=per_task:raise ValueError(f'Insufficient disjoint validation tables for {d}/{name}: {len(selected)}')
            chosen[d]=selected
        out[name]=[chosen[d][i] for i in range(per_task) for d in ['wikitq','tabfact']]
    return out
