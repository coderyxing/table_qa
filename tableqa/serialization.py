# 模块职责：在保留表格信息的前提下转换输入表示，重建状态并适配模型调用。
# 接口约定：样本、操作和结果采用本项目的结构化字典；细节见各入口及字段检查。
"""Lossless input representation candidates, independent of running experiments."""
from copy import deepcopy
import json


# 把当前表状态改为列模式与二维数组表示，压缩重复字段。
# 输入：view。
# 返回形式含：result。
def array_view(view):
    result=deepcopy(view)
    columns=result['columns']
    result['rows']=[[row[c] for c in columns] for row in result['rows']]
    result['row_format']='value arrays aligned to columns; row_ids aligned to rows'
    return result


# 从数组表示重建表状态视图，检查可恢复性。
# 输入：view。
# 返回形式含：result。
# 异常约束：Ragged serialized row
def restore_view(view):
    result=deepcopy(view)
    result.pop('row_format',None)
    columns=result['columns']
    if any(len(row)!=len(columns) for row in result['rows']): raise ValueError('Ragged serialized row')
    if len(result['row_ids'])!=len(result['rows']): raise ValueError('Row ID alignment mismatch')
    result['rows']=[dict(zip(columns,row)) for row in result['rows']]
    return result


# ArrayInputModel：在模型调用边界替换状态表示，并保留原模型生成接口。
class ArrayInputModel:
    """A4 candidate adapter; changes only state serialization, not data/operations."""
    # 初始化 ArrayInputModel 的依赖及持久状态；具体预算和身份字段见赋值。
    # 输入：backend。
    def __init__(self,backend): self.backend=backend
    # 在调用后端前替换状态序列化表示，保持题目和操作协议一致。
    # 输入：messages（模型可见消息）。
    # 相关调用：self.backend.generate。
    # 异常约束：Candidate changed table information
    def generate(self,messages):
        converted=deepcopy(messages)
        for message in converted:
            if message['role']!='user': continue
            try: payload=json.loads(message['content'])
            except (ValueError,TypeError): continue
            if not isinstance(payload,dict) or not isinstance(payload.get('state'),dict): continue
            view=payload['state']
            if 'columns' not in view or 'rows' not in view: continue
            if view.get('row_format'): continue
            alternative=array_view(view)
            if restore_view(alternative)!=view: raise ValueError('Candidate changed table information')
            payload['state']=alternative
            message['content']=json.dumps(payload,ensure_ascii=False)
        return self.backend.generate(converted)
