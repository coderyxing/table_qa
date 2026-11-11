"""Lossless input representation candidates, independent of running experiments."""
from copy import deepcopy
import json


def array_view(view):
    result=deepcopy(view)
    columns=result['columns']
    result['rows']=[[row[c] for c in columns] for row in result['rows']]
    result['row_format']='value arrays aligned to columns; row_ids aligned to rows'
    return result


def restore_view(view):
    result=deepcopy(view)
    result.pop('row_format',None)
    columns=result['columns']
    if any(len(row)!=len(columns) for row in result['rows']): raise ValueError('Ragged serialized row')
    if len(result['row_ids'])!=len(result['rows']): raise ValueError('Row ID alignment mismatch')
    result['rows']=[dict(zip(columns,row)) for row in result['rows']]
    return result


class ArrayInputModel:
    """A4 candidate adapter; changes only state serialization, not data/operations."""
    def __init__(self,backend): self.backend=backend
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
