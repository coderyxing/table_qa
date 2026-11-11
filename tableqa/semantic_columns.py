# 模块职责：用精确字符跨度提取派生列，保留当前单元格与原表来源之间的审核关系。
# 接口约定：样本、操作和结果采用本项目的结构化字典；细节见各入口及字段检查。
"""Bounded substring extraction with auditable character spans, no generated code.

Copied text has a mechanical source check. Selecting the right field or meaning
is still a model decision and is NOT certified by substring membership.
"""
import re


# ExtractionError：字符抽取参数或源文本结构不满足规则时抛出的错误。
class ExtractionError(ValueError):
    pass


# 调整提取片段两端空白，同时维护原字符串上的字符位置。
# 输入：text（待解析文本）；start；end。
# 返回形式含：2 项组成的结果。
# 异常约束：Extraction produced an empty field
def _trim_span(text, start, end):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start == end:
        raise ExtractionError('Extraction produced an empty field')
    return start, end


# 按白名单规则定位完整字符片段，拒绝不支持或含糊结构。
# 输入：value；rule；index；delimiter。
# 返回形式含：字段 start、end、value、coordinate_system、copy_verified、semantic_correctness。
# 异常约束：Extraction requires a bounded string cell
def extract_span(value, rule, index=0, delimiter=None):
    """Return an exact Python-codepoint slice of the current source cell.

    parenthesized accepts balanced, non-nested ASCII parentheses only. split
    requires an explicit literal delimiter. ordinal accepts a complete rank cell
    (e.g. T3, 14th (q)); it does not extract an arbitrary number from mixed text.
    Any missing or ambiguous structure fails; no row is silently removed.
    """
    if not isinstance(value, str) or len(value) > 8192:
        raise ExtractionError('Extraction requires a bounded string cell')
    if type(index) is not int or abs(index) > 128:
        raise ExtractionError('Extraction index must be a bounded integer')
    if rule == 'parenthesized':
        if delimiter is not None:
            raise ExtractionError('parenthesized does not accept a delimiter')
        start = None; spans = []
        for i, char in enumerate(value):
            if char == '(':
                if start is not None:
                    raise ExtractionError('Nested parentheses require explicit disambiguation')
                start = i + 1
            elif char == ')':
                if start is None:
                    raise ExtractionError('Unbalanced parentheses')
                spans.append((start, i)); start = None
        if start is not None:
            raise ExtractionError('Unbalanced parentheses')
    elif rule == 'split':
        if not isinstance(delimiter, str) or not delimiter or len(delimiter) > 32:
            raise ExtractionError('split needs a nonempty bounded literal delimiter')
        if delimiter not in value:
            raise ExtractionError('Required delimiter is absent')
        spans = []; cursor = 0
        for part in value.split(delimiter):
            spans.append((cursor, cursor + len(part)))
            cursor += len(part) + len(delimiter)
    elif rule == 'ordinal':
        if delimiter is not None or index != 0:
            raise ExtractionError('ordinal accepts index 0 and no delimiter')
        match = re.fullmatch(r'\s*(?:[Tt=]\s*)?(?P<n>[0-9]+)(?P<suffix>st|nd|rd|th)?(?:\s*\([^()]+\))?\s*', value)
        if match is None or len(match['n']) > 9 or int(match['n']) < 1:
            raise ExtractionError('Unsupported complete rank cell')
        n = int(match['n'])
        expected = 'th' if 10 < n % 100 < 14 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        if match['suffix'] is not None and match['suffix'] != expected:
            raise ExtractionError('Invalid ordinal suffix')
        spans = [match.span('n')]
    else:
        raise ExtractionError(f'Unsupported extraction rule: {rule!r}')
    try:
        start, end = _trim_span(value, *spans[index])
    except IndexError as exc:
        raise ExtractionError('Requested extraction field is absent') from exc
    return {'start': start, 'end': end, 'value': value[start:end],
            'coordinate_system': 'Python Unicode codepoints; end exclusive',
            'copy_verified': True, 'semantic_correctness': 'not_proven'}


# 遍历表达式中的抽取节点，供能力开关和证据审核。
# 输入：value。
def extraction_nodes(value):
    if not isinstance(value, dict):
        return
    if value.get('op') == 'extract':
        yield value
    for arg in value.get('args', []):
        yield from extraction_nodes(arg)


# 校验抽取表达式需要的参数与源列引用。
# 输入：expr。
# 异常约束：extract requires exactly one direct source-column argument
def extraction_argument(expr):
    args = expr.get('args', [])
    if not isinstance(args, list) or len(args) != 1 or not isinstance(args[0], dict) or set(args[0]) != {'col'}:
        raise ExtractionError('extract requires exactly one direct source-column argument')
    return args[0]['col']


# 对当前行的源单元格执行抽取，并返回可追溯结果。
# 输入：expr；row（当前行记录）。
def extract_from_row(expr, row):
    column = extraction_argument(expr)
    return extract_span(row[column], expr.get('rule'), expr.get('index', 0), expr.get('delimiter'))


# 为派生抽取保存当前单元格跨度与根单元格来源。
# 输入：state（当前执行状态）；op。
# 返回形式含：records。
def extraction_audit(state, op):
    """Character positions refer to the current cell, with root cell lineage.

    A source may itself be derived. Root lineage alone does not imply that these
    offsets apply directly to every original cell; the scope is recorded below.
    """
    # 遍历实际被选择的条件分支，收集本次执行发生的抽取节点。
    # 输入：expr；row（当前行记录）。
    def active_nodes(expr, row):
        if not isinstance(expr, dict):
            return
        if expr.get('op') == 'if':
            from .core import evaluate_expression
            condition, yes, no = expr['args']
            yield from active_nodes(condition, row)
            yield from active_nodes(yes if evaluate_expression(condition, row) else no, row)
        elif expr.get('op') == 'extract':
            yield expr
        else:
            for arg in expr.get('args', []):
                yield from active_nodes(arg, row)
    records = []
    for rid, row in zip(state.row_ids, state.rows):
        for expr in active_nodes(op.get('expression'), row):
            column = extraction_argument(expr)
            records.append({'row_id': rid, 'source_column': column, 'derived_column': op['as'],
                            'span_scope': 'current_source_cell',
                            'source_cells': sorted(state.lineage[rid][column]),
                            **extract_from_row(expr, row)})
    return records


# 判断操作链是否依赖抽取工具，以确保提示和执行配置一致。
# 输入：operations（白名单操作序列）。
def uses_extraction(operations):
    if isinstance(operations, dict):
        return operations.get('op') == 'extract' or any(uses_extraction(v) for v in operations.values())
    return isinstance(operations, list) and any(uses_extraction(v) for v in operations)


EXTRACTION_PROMPT = '''
Optional source-copy extraction (extract-v1): derive.expression accepts
{"op":"extract","args":[{"col":"c0"}],"rule":"parenthesized","index":0}.
It copies a nonempty field from balanced NON-NESTED ASCII parentheses.
rule:split requires a literal delimiter and index, e.g. delimiter:",", index:-1.
rule:ordinal copies rank digits from complete rank cells such as T3 or 14th (q).
index uses Python ordering; default 0. No regex, arbitrary code, entity expansion,
date guessing, external knowledge, invented text, or silently dropping failed rows.
The argument must be one direct current column. Retain original columns.
A copied substring is source-checked, but its meaning and relevance are not proved.
'''

EXTRACTION_DEMO = '''
Synthetic tool demonstration, not a benchmark sample:
Columns c0=athlete; rows Rina (RED), Kaito (BLU), Noor (RED).
Question: Which affiliation code has the most athletes?
{"operations":[{"op":"derive","as":"affiliation","expression":{"op":"extract","args":[{"col":"c0"}],"rule":"parenthesized","index":0}},{"op":"group_aggregate","by":["affiliation"],"aggregations":[{"fn":"count","as":"n"}]},{"op":"select","mode":"extreme","column":"n","which":"max"},{"op":"answer","columns":["affiliation"]}]}
Example of rank conversion: derive with rule:ordinal from cells 12th, 3rd, T3
produces numeric strings 12, 3, 3; extreme:min must retain both tied rows.
'''

QUANTITY_DEMO = '''
Synthetic tool demonstration, not a benchmark sample:
Columns c0=name,c1=distance; rows A,500 m and B,2 km. To compare distances,
{"operations":[{"op":"derive","as":"distance_km","expression":{"op":"quantity","args":[{"col":"c1"}],"unit":"km"}},{"op":"select","mode":"extreme","column":"distance_km","which":"max"},{"op":"answer","columns":["c0"]}]}
Converted values are 0.5 and 2. Output original name B, preserving source columns.
'''
