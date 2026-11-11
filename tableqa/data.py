# 模块职责：适配三项表格任务、保留原始值和稳定坐标，并通过表指纹检查跨划分重叠。
# 接口约定：样本、操作和结果采用本项目的结构化字典；细节见各入口及字段检查。
"""Explicit adapters, reproducible table-level splits, and leakage auditing."""
from __future__ import annotations
import csv
import hashlib
import json
import random
from pathlib import Path

TASKS = {"wikitq": "short", "tabfact": "verification", "fetaqa": "freeform"}


# 恢复 WTQ TSV 的换行、竖线和反斜杠转义，避免直接替换导致二次解码。
# 输入：text（待解析文本）。
def wtq_unescape(text):
    result=[]; i=0; mapping={'n':'\n','p':'|','\\':'\\'}
    while i<len(text):
        if text[i]=='\\' and i+1<len(text) and text[i+1] in mapping:
            result.append(mapping[text[i+1]]); i+=2
        else: result.append(text[i]); i+=1
    return ''.join(result)


# 按 UTF-8 读取非空 JSONL 行，返回结构化记录列表。
# 输入：path（输入或输出路径）。
# 返回形式含：列表记录。
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# 创建输出目录并逐条序列化记录，保持中文及结构化字段。
# 输入：path（输入或输出路径）；records（规范记录序列）。
def write_jsonl(path, records):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r, ensure_ascii=False) + "\n")


# 对键排序后的 JSON 求 SHA-256，作为数据和配置的稳定结构摘要。
# 输入：value。
# 相关调用：hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest。
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# 把表头和值配对后规范排序，使行列重排及重复表头仍可进行整表查重。
# 输入：table（原始表头与行矩阵）。
# 相关调用：digest。
def table_fingerprint(table):
    # Header/value pairs survive both row and column permutations, including
    # repeated headers. Raw string/number distinctions deliberately remain.
    rows=[sorted(zip(table["headers"],row),key=lambda x:json.dumps(x,ensure_ascii=False))
          for row in table["rows"]]
    return digest([sorted(table["headers"]),sorted(rows,key=lambda x:json.dumps(x,ensure_ascii=False))])


# 构建规范样本并验证原表矩阵，明确任务、划分和原表身份。
# 输入：dataset；split（数据划分）；uid；table_id；headers（原始显示表头）；rows；question（原题文本）；reference；**extra。
# 返回形式含：result。
# 异常约束：Explicit canonical dataset and split required
def example(dataset, split, uid, table_id, headers, rows, question, reference, **extra):
    if dataset not in TASKS or split not in {"train", "validation", "test"}:
        raise ValueError("Explicit canonical dataset and split required")
    result = {"id": str(uid), "dataset": dataset, "split": split,
              "table_id": str(table_id), "task": TASKS[dataset],
              "table": {"headers": headers, "rows": rows},
              "question": question, "reference": reference, **extra}
    from .core import TableState
    TableState.from_table(result["table"])
    return result


# 按各作者数据格式恢复题目、原表和参考答案，保留任务特定字段。
# 输入：dataset；source（原始来源或原链）；split（数据划分）；table_root。
# 相关调用：source.read_text、read_jsonl、digest。
# 返回形式含：output。
# 异常约束：WTQ requires --table-root
def adapt(dataset, source, split, table_root=None):
    """Read official WTQ TSV, TabFact split JSON, or FeTaQA split JSON/JSONL.

    WTQ source columns: id, utterance, context, targetValue. Table CSV is
    resolved under table_root; raw cell strings are preserved.
    TabFact source: {table_filename: [statements, labels, caption]}.
    FeTaQA source: list or {data: list}, table_array includes header row.
    """
    source = Path(source)
    if dataset == "wikitq":
        if table_root is None: raise ValueError("WTQ requires --table-root")
        with source.open(encoding="utf-8", newline="") as f:
            records = list(csv.DictReader(f, delimiter="\t"))
        output = []
        for r in records:
            root = Path(table_root).resolve()
            path = (root / r["context"]).resolve()
            if not path.is_relative_to(root): raise ValueError("Table path escapes root")
            with path.open(encoding="utf-8", newline="") as f:
                cells = list(csv.reader(f,escapechar="\\"))
            output.append(example(dataset, split, r["id"], r["context"], cells[0], cells[1:],
                                  wtq_unescape(r["utterance"]), [wtq_unescape(v) for v in r["targetValue"].split("|")] ))
        return output
    if dataset == "tabfact":
        if table_root is None: raise ValueError("TabFact requires --table-root")
        records = json.loads(source.read_text(encoding="utf-8")); output = []
        for tid, (statements, labels, caption) in records.items():
            if len(statements) != len(labels): raise ValueError("Statement/label mismatch")
            root = Path(table_root).resolve(); path = (root / tid).resolve()
            if not path.is_relative_to(root): raise ValueError("Table path escapes root")
            with path.open(encoding="utf-8", newline="") as f:
                cells = list(csv.reader(f, delimiter="#"))
            for i, (q, y) in enumerate(zip(statements, labels)):
                output.append(example(dataset, split, f"{tid}:{i}", tid, cells[0], cells[1:], q,
                                      int(y), title=caption))
        return output
    if dataset == "fetaqa":
        # 读取结构化输入：source，保留原有样本身份字段。
        records = read_jsonl(source) if source.suffix == ".jsonl" else json.loads(source.read_text(encoding="utf-8"))
        if isinstance(records, dict): records = records["data"]
        output = []
        for r in records:
            cells = r["table_array"]
            output.append(example(dataset, split, r["feta_id"], digest(cells), cells[0], cells[1:],
                                  r["question"], r["answer"],
                                  highlighted_cells=r.get("highlighted_cell_ids", []),
                                  title=r.get("table_page_title", "")))
        return output
    raise ValueError(dataset)


# 检查样本重复、划分分布与跨划分整表重叠。
# 输入：records（规范记录序列）。
# 返回形式含：字段 count、duplicate_ids、table_overlaps、unseen_test_ids。
def audit(records):
    ids = set(); duplicates = []; fingerprints = {}
    for r in records:
        key = (r["dataset"], r["split"], r["id"])
        if key in ids: duplicates.append(key)
        ids.add(key)
        # Ignore row order for overlap auditing, retain column semantics.
        table = r["table"]
        fp = table_fingerprint(table)
        fingerprints.setdefault(fp, set()).add((r["dataset"], r["split"]))
    overlaps = {fp: sorted(locations) for fp, locations in fingerprints.items() if len(locations) > 1}
    non_test = {table_fingerprint(r["table"]) for r in records if r["split"] != "test"}
    return {"count": len(records), "duplicate_ids": duplicates, "table_overlaps": overlaps,
            "unseen_test_ids": [{"dataset": r["dataset"], "id": r["id"]} for r in records
                                if r["split"] == "test" and table_fingerprint(r["table"]) not in non_test]}


# 按表优先抽样并构建精确嵌套的少样本子集。
# 输入：records（规范记录序列）；sizes；seed（随机种子）。
# 异常约束：Few-shot samples must come only from training split
def nested_samples(records, sizes, seed=42):
    """One example per shuffled table first, then remaining examples; exact nested N."""
    if any(r["split"] != "train" for r in records):
        raise ValueError("Few-shot samples must come only from training split")
    rng = random.Random(seed); groups = {}
    for r in records: groups.setdefault((r["dataset"], r["table_id"]), []).append(r)
    keys = sorted(groups); rng.shuffle(keys)
    first, rest = [], []
    for k in keys:
        group = sorted(groups[k], key=lambda r: r["id"]); rng.shuffle(group)
        first.append(group[0]); rest.extend(group[1:])
    rng.shuffle(rest); ordered = first + rest
    if max(sizes, default=0) > len(ordered): raise ValueError("Not enough training examples")
    return {str(n): ordered[:n] for n in sizes}


# 按目标任务或经审核主题划分来源，维护零目标监督与整表隔离。
# 输入：records（规范记录序列）；target；topic；annotations。
# 返回形式含：字段 train、validation、test、target_train。
# 异常约束：Choose target dataset OR held-out topic
def protocol_split(records, target=None, topic=None, annotations=None):
    if bool(target) == bool(topic): raise ValueError("Choose target dataset OR held-out topic")
    if topic:
        annotations = annotations or {}
        for r in records:
            if r["dataset"] != "wikitq": raise ValueError("Topic protocol uses only WikiTQ")
            a = annotations.get(r["table_id"])
            if not a or a.get("reviewer1") != a.get("reviewer2") or a.get("topic") not in {"sports", "geography", "people", "other"}:
                raise ValueError(f"Unreviewed topic: {r['table_id']}")
            if a["topic"] != a["reviewer1"]: raise ValueError("Topic/review mismatch")
    # 根据目标任务或主题确定保留集合，供来源划分排除。
    # 输入：r。
    def held(r):
        return r["dataset"] == target if target else annotations[r["table_id"]]["topic"] == topic
    train = [r for r in records if r["split"] == "train" and not held(r)]
    validation = [r for r in records if r["split"] == "validation" and not held(r)]
    test = [r for r in records if r["split"] == "test" and held(r)]
    # Content duplicates across datasets cannot cross source-training / target-test boundary.
    target_tables = {table_fingerprint(r["table"]) for r in test}
    train = [r for r in train if table_fingerprint(r["table"]) not in target_tables]
    validation = [r for r in validation if table_fingerprint(r["table"]) not in target_tables]
    return {"train": train, "validation": validation, "test": test,
            "target_train": [r for r in records if r["split"] == "train" and held(r)
                             and table_fingerprint(r["table"]) not in target_tables]}
