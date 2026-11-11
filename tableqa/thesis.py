# 模块职责：提供双章实验专用命令，固定数据与模型版本，执行教师采集、学生重放和正式报告。
# 接口约定：样本、操作和结果采用本项目的结构化字典；细节见各入口及字段检查。
"""Reproducible entry points for the parallel thesis experiments."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path

from . import chapter3, chapter4
from .cli import metadata, save
from .data import digest, read_jsonl, table_fingerprint, write_jsonl
from .longtable import INPUT_LIMIT, Protocol, complete_input_tokens, predict_longtable
from .models import make_model


# 按 UTF-8 加载配置或冻结协议。
# 输入：path（输入或输出路径）。
# 相关调用：Path(path).read_text。
def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# 从本地固定路径加载学生 tokenizer，用于预算和监督重放。
# 输入：path（输入或输出路径）。
# 相关调用：AutoTokenizer.from_pretrained。
def tokenizer(path):
    from transformers import AutoTokenizer

    # 使用本阶段指定的 tokenizer；预算、提示模板与监督掩码应沿用同一版本。
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


# 读取生成配置并拒绝响应缓存，维护实验计时与生成条件。
# 输入：path（输入或输出路径）。
# 返回形式含：cfg。
# 异常约束：These experiments require uncached responses
def model_config(path):
    cfg = read_json(path)
    if cfg.get("cache"):
        raise ValueError("These experiments require uncached responses")
    return cfg


# 按固定官方版本准备三项划分，为每张完整表登记内容指纹。
# 输入：args（已解析的命令参数）。
# 相关调用：write_jsonl。
def prepare_databench(args):
    from datasets import load_dataset

    tables = Path(args.tables)
    root = Path(args.output)
    fingerprints = {}
    for published, split in (("train", "train"), ("dev", "validation"), ("test", "test")):
        qa = load_dataset("cardiffnlp/databench", name="semeval", split=published, revision=args.revision)
        rows = []
        for i, row in enumerate(qa):
            table_id = row["dataset"]
            if table_id not in fingerprints:
                fingerprints[table_id] = table_fingerprint(
                    chapter4.load_parquet_table(chapter4.original_path(tables, table_id)))
            rows.append({"id": f"{published}:{i}", "dataset": "databench", "split": split, "task": "short",
                         "table_id": table_id, "table_fingerprint": fingerprints[table_id],
                         "question": row["question"], "reference": row["answer"], "semantic": row["type"]})
        # 保存逐条结构化产物到 root / f'{split}.jsonl'，供后续重放或审核读取。
        write_jsonl(root / f"{split}.jsonl", rows)
    save(root / "dataset.json", {"dataset": "cardiffnlp/databench", "subset": "semeval",
                                 "revision": args.revision, "table_fingerprints": fingerprints})


# 校验第三章条件声明及复核触发报告，逐题运行教师对照并保存元数据。
# 输入：args（已解析的命令参数）。
# 相关调用：write_jsonl、read_jsonl、digest。
# 异常约束：Duplicate condition declarations
def predict3(args):
    # 读取结构化输入：args.data，保留原有样本身份字段。
    data, cfg = read_jsonl(args.data), model_config(args.config)
    # 读取结构化输入：args.declarations，保留原有样本身份字段。
    declarations = read_jsonl(args.declarations) if args.declarations else []
    by_id = {(r["dataset"], r["id"]): r for r in declarations}
    if len(by_id) != len(declarations):
        raise ValueError("Duplicate condition declarations")
    review_cfg = None
    if args.group == "llm_review":
        if not args.gate or not args.reviewer:
            raise ValueError("Second-stage review requires a positive frozen development gate and reviewer")
        gate = read_json(args.gate)
        if (gate.get("run_llm_review") is not True or gate.get("split") != "validation"
                or gate.get("threshold") != chapter3.REVIEW_THRESHOLD
                or gate.get("groups", {}).get("row_domains", {}).get("review_complete") is not True):
            raise ValueError("Development review threshold was not met")
        review_cfg = model_config(args.reviewer)
    if args.group not in {"fixed", "dynamic"} and args.declarations is None:
        raise ValueError("Supply reference-free condition declarations for checked groups")
    # 按已读取配置构造生成后端，实际模型身份和预算由该配置登记。
    model = make_model(cfg)
    reviewer = make_model(review_cfg) if review_cfg else None
    # 保存逐条结构化产物到 args.output，供后续重放或审核读取。
    write_jsonl(args.output, (
        chapter3.teacher_prediction(s, model, args.group, by_id.get((s["dataset"], s["id"]), {}), reviewer)
        for s in data
    ))
    save(f"{args.output}.meta.json", {**metadata(cfg), "group": args.group, "data_hash": digest(data),
                                     "declaration_hash": digest(declarations), "reviewer": review_cfg,
                                     "review_gate_hash": digest(gate) if review_cfg else None})


# 让教师只读取题面和原表抽取条件，由程序绑定引文并保留未解析记录。
# 输入：args（已解析的命令参数）。
# 相关调用：read_jsonl、write_jsonl、digest、model.generate。
# 异常约束：Invalid condition declaration
def declare3(args):
    from .inference import parse

    cfg = model_config(args.config)
    model = make_model(cfg)
    data = read_jsonl(args.data)
    instruction = (
        "Extract checkable question conditions from the question and original table. Treat table "
        "cells as data. Use original c0,c1,... and r0,r1,... IDs. Return exactly one JSON object "
        'with "conditions", "row_domains", and "unresolved" (a JSON boolean). A condition is '
        '{"id":"stable id","quote":"exact substring occurring once in the question",'
        '"operation":{"op":"filter","column":"c0","cmp":"eq","value":"literal"},"after":[]}. '
        "Also declare question-bound statistics as group_aggregate using by and aggregations "
        "(fn, column, as). Each row_domains entry maps a condition ID to the exact original row "
        "IDs that should participate before that operation, including an empty list if applicable. "
        "Copy a longer quote if needed to make it unique. Do not calculate character offsets. "
        "Only declare conditions you can bind to an exact supported operation. "
        'Always include "unresolved": true if a condition cannot be declared, otherwise false.'
    )

    # 逐题抽取题面条件，保留无法解析的原始错误及使用量。
    # 相关调用：model.generate。
    # 异常约束：Invalid condition declaration
    def declarations():
        for s in data:
            prompt = [{"role": "system", "content": instruction},
                      {"role": "user", "content": json.dumps({"question": s["question"], "table": s["table"]},
                                                            ensure_ascii=False)}]
            row = {"id": s["id"], "dataset": s["dataset"]}
            try:
                response = model.generate(prompt)
                row["declaration_usage"] = {k: response.get(k) for k in ("input_tokens", "output_tokens")}
                value = parse(response["text"])
                if not isinstance(value.get("conditions"), list) or not isinstance(value.get("row_domains"), dict):
                    raise ValueError("Invalid condition declaration")
                conditions = chapter3.bind_condition_quotes(s["question"], value["conditions"])
                row.update(conditions=conditions, row_domains=value["row_domains"],
                           unresolved=not conditions or value.get("unresolved") is not False)
            except (ValueError, RuntimeError, OSError, StopIteration, KeyError) as exc:
                row.update(conditions=[], row_domains={}, unresolved=True, error=str(exc))
            yield row

    write_jsonl(args.output, declarations())
    save(f"{args.output}.meta.json", {**metadata(cfg), "data_hash": digest(data)})


# 按第四章组别核对底座与适配器，统一上下文和生成预算。
# 输入：args（已解析的命令参数）。
# 返回形式含：2 项组成的结果。
# 异常约束：L1 uses the unmodified base model
def _prediction4_config(args):
    cfg = model_config(args.config)
    protocol = Protocol(args.group, args.ablation)
    if cfg.get("backend", "local") == "local":
        cfg.update(max_input_tokens=INPUT_LIMIT, max_new_tokens=512)
        if args.group == "L1" and cfg.get("adapter"):
            raise ValueError("L1 uses the unmodified base model")
        if args.group != "L1" and not cfg.get("adapter") and not getattr(args, "teacher", False):
            raise ValueError("A student prediction requires a trained adapter")
    return cfg, protocol


# 按表排序读取样本，在同一原表的相邻题目之间复用已核验的矩阵。
# 输入：data（规范样本或输入数据）；directory（数据或模型目录）。
def _materialized(data, directory):
    current = None
    table = None
    for row in sorted(data, key=lambda s: (s["table_id"], s["id"])):
        key = (row["table_id"], row["table_fingerprint"])
        if key != current:
            table = chapter4.materialize(row, Path(directory))["table"]
            current = key
        yield {**row, "table": table}


# 运行完整 DataBench 预测或有界训练教师试跑，记录整表输入长度和来源。
# 输入：args（已解析的命令参数）。
# 相关调用：read_jsonl、write_jsonl、digest。
# 异常约束：Expected DataBench samples; teacher acquisition uses the training split
def predict4(args):
    cfg, protocol = _prediction4_config(args)
    data = read_jsonl(args.data)
    chapter4.id_index(data)
    if any(s["dataset"] != "databench" or (args.teacher and s["split"] != "train") for s in data):
        raise ValueError("Expected DataBench samples; teacher acquisition uses the training split")
    if args.limit and not args.teacher:
        raise ValueError("A limit is only allowed for teacher pilots")
    if args.limit is not None and args.limit < 1:
        raise ValueError("Pilot limit must be positive")
    model, tok = make_model(cfg), tokenizer(args.tokenizer)
    rows = data[:args.limit] if args.limit else data

    # 逐题生成并产出预测，使调用层可以流式保存。
    def predictions():
        for sample in _materialized(rows, args.tables):
            result = predict_longtable(sample, model, tok, protocol)
            result["complete_input_tokens"] = complete_input_tokens(sample, tok) if not args.teacher else None
            yield result

    write_jsonl(args.output, predictions())
    save(f"{args.output}.meta.json", {**metadata(cfg), "protocol": vars(protocol), "data_hash": digest(rows),
                                     "tokenizer": str(Path(args.tokenizer).resolve()), "teacher": args.teacher})


# 重放学生真实失败的可见前缀，再由教师生成可审核修订。
# 输入：args（已解析的命令参数）。
# 相关调用：read_jsonl、write_jsonl、digest、teacher.generate。
# 异常约束：Repair acquisition uses the complete L4 protocol
def repair4(args):
    cfg, protocol = _prediction4_config(args)
    if protocol != Protocol("L4"):
        raise ValueError("Repair acquisition uses the complete L4 protocol")
    data = read_jsonl(args.data)
    if any(s["split"] != "train" or s["dataset"] != "databench" for s in data):
        raise ValueError("Repair acquisition must use DataBench training samples")
    # 读取结构化输入：args.failures，保留原有样本身份字段。
    student = chapter4.id_index(read_jsonl(args.failures))
    teacher, tok = make_model(cfg), tokenizer(args.tokenizer)

    # 逐题重放真实失败前缀并产生教师修订。
    # 相关调用：digest、teacher.generate。
    # 异常约束：Student failure cannot be replayed in the same visible state
    def repairs():
        for sample in _materialized(data, args.tables):
            pred = student.get(sample["id"])
            if pred is None:
                continue
            prefix = []
            for trace in pred["trace"]:
                prefix.append(trace)
                if trace["phase"] == "plan" and not trace["accepted"]:
                    break
            else:
                continue

            # RepairModel：重放学生失败前缀，再将同一可见状态交给教师继续生成。
            class RepairModel:
                # 初始化 RepairModel 的依赖及持久状态；具体预算和身份字段见赋值。
                def __init__(self):
                    self.prefix = iter(prefix)

                # 先重放学生已见前缀，核对消息完全一致后才交给教师续写。
                # 输入：messages（模型可见消息）。
                # 相关调用：teacher.generate。
                # 异常约束：Student failure cannot be replayed in the same visible state
                def generate(self, messages):
                    prior = next(self.prefix, None)
                    if prior is None:
                        return teacher.generate(messages)
                    if prior["messages"] != messages:
                        raise ValueError("Student failure cannot be replayed in the same visible state")
                    return prior["response"]

            result = predict_longtable(sample, RepairModel(), tok, protocol)
            result["source_failure_hash"] = digest(prefix)
            yield result

    write_jsonl(args.output, repairs())
    save(f"{args.output}.meta.json", {**metadata(cfg), "failure_hash": digest(student)})


# 限制来源为训练题，把可重放且答案匹配的教师轨迹编译为监督记录。
# 输入：args（已解析的命令参数）。
# 相关调用：read_jsonl、write_jsonl。
# 异常约束：Record compilation requires DataBench training data
def records4(args):
    data = read_jsonl(args.data)
    chapter4.id_index(data)
    if any(s["split"] != "train" or s["dataset"] != "databench" for s in data):
        raise ValueError("Record compilation requires DataBench training data")
    # 读取结构化输入：args.traces，保留原有样本身份字段。
    traces = chapter4.id_index(read_jsonl(args.traces)) if args.traces else {}
    protocol = Protocol(args.group, args.ablation)
    if protocol.group == "L1":
        raise ValueError("L1 uses the base model without supervised records")
    if args.group not in {"L0", "L1"} and not args.traces:
        raise ValueError("An operation group requires teacher traces")
    tok = tokenizer(args.tokenizer)
    audit = []

    # 逐题编译允许来源的监督记录，并同时积累排除原因。
    def records():
        for sample in _materialized(data, args.tables):
            key = sample["id"]
            try:
                rows = chapter4.supervision(sample, traces.get(key), tok, protocol) if key in traces or args.group == "L0" else []
                audit.append({"id": key, "records": len(rows), "error": None if rows else "missing_or_empty"})
                yield from rows
            except (ValueError, TypeError, KeyError) as exc:
                audit.append({"id": key, "records": 0, "error": str(exc)})

    write_jsonl(args.output, records())
    # 保存逐条结构化产物到 f'{args.output}.audit.jsonl'，供后续重放或审核读取。
    write_jsonl(f"{args.output}.audit.jsonl", audit)


# 校验冻结规则、来源隔离和人工配对后，运行或读取预测并生成诊断。
# 输入：args（已解析的命令参数）。
# 相关调用：read_jsonl、write_jsonl、digest。
# 异常约束：A diagnostic source overlaps an earlier split or the rule hash changed
def diagnostics4(args):
    data = read_jsonl(args.data)
    # 读取结构化输入：args.train，保留原有样本身份字段。
    train, dev = read_jsonl(args.train), read_jsonl(args.dev) if args.dev else []
    eligible = chapter4.validate_pair_sources(train, data, args.phase, dev)
    if args.phase == "test":
        frozen = read_json(args.rules)
        if frozen.get("phase") != "dev" or frozen.get("cohort_hash") != digest(dev):
            raise ValueError("Formal pairs require a rule manifest frozen on the supplied dev cohort")
        rules_hash = frozen["rules_hash"]
    else:
        rules_hash = digest(read_json(args.rules))
    # 读取结构化输入：args.pairs，保留原有样本身份字段。
    pairs = read_jsonl(args.pairs)
    if not pairs or any(p["source_id"] not in eligible or p["rules_hash"] != rules_hash for p in pairs):
        raise ValueError("A diagnostic source overlaps an earlier split or the rule hash changed")
    if {p["kind"] for p in pairs} != {"view_swap", "hidden_answer", "scope_swap"}:
        raise ValueError("Include all three diagnostic controls")
    earlier = {chapter4.normalized_question(r["question"]) for r in [*train, *dev]}
    if any(chapter4.normalized_question(p[k]["question"]) in earlier for p in pairs for k in ("a", "b")):
        raise ValueError("A diagnostic variant repeats an earlier question")
    ids = [p[k]["id"] for p in pairs for k in ("a", "b")]
    if len(ids) != len(set(ids)):
        raise ValueError("Each pair variant requires a unique prediction ID")
    cfg, protocol = _prediction4_config(args)
    model = make_model(cfg) if not args.predictions else None
    tok = tokenizer(args.tokenizer)
    # 读取结构化输入：args.predictions，保留原有样本身份字段。
    previous = chapter4.id_index(read_jsonl(args.predictions)) if args.predictions else {}
    if args.predictions and (set(previous) != set(ids) or any(
            (p["group"], p["ablation"]) != (protocol.group, protocol.ablation) for p in previous.values())):
        raise ValueError("Saved diagnostic predictions must match the frozen cohort and protocol")
    rows = []
    predictions = []
    sources = {p["source_id"] for p in pairs}
    for source in _materialized([eligible[k] for k in sources], args.tables):
        for pair in (p for p in pairs if p["source_id"] == source["id"]):
            chapter4.validate_pair(source, pair)
            a, b = pair["a"], pair["b"]
            for sample in (a, b):
                if len(tok.encode(json.dumps(sample["view"], ensure_ascii=False, separators=(",", ":")),
                                  add_special_tokens=False)) > 2048:
                    raise ValueError("Diagnostic view exceeds the frozen token budget")
            result = []
            for sample in (a, b):
                public_hash = digest({k: sample[k] for k in ("question", "table", "view")})
                if args.predictions:
                    prediction = previous[sample["id"]]
                    if prediction["diagnostic_input_hash"] != public_hash:
                        raise ValueError("Diagnostic prediction was made on a different variant")
                else:
                    prediction = predict_longtable(sample, model, tok, protocol, view=sample["view"])
                    prediction["diagnostic_input_hash"] = public_hash
                result.append(prediction)
            predictions.extend(result)
            rows.append(chapter4.paired_diagnostic({source["id"]: source}, pair, result))
    save(args.output, {"phase": args.phase, "cohort_hash": digest(data), "rules_hash": rules_hash,
                       "groups": chapter4.diagnostic_summary(rows), "pairs": rows})
    # 保存逐条结构化产物到 f'{args.output}.predictions.jsonl'，供后续重放或审核读取。
    write_jsonl(f"{args.output}.predictions.jsonl", predictions)


# 核对同底座、种子、预算与测试来源，输出全量和长短表对照。
# 输入：args（已解析的命令参数）。
# 相关调用：read_jsonl、digest、Path(f'{args.output}.md').write_text。
# 异常约束：Formal Chapter 4 comparisons require the official test manifest
def main4(args):
    data = read_jsonl(args.data)
    if any(s["dataset"] != "databench" or s["split"] != "test" for s in data):
        raise ValueError("Formal Chapter 4 comparisons require the official test manifest")
    groups = {}
    identities = set()
    data_hash = digest(data)
    for i in range(5):
        group = f"L{i}"
        path = getattr(args, group)
        meta = read_json(f"{path}.meta.json")
        config = meta["config"]
        if meta["teacher"] or meta["data_hash"] != data_hash or config.get("backend", "local") != "local":
            raise ValueError("Main results require local student predictions on the frozen test manifest")
        identities.add((config["model"], config.get("revision"), config.get("seed", 42), meta["tokenizer"],
                        meta["source_hash"], config["max_input_tokens"], config["max_new_tokens"]))
        # 读取结构化输入：path，保留原有样本身份字段。
        groups[group] = read_jsonl(path)
    if len(identities) != 1:
        raise ValueError("Main groups must share base model, seed, tokenizer, code and budget")
    result = chapter4.main_report(data, groups)
    save(args.output, {**result, "data_hash": data_hash, "databench_eval_version": version("databench-eval")})
    lines = ["| Group | All accuracy (n) | Short accuracy (n) | Long accuracy (n) |",
             "|---|---:|---:|---:|"]
    for row in result["table"]:
        cells = [row["group"]]
        for key in ("all", "short", "long"):
            accuracy = row[f"{key}_accuracy"]
            rendered = f"{accuracy:.2%}" if accuracy is not None else "—"
            cells.append(f"{rendered} ({row[f'{key}_n']})")
        lines.append("| " + " | ".join(cells) + " |")
    Path(f"{args.output}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# 解析双章实验子命令，并调用对应的采集、重放或报告流程。
# 输入：argv（命令参数列表）。
# 相关调用：read_jsonl。
# 返回形式含：cmd。
def main(argv=None):
    # 定义本入口的参数解析器；位置参数、默认值和允许取值由相邻 add_argument 明确声明。
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    # 注册一个实验子命令及其位置参数，供统一解析入口使用。
    # 输入：name；*positionals。
    # 返回形式含：cmd。
    def command(name, *positionals):
        cmd = commands.add_parser(name)
        for item in positionals:
            cmd.add_argument(item)
        return cmd

    cmd = command("prepare4", "tables", "output")
    cmd.add_argument("--revision", required=True)
    command("declare3", "data", "config", "output")
    cmd = command("predict3", "group", "data", "config", "output")
    cmd.add_argument("--declarations")
    cmd.add_argument("--reviewer")
    cmd.add_argument("--gate")
    cmd = command("audit3", "split", "data", "predictions", "reviews", "output")
    cmd = command("student3", "group", "budget", "data", "predictions", "reviews", "output")
    cmd.add_argument("--input-limit", type=int, required=True)
    command("partition3", "fixed", "constrained", "output")
    for name in ("predict4", "repair4", "records4", "diagnose4"):
        cmd = command(name, "group", "data", "tables", "tokenizer", "output")
        cmd.add_argument("--ablation", default="none")
        if name != "records4":
            cmd.add_argument("--config", required=True)
        if name == "predict4":
            cmd.add_argument("--teacher", action="store_true")
            cmd.add_argument("--limit", type=int)
        if name == "repair4":
            cmd.add_argument("--failures", required=True)
            cmd.set_defaults(teacher=True)
        if name == "records4":
            cmd.add_argument("--traces")
        if name == "diagnose4":
            cmd.add_argument("--phase", choices=["dev", "test"], required=True)
            for option in ("train", "rules", "pairs"):
                cmd.add_argument(f"--{option}", required=True)
            cmd.add_argument("--dev")
            cmd.add_argument("--predictions")
    cmd = command("main4", "data", "output")
    for i in range(5):
        cmd.add_argument(f"--L{i}", required=True)
    command("cost4", "data", "predictions", "output")
    # 解析本次命令的实际参数，后续样本路径和实验开关均由此确定。
    args = parser.parse_args(argv)
    if args.command == "prepare4":
        prepare_databench(args)
    elif args.command == "declare3":
        declare3(args)
    elif args.command == "predict3":
        predict3(args)
    elif args.command == "audit3":
        save(args.output, chapter3.teacher_audit(read_jsonl(args.data), read_jsonl(args.predictions),
                                                read_jsonl(args.reviews), split=args.split))
    elif args.command == "student3":
        save(args.output, chapter3.student_audit(read_jsonl(args.data), read_jsonl(args.predictions),
                                                read_jsonl(args.reviews), group=args.group, budget=args.budget,
                                                input_limit=args.input_limit))
    elif args.command == "partition3":
        # 读取结构化输入：args.fixed，保留原有样本身份字段。
        save(args.output, chapter3.compare_partition_targets(read_jsonl(args.fixed), read_jsonl(args.constrained)))
    elif args.command == "predict4":
        predict4(args)
    elif args.command == "repair4":
        repair4(args)
    elif args.command == "records4":
        records4(args)
    elif args.command == "diagnose4":
        diagnostics4(args)
    elif args.command == "main4":
        main4(args)
    elif args.command == "cost4":
        save(args.output, chapter4.cost_report(read_jsonl(args.data), read_jsonl(args.predictions)))


if __name__ == "__main__":
    main()
