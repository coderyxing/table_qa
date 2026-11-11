# 模块职责：提供通用命令行入口，串联数据适配、预测、蒸馏记录、训练、预算和报告。
# 接口约定：样本、操作和结果采用本项目的结构化字典；细节见各入口及字段检查。
"""CLI: python -m tableqa <command>. No network or model downloads in demo."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from .data import read_jsonl, write_jsonl, digest


# 创建父目录后保存 UTF-8 JSON，为元数据与报告提供统一写入入口。
# 输入：path（输入或输出路径）；value。
# 相关调用：p.write_text。
def save(path,value):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")


# 登记依赖版本、源码摘要和运行配置，供后续结果追溯。
# 输入：config（显式运行配置）。
# 相关调用：digest、p.read_text。
# 返回形式含：字段 python、platform、packages、git_commit、source_hash、config。
def metadata(config):
    packages={}
    for name in ["torch","transformers","peft","accelerate"]:
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]=None
    try: commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError): commit=None
    # Never store API keys or entire environment.
    source_hash=digest({p.name:p.read_text(encoding="utf-8") for p in sorted(Path(__file__).parent.glob("*.py"))})
    return {"python":sys.version,"platform":platform.platform(),"packages":packages,
            "git_commit":commit,"source_hash":source_hash,"config":config}


# 解析子命令并路由到数据、推理、训练或分析接口。
# 输入：argv（命令参数列表）。
# 相关调用：write_jsonl、read_jsonl、Path(args.official_wtq).read_text、AutoTokenizer.from_pretrained。
# 异常约束：Search requires a compressed supervision mode
def main(argv=None):
    # 定义本入口的参数解析器；位置参数、默认值和允许取值由相邻 add_argument 明确声明。
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    q=sub.add_parser("adapt"); q.add_argument("dataset",choices=["wikitq","tabfact","fetaqa"])
    q.add_argument("source"); q.add_argument("output"); q.add_argument("--split",required=True,choices=["train","validation","test"]); q.add_argument("--table-root")
    q=sub.add_parser("audit"); q.add_argument("data"); q.add_argument("output")
    q=sub.add_parser('audit-trajectories'); q.add_argument('data'); q.add_argument('predictions'); q.add_argument('output');q.add_argument('--official-wtq')
    q=sub.add_parser("manifest"); q.add_argument("output")
    q=sub.add_parser("predict"); q.add_argument("data"); q.add_argument("config"); q.add_argument("output"); q.add_argument("--resume",action="store_true")
    q=sub.add_parser("evaluate"); q.add_argument("data"); q.add_argument("predictions"); q.add_argument("output")
    q=sub.add_parser("build-records"); q.add_argument("data"); q.add_argument("predictions"); q.add_argument("output"); q.add_argument("--mode",default="B5"); q.add_argument("--max-block",type=int,default=4); q.add_argument("--validation-records",action="store_true")
    q.add_argument('--compression-policy',choices=['legacy','search','guarded_search'],default='legacy')
    q.add_argument('--typed-quantities', action='store_true', help='Enable the experimental quantity conversion prompt')
    q.add_argument('--semantic-extract', action='store_true', help='Use extract-v1 source-copy operation prompt')
    q.add_argument('--tool-demonstrations', action='store_true', help='Include synthetic demonstrations of enabled tools')
    q.add_argument('--partition-policy',choices=['fixed','state_cost','state_blind'],default='fixed')
    q.add_argument('--tokenizer',help='Pinned local tokenizer path; required for state-cost partitions')
    q.add_argument('--max-target-tokens',type=int,default=512,help='Student output cap used in partition feasibility')
    q=sub.add_parser("train"); q.add_argument("records"); q.add_argument("config"); q.add_argument("output"); q.add_argument("--validation")
    q=sub.add_parser("split"); q.add_argument("data"); q.add_argument("output_dir"); q.add_argument("--target"); q.add_argument("--topic"); q.add_argument("--annotations"); q.add_argument("--seed",type=int,default=42)
    q=sub.add_parser("cases"); q.add_argument("predictions"); q.add_argument("output"); q.add_argument("--seed",type=int,default=42)
    q=sub.add_parser("perturb"); q.add_argument("data"); q.add_argument("reviews"); q.add_argument("kind",choices=["row_shuffle","column_shuffle","irrelevant_column"]); q.add_argument("output"); q.add_argument("--seed",type=int,default=42)
    q=sub.add_parser("budget"); q.add_argument("records"); q.add_argument("tokenizer"); q.add_argument("output"); q.add_argument("--target-tokens",type=int); q.add_argument("--seed",type=int,default=42); q.add_argument("--revision")
    q=sub.add_parser("compare"); q.add_argument("data"); q.add_argument("a"); q.add_argument("b"); q.add_argument("output")
    q=sub.add_parser("report"); q.add_argument("data"); q.add_argument("predictions"); q.add_argument("output_dir"); q.add_argument("--freeform-metrics",action="store_true")
    q=sub.add_parser("diagnose"); q.add_argument("data"); q.add_argument("predictions"); q.add_argument("replacements"); q.add_argument("output")
    q=sub.add_parser("demo"); q.add_argument("--output",default="runs/demo")
    # 解析本次命令的实际参数，后续样本路径和实验开关均由此确定。
    args=p.parse_args(argv)
    if args.command=="adapt":
        from .data import adapt
        # 保存逐条结构化产物到 args.output，供后续重放或审核读取。
        write_jsonl(args.output,adapt(args.dataset,args.source,args.split,args.table_root))
    elif args.command=="audit":
        from .data import audit
        save(args.output,audit(read_jsonl(args.data)))
    elif args.command=='audit-trajectories':
        from .quality import audit_run_trajectories
        official=json.loads(Path(args.official_wtq).read_text(encoding='utf-8')) if args.official_wtq else None
        write_jsonl(args.output,audit_run_trajectories(read_jsonl(args.data),read_jsonl(args.predictions),official))
    elif args.command=="manifest":
        from .experiments import manifest
        write_jsonl(args.output,manifest())
    elif args.command=="predict":
        predict_command(args)
    elif args.command=="evaluate":
        from .evaluation import evaluate, subgroup_report
        # 读取结构化输入：args.predictions，保留原有样本身份字段。
        ps=read_jsonl(args.predictions); data=read_jsonl(args.data)
        save(args.output,{"metrics":evaluate(ps,data),"subgroups":subgroup_report(ps,data)})
    elif args.command=="build-records":
        from .distillation import build_records
        from .evaluation import diagnostic_score
        from .quality import audit_trajectory
        data={(r["dataset"],r["id"]):r for r in read_jsonl(args.data)}
        tokenizer=None
        if args.partition_policy!='fixed':
            if args.mode!='B5' or not args.tokenizer:raise ValueError('State-cost partitions require B5 and a pinned local tokenizer')
            from transformers import AutoTokenizer
            # 使用本阶段指定的 tokenizer；预算、提示模板与监督掩码应沿用同一版本。
            tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
        if args.compression_policy!='legacy' and args.mode not in {'B3','B4','B5','B5_no_blocks'}:
            raise ValueError('Search requires a compressed supervision mode')
        out=[]; rejected=[]; optimization=[]
        for pred in read_jsonl(args.predictions):
            sample=data[(pred["dataset"],pred["id"])]
            expected_split="validation" if args.validation_records else "train"
            if sample["split"]!=expected_split: raise ValueError("Refusing trajectory filtering outside explicit train/validation purpose")
            # Exact short/label checks are weak supervision, not proof of a correct chain.
            if pred["status"]!="ok" or (sample["task"]!="freeform" and diagnostic_score(pred,sample)!=1):
                rejected.append(sample["id"]); continue
            from .review import review_approved
            if sample["task"]=="freeform" and not review_approved(sample,pred):
                rejected.append(sample["id"]); continue
            if args.mode not in {'B0','B1'} and not audit_trajectory(sample,pred)['decision'].startswith('accepted_'):
                rejected.append(sample['id']); continue
            if args.mode=="B1":
                response=json.loads(pred["responses"][-1]["text"])
                sample={**sample,"teacher_rationale":response["rationale"]}
            prepared=None;partition=None;compression_report=None;partition_report=None
            if args.compression_policy!='legacy' or args.partition_policy!='fixed':
                from .core import TableState
                from .distillation import compress
                from .optimization import optimize_operations,state_cost_partition
                state=TableState.from_table(sample['table'])
                if args.compression_policy!='legacy':
                    protect=('filter','sort','select','group_aggregate') if args.compression_policy=='guarded_search' else ()
                    prepared,compression_report=optimize_operations(state,pred['operations'],preserve_kinds=protect)
                else:prepared=compress(state,pred['operations'])
                if args.partition_policy!='fixed':
                    partition,partition_report=state_cost_partition(sample,prepared,tokenizer,args.max_block,
                        state_sensitive=args.partition_policy=='state_cost',allow_validation=args.validation_records,
                        max_output_tokens=args.max_target_tokens,typed_quantities=args.typed_quantities,
                        semantic_extract=args.semantic_extract,tool_demonstrations=args.tool_demonstrations)
                optimization.append({'dataset':sample['dataset'],'id':sample['id'],
                                     'compression':compression_report,'partition':partition_report})
            out.extend(build_records(sample,pred["operations"],args.mode,args.max_block,args.validation_records,
                compression_policy=args.compression_policy,partition=partition,prepared_operations=prepared,
                typed_quantities=args.typed_quantities,semantic_extract=args.semantic_extract,
                tool_demonstrations=args.tool_demonstrations))
        write_jsonl(args.output,out)
        # 保存逐条结构化产物到 args.output + '.optimization.jsonl'，供后续重放或审核读取。
        write_jsonl(args.output+'.optimization.jsonl',optimization)
        save(args.output+".filter.json",{"rejected_ids":rejected,"records":len(out),"semantic_correctness_proven":False,
            'compression_policy':args.compression_policy,'partition_policy':args.partition_policy,'tokenizer':args.tokenizer,
            'typed_quantities':args.typed_quantities,'semantic_extract':args.semantic_extract,
            'tool_demonstrations':args.tool_demonstrations})
    elif args.command=="train":
        from .training import train
        cfg=json.loads(Path(args.config).read_text(encoding="utf-8"))
        # 读取结构化输入：args.records，保留原有样本身份字段。
        stats=train(cfg,read_jsonl(args.records),args.output,read_jsonl(args.validation) if args.validation else None)
        save(Path(args.output)/"run.json",{**metadata(cfg),"statistics":stats})
    elif args.command=="split":
        from .data import protocol_split,nested_samples
        annotations=json.loads(Path(args.annotations).read_text(encoding="utf-8")) if args.annotations else None
        partitions=protocol_split(read_jsonl(args.data),args.target,args.topic,annotations)
        # 保存逐条结构化产物到 Path(args.output_dir) / (name + '.jsonl')，供后续重放或审核读取。
        for name,records in partitions.items(): write_jsonl(Path(args.output_dir)/(name+".jsonl"),records)
        if args.target:
            for n,records in nested_samples(partitions["target_train"],[32,128,512],args.seed).items():
                # 保存逐条结构化产物到 Path(args.output_dir) / f'fewshot_{n}.jsonl'，供后续重放或审核读取。
                write_jsonl(Path(args.output_dir)/f"fewshot_{n}.jsonl",records)
    elif args.command=="cases":
        from .evaluation import select_cases
        write_jsonl(args.output,select_cases(read_jsonl(args.predictions),args.seed))
    elif args.command=="perturb":
        from .protocols import perturb
        # 读取结构化输入：args.reviews，保留原有样本身份字段。
        reviews={(r["dataset"],r["id"]):r for r in read_jsonl(args.reviews)}
        result=[]
        for r in read_jsonl(args.data):
            review=reviews.get((r["dataset"],r["id"]),{})
            if args.kind in review.get("approved_perturbations",[]):
                result.append(perturb(r,args.kind,args.seed,True))
        if not result: raise ValueError("No reviewed semantic-preserving perturbations")
        write_jsonl(args.output,result)
    elif args.command=="budget":
        from .protocols import budget_records
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(args.tokenizer,revision=args.revision)
        records,stats=budget_records(read_jsonl(args.records),tok,args.target_tokens,args.seed)
        write_jsonl(args.output,records); save(args.output+".budget.json",stats)
    elif args.command=="compare":
        from .evaluation import evaluate,diagnostic_score,paired_bootstrap
        # 读取结构化输入：args.a，保留原有样本身份字段。
        data=read_jsonl(args.data); a=read_jsonl(args.a); b=read_jsonl(args.b)
        evaluate(a,data); evaluate(b,data)
        a={(p["dataset"],p["id"]):p for p in a}; b={(p["dataset"],p["id"]):p for p in b}
        result={}
        for d in sorted({r["dataset"] for r in data}):
            rs=sorted([r for r in data if r["dataset"]==d],key=lambda x:x["id"])
            result[d]=paired_bootstrap([diagnostic_score(a[(d,r["id"])],r) for r in rs],
                                      [diagnostic_score(b[(d,r["id"])],r) for r in rs])
        save(args.output,{"metric":"diagnostic_score_not_official_wikitq","paired":result})
    elif args.command=="report":
        from .evaluation import evaluate,subgroup_report
        from .protocols import mechanism_report
        from .reporting import export_official,html_report,freeform_metrics
        data=read_jsonl(args.data); predictions=read_jsonl(args.predictions)
        metrics=evaluate(predictions,data); root=Path(args.output_dir); root.mkdir(parents=True,exist_ok=True)
        result={"metrics":metrics,"subgroups":subgroup_report(predictions,data)}
        if args.freeform_metrics: result["freeform_metrics"]=freeform_metrics(predictions,data)
        mechanisms=mechanism_report(predictions)
        save(root/"evaluation.json",result); save(root/"mechanisms.json",mechanisms)
        export_official(predictions,data,root/"official_inputs")
        html_report(result,mechanisms,root/"report.html")
    elif args.command=="diagnose":
        from .protocols import replacement_diagnostic
        data={(r["dataset"],r["id"]):r for r in read_jsonl(args.data)}
        predictions={(r["dataset"],r["id"]):r for r in read_jsonl(args.predictions)}
        out=[]
        for replacement in read_jsonl(args.replacements):
            key=(replacement["dataset"],replacement["id"])
            out.append(replacement_diagnostic(data[key],predictions[key]["operations"],
                       replacement["index"],replacement["operation"],replacement.get("reviewed",False)))
        write_jsonl(args.output,out)
    elif args.command=="demo": demo(args.output)


# 校验输入及续跑身份，创建生成后端并逐题保存预测和计时。
# 输入：args（已解析的命令参数）。
# 相关调用：read_jsonl、digest、Path(args.config).read_text、meta.read_text。
# 异常约束：Duplicate dataset/id in prediction input
def predict_command(args):
    from .models import make_model
    from .experiments import settings
    from .inference import predict
    cfg=json.loads(Path(args.config).read_text(encoding="utf-8"))
    # 读取结构化输入：args.data，保留原有样本身份字段。
    data=read_jsonl(args.data); keys=[(r["dataset"],r["id"]) for r in data]
    if len(keys)!=len(set(keys)): raise ValueError("Duplicate dataset/id in prediction input")
    run_metadata=metadata(cfg)
    identity=digest({"config":cfg,"data":data,"source_hash":run_metadata["source_hash"]})
    out=Path(args.output); meta=Path(str(out)+".meta.json")
    if out.exists() and not args.resume: raise ValueError("Output exists; use --resume with matching config/data")
    # 读取结构化输入：out，保留原有样本身份字段。
    previous=read_jsonl(out) if out.exists() else []
    if args.resume and out.exists():
        if not meta.exists() or json.loads(meta.read_text(encoding="utf-8"))["identity"]!=identity:
            raise ValueError("Resume config or input changed")
    save(meta,{**run_metadata,"identity":identity,"data_hash":digest(data)})
    # 按已读取配置构造生成后端，实际模型身份和预算由该配置登记。
    model=make_model(cfg["model_config"]); done={(r["dataset"],r["id"]) for r in previous}
    options=dict(cfg.get("settings",{})); mode=options.pop("mode",cfg.get("mode","B5")); s=settings(mode,**options)
    if cfg.get("benchmark") and cfg["model_config"].get("cache"):
        raise ValueError("Benchmark forbids response cache")
    if cfg.get("benchmark"):
        if cfg["model_config"].get("backend","local")!="local": raise ValueError("GPU benchmark requires local backend")
        import torch
        if data: predict(data[0],model,s)  # warm-up never included in measured results
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("a",encoding="utf-8") as f:
        for sample in data:
            if (sample["dataset"],sample["id"]) in done: continue
            result=predict(sample,model,s)
            if cfg.get("benchmark"):
                result["metrics"]["peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
            f.write(json.dumps(result,ensure_ascii=False)+"\n"); f.flush()


# 用明确的合成题与回放响应验证执行、蒸馏记录和报告数据流。
# 输入：directory（数据或模型目录）。
# 相关调用：write_jsonl。
def demo(directory):
    from .data import example
    from .models import ReplayModel
    from .inference import predict
    from .experiments import settings
    from .distillation import compress,build_records
    from .core import TableState
    from .evaluation import evaluate
    sample=example("wikitq","train","synthetic-medals","demo-table",
          ["Year","Country","Medal"],[[2021,"A","Gold"],[2022,"B","Gold"],[2023,"A","Gold"],[2019,"B","Gold"]],
          "Which country won most gold medals after 2020?",["A"])
    ops=[{"op":"project","columns":["c0","c1","c2"]},
         {"op":"filter","column":"c0","cmp":"gt","value":2020,"type":"number"},
         {"op":"filter","column":"c2","cmp":"eq","value":"Gold"},
         {"op":"group_aggregate","by":["c1"],"aggregations":[{"fn":"count","as":"total"}]},
         {"op":"select","mode":"extreme","column":"total","which":"max"},
         {"op":"answer","columns":["c1"]}]
    pred=predict(sample,ReplayModel([{"operations":ops}]),settings("A4"))
    # 保存逐条结构化产物到 root / 'examples.jsonl'，供后续重放或审核读取。
    root=Path(directory); write_jsonl(root/"examples.jsonl",[sample]); write_jsonl(root/"predictions.jsonl",[pred])
    # 保存逐条结构化产物到 root / 'records.jsonl'，供后续重放或审核读取。
    write_jsonl(root/"records.jsonl",build_records(sample,ops))
    save(root/"report.json",{"fixture_only":True,"metrics":evaluate([pred],[sample]),
         "compressed_operations":compress(TableState.from_table(sample["table"]),ops)})
    print(json.dumps({"fixture_only":True,"status":pred["status"],"answer":pred["answer"],"output":str(root)},ensure_ascii=False))
