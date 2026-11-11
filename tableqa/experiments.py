# 模块职责：把实验组名转换为明确推理开关，并生成带预算、种子和划分约束的实验清单。
# 接口约定：样本、操作和结果采用本项目的结构化字典；细节见各入口及字段检查。
"""Executable experiment settings and bounded one-factor manifest generation."""
from dataclasses import asdict
from .inference import Settings,compact_state


# 由实验模式设置默认开关，再严格检查显式覆盖参数。
# 输入：mode（实验模式）；**overrides。
# 返回形式含：cfg。
def settings(mode, **overrides):
    cfg = Settings(mode=mode)
    cfg.compact = compact_state(mode)
    cfg.max_block = 4 if mode in {"B4", "B5"} else 1
    if mode in {"A0","A1","B0","B1"}: cfg.max_repairs=0
    if mode == "A3": cfg.dependency=False; cfg.evidence=False; cfg.max_repairs=0
    for k,v in overrides.items():
        if not hasattr(cfg,k): raise ValueError(f"Unknown setting {k}")
        setattr(cfg,k,v)
    return cfg


# 枚举主实验、消融、迁移和参数扫描规格，元数据与推理设置分开保存。
# 返回形式含：runs。
def manifest():
    runs=[]
    # 登记一条实验规格，把外层元数据和推理覆盖设置分别保存。
    # 输入：family；mode（实验模式）；model（生成后端）；seed（随机种子）；datasets；**kwargs。
    def add(family, mode="B5", model="3B", seed=42, datasets=None, **kwargs):
        runs.append({"family":family,"mode":mode,"student_size":model,"seed":seed,
                     "datasets":datasets or ["wikitq","tabfact","fetaqa"],**kwargs})
    for seed in [42,123,2026]:
        for mode in ["A0","A1","A2","A3","A4"]:
            add("teacher_main",mode,"32B",seed,implementation="adapted" if mode in {"A2","A3"} else "ours")
        for mode in ["B0","B1","B2","B3","B4","B5","B5_no_blocks"]:
            for budget in ["equal_examples","equal_target_tokens"]:
                add("student_main",mode,seed=seed,budget=budget)
        for name,override in [("no_dependency",{"dependency":False}),("no_evidence",{"evidence":False}),
                              ("no_repair",{"max_repairs":0}),("neither",{"dependency":False,"evidence":False})]:
            add("teacher_ablation","A4","32B",seed,ablation=name,settings=override)
    for model in ["1.5B","7B"]:
        for mode in ["B0","B2","B5"]: add("student_scale",mode,model)
    for target in ["wikitq","tabfact","fetaqa"]:
        for mode in ["B0","B2","B5"]:
            for n in [0,32,128,512]: add("task_transfer",mode,target_dataset=target,target_train_examples=n)
    for topic in ["sports","geography","people","other"]:
        for mode in ["B0","B2","B5"]: add("topic_transfer",mode,datasets=["wikitq"],target_topic=topic)
    for parameter,values in {"max_block":[1,2,4,8],"lora_rank":[8,16,32,64],
                             "answer_weight":[.5,1,2],"max_repairs":[0,1,2]}.items():
        for value in values: add("parameter",datasets=["wikitq"],selection_split="validation",parameter=parameter,value=value)
    for mode in ["B2","B5"]:
        for fraction in [.25,.5,1.]: add("data_scale",mode,fraction=fraction)
    for mode in ["A4","B0","B2","B5"]:
        for perturbation in ["row_shuffle","column_shuffle","irrelevant_column"]:
            add("robustness",mode,perturbation=perturbation,requires_semantic_review=True)
    for i,r in enumerate(runs):
        r["run_id"]=f"exp-{i:04d}"; r["status"]="planned"
        if r['mode']=='B5_no_blocks': r['equivalent_to']='B3; reuse B3 result with matched seed/budget'
        r["prediction_settings"]=asdict(settings(r["mode"],**r.get("settings",{})))
        if r.get("parameter") in {"max_block","max_repairs"}:
            r["prediction_settings"][r["parameter"]]=r["value"]
    return runs
