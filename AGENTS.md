# Table-understanding 项目指南

## 项目定位

本仓库是表格问答研究工程，Python 包为 `tableqa`。论文方案包含两章独立实验，两章各自采集教师轨迹、训练学生适配器和评测。

| 章节 | 研究问题 | 数据与核心对照 |
|---|---|---|
| 第三章：条件校验与操作链蒸馏 | 校验题意条件及计算行域，检验学生能否自主生成操作链，并分析分块对上下文与效率的影响 | WikiTQ 为主，TabFact 和 FeTaQA 为扩展。比较教师校验方式、学生监督方式及分块策略 |
| 第四章：长表视图与代码化蒸馏 | 学生只见局部样例时，能否按题意选择候选范围，在完整原表上执行查询，并按需探查和修订 | DataBench SemEval 完整表任务。比较 L0–L4、探查与修复消融，以及三类配对诊断 |

以[双章实验方案的复现章节](docs/thesis-method-experiment-plan.md#修订版实验入口与复现顺序)为当前实验入口索引。较早的 README 和实验文档记录各自冻结版本的开发结果。组名有出入时，用实际 CLI、运行配置和产物元数据确认含义。

## 代码导航

| 位置 | 职责 |
|---|---|
| `tableqa/cli.py`、`tableqa/thesis.py` | 通用数据、预测、训练和评测命令。双章专用实验入口为 `python -m tableqa.thesis` |
| `data.py`、`serialization.py` | 三任务适配、表指纹、数据泄露审计和输入序列化 |
| `core.py`、`inference.py`、`models.py` | 表状态、白名单操作、来源追踪、推理与回滚。模型后端包括本地、API 和回放 |
| `contracts.py`、`chapter3.py` | 题意声明、行域核验、教师对照、二阶复核和过程审计 |
| `distillation.py`、`optimization.py`、`training.py` | 轨迹压缩、固定或状态成本分块、训练记录构建、LoRA 和实际曝光统计 |
| `longtable.py`、`chapter4.py` | 局部视图、全表 SQLite 查询、探查与修复、监督重放、DataBench 分组评测 |
| `evaluation.py`、`analysis.py`、`review.py`、`reporting.py` | 判分与配对分析、人工审核、报告和评测导出 |
| `configs/`、`scripts/`、`tests/` | 模型与训练配置、预检和各阶段实验脚本、基于 `unittest` 的测试 |
| `docs/`、`runs/` | 研究协议和历史记录、运行产物和冻结代码副本。修改当前实现时定位到根目录的源码 |

表中未带目录的 Python 文件都位于 `tableqa/`。

## 运行与验证

要求 Python 3.10+。核心执行和回放可使用标准库。训练装 `.[train]`，FeTaQA 自动评测装 `.[metrics]`，DataBench 数据与判分装 `.[databench]`。GPU 部署前核对 [requirements-server.txt](requirements-server.txt) 与目标环境的 CUDA、驱动和 PyTorch 兼容性，正式运行保存依赖锁定记录。

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s tests -p test_thesis_experiments.py -v
python -m tableqa demo --output runs/demo
python -m tableqa --help
python -m tableqa.thesis --help
```

- `demo` 使用合成表与 `ReplayModel`。需要验证新改动时可指定新的输出目录，保留既有运行。
- `manifest` 生成待执行的实验规格。常见通用流程为 `adapt → audit → predict → build-records → train → predict → report`。
- GPU12 记录对应历史试跑。双章正式方案使用 GPU14，运行前执行 `bash scripts/preflight.sh`。教师为双 A40 上的 32B BF16，学生为单 A40 上独立训练的 3B LoRA。模型、数据、缓存和运行产物放在有写权限的 `/memory` 项目目录。
- `data/`、`runs/`、`models/` 和 `.env` 已忽略。API 密钥通过 `TABLEQA_API_KEY` 或配置指定的环境变量读取。预测和训练产物使用独立目录，保留原始日志、失败记录和冻结配置。

## 数据、执行与训练约定

- 规范样本保留 `dataset`、`id`、`split`、`table_id`、`task`、`question`、`reference` 和 `table`。原表由 `headers` 和二维 `rows` 表示。跨任务按 `(dataset, id)` 对齐，跨划分同时检查表 ID 和内容指纹。DataBench 发布的 dev 在本地清单中记为 `validation`。
- 使用稳定的 `c0`、`r0` 等内部 ID，重复表头保留独立列 ID。保留原始值和来源坐标，区分空值、零和解析失败。块执行在成功后提交，失败使用原状态修复。操作参数走白名单 JSON。第四章由执行器编译参数化查询，模型只输出受限计划。
- 推理只读取可见输入。参考答案、人工裁决和教师未来步骤分别留在判分或监督目标侧。训练轨迹来自训练划分，开发集用于选择与冻结，测试集用于最终评价。短答答案匹配属于弱筛选。FeTaQA 操作链监督要求绑定样本、预测与操作哈希的独立双人审核。
- 训练和推理使用一致的提示模板与操作配置。核对 prompt、padding 和 EOS 的损失边界。沿用修复后的梯度累积实现，检查实际监督 token、原始来源曝光、有效批次和优化步数。相同来源与近似监督 token 分开报告。历史梯度尺度问题见[训练修复记录](docs/gradient-accumulation-fix.md)。

## 双章实验口径

### 第三章

- `declare3` 生成参考答案隔离的条件声明。`predict3` 运行 `fixed`、`dynamic`、`conditions`、`row_domains` 和 `no_repair`。声明校验目前按绑定操作与输入行域工作，保留未验证状态。人工审核通过候选计划哈希关联。
- `audit3` 以人工确认合法的首个候选计划为误拒率分母。开发集误拒率严格大于 5% 时启用 `llm_review` 对照，报告错误放行与额外成本。
- `B0` 为答案监督，`B2` 为完整链逐步，`B3` 为压缩链逐步，`B5` 为紧凑状态下的压缩分块。比较固定与状态成本分块时保持底层链一致，用 `partition3` 检查标签边界差异，再用 `student3` 报告准确度、调用、延迟、上下文超限和人工审核的错误传播。
- 历史 192 条弱筛选轨迹的状态分块仅降低约 0.23% 的 token 代理成本。论文结论分别依据实测准确度、延迟、上下文与错误传播指标。A2 固定四步和 A3 动态链是适配基线，原版 Chain-of-Table 复现记录见[复现核验](docs/chain-of-table-reproduction-audit.md)。

### 第四章

- `prepare4` 固定 DataBench 数据版本和完整表指纹。`predict4` 采集或推理，`records4` 从学生可见状态重放后生成训练记录，`repair4` 用训练题中的真实失败生成修订。L0 是固定视图的答案监督，L1 是固定视图下的原始 3B 计划生成，L2 增加计划和读出蒸馏，L3 学习视图，L4 增加按需探查与修复。
- 视图标记 `coverage: sample`，最多 8 列、16 行、2048 token。候选范围由题意约束，统计在完整表执行。单轮 4096 token 中预留 512 token 生成空间。`main4` 用统一 tokenizer 和模板把完整提示超过 3584 token 的题归入长表，报告全量与长短表准确率，并按原表重采样估计 L3、L4 相对 L2 的长短表增益差。
- `cost4` 区分 SQLite 导入、CPU 探查扫描、CPU 查询、模型推理和端到端延迟。单查询白名单覆盖常用筛选、分组、去重及聚合，复杂计划和题意绑定的覆盖仍需审计。
- `diagnose4` 的三种配对分别改变视图样例、视图外答案或题意范围。dev 试构造并冻结规则，test 用于正式诊断。用表 ID、指纹和重复题检查与 train、dev 的交叉覆盖。人工核验成对答案及合法范围，分别报告首次计划质量与执行器拦截。

## 判分与结果使用

- 预测覆盖整个冻结评测集，失败保留在分母和延迟统计中。WikiTQ 的内置匹配用于诊断，正式分数使用 `scripts/evaluate_wtq_official.py` 和作者 v1.0.2 评测算法。TabFact 报告标签准确率。FeTaQA 的 BLEU、ROUGE 与人工事实一致性分开。DataBench 使用 `databench-eval`。详见[判分与审核协议](docs/analysis-protocol.md)。
- 公平计时固定硬件、精度、上下文和后端，先预热并禁用响应缓存。记录 token、调用、失败率及 P50/P95。正式核心对照按多个种子报告，分别保留样本或表级置信区间和训练种子方差。
- 将回放、预实验、开发结果与正式测试分别标注。已有代码验收和历史回放可作为工程证据。双章正式模型训练、人工审核和硬件计时需查验实际运行产物后登记。
