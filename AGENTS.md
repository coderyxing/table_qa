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

## 2026-11-11 验收记录与后续核查

- 用户已同意第三章先用两名独立子代理做 AI 双审。已覆盖 94 个不同首候选、55 道不同题目；原始判断、15 条分歧裁决和完整报告位于本地 `runs/thesis-20261111-acceptance/chapter3-review/`。必须标为 AI 审核，不能称为真人标注。
- 指定提交文件 `review-template.jsonl` 已填入全部 94 条最终 AI 裁决，与 `review-final-ai.jsonl` 相同；原空白模板为 `review-template.blank.jsonl`。分任务统计见 `audit-ai-by-dataset.json`，不要将三任务合并误拒率直接当作 WikiTQ 的正式结论。
- 冻结试跑的 conditions、row_domains 均有 17/20 合法首候选被误拒。9/60 与 39/60 是执行成功数，不是答案准确率。当前实现已规范化操作指纹中的执行默认参数；同候选离线检查中 conditions 误拒降至 8/20，但错误放行增加，row_domains 仍为 17/20，不能据此声称端到端问题已解决。
- 条件声明仍存在错误实体绑定、严格操作绑定与筛选前后行域混淆。尤其不能将教师声明的筛选后行 ID 直接当作操作前输入行域。真实二阶复核已完成，效果及成本见下文；默认参数修复与二阶复核是两项独立对照，不能合并成未经测量的新结果。
- 第四章 phase-v4 的六题预实验得到 6 条监督记录，实际仅来自 2 道合格来源题。扩大采集按不同来源题计数，不能用同一道题的多条监督记录充足 100 条轨迹。
- GPU14 扩大采集目录为 `/memory/tableqa/runs/chapter4-scale-20261111`，冻结源码为 `/memory/tableqa/runs/20261111-acceptance/source-plan-v4`。采集清单含 500 道训练题、49 张原表，按题原子保存并可续跑；进度以 `progress.json`、`audit.jsonl` 和实际轨迹文件核对。分别报告全部尝试中的可重放率、执行成功轨迹的重放一致率和官方答案通过产出率。不要改写运行中的冻结源码。
- 扩大采集有独立 CPU 验证脚本 `verify_artifacts.py`，会按冻结源码重新恢复原表、重放并核对官方答案与监督记录。`queue_v5.py` 仅在 v4 至少完成 200 题且有效可重放率仍低于 70%、原进程退出及 GPU 空闲后，启动独立的 `/memory/tableqa/runs/chapter4-scale-v5-20261111`。v5 仅改变计划提示，不更改执行器。必须核实实际 PID 和新旧进度，不能将准备完成视为新版本已运行或目标已完成。
- v4 已按冻结版本完成 200 题，得到 77 条合格独立轨迹、233 条监督记录，全部尝试的有效可重放率为 57%，执行成功子集重放一致率为 100%。已据此触发独立 v5 采集；最终不能混用两个版本的计数或仅依据前十题判断改进。
- v4 完整 200 题已通过独立重验，本地 `runs/chapter4-scale-20261111/final-v4-verified.zip` 保存 404 个经摘要校验的源码与运行文件；v5 同类归档为 `runs/chapter4-scale-v5-20261111/final-v5-verified.zip`，含 406 个已校验文件。进度说明为前者目录的 `scale-progress.md`。
- v5 的 `verify_after_exit.py` 在采集结束后启动完整 CPU 验收，句柄见运行目录中的 `verification-waiter.pid`。服务器 Conda Python 没有 `os.pidfd_open`，已使用 PID 与 `/proc` 启动时钟联合校验；首轮 pidfd 兼容性失败日志保留，不影响采集模型。
- v5 已完成 200 题，得到 92 条合格独立轨迹、276 条监督记录，有效可重放率 66.5%，仍未达标。已据此启动独立 `/memory/tableqa/runs/chapter4-scale-v6-20261111`，冻结源码为 `/memory/tableqa/runs/20261111-acceptance/source-plan-v6`，源码摘要 `dd70ccdb0885f6df24f501dc617377ec7a0579892d339cf0a8e0c3e1374ed89e`。不得把旧版本的轨迹数量相加当作新版本达标。
- v6 增加执行结果上的 exists/empty/数值字段比较，以及计数结果与结构常数 0/1 的比较；MIN/MAX 保留布尔类型，SUM/MEAN(bool) 返回计数/比例。缺少筛选 value 时反馈明确错误码。枚举范围校验分支 AST 保持不变。服务器完整检查共 214 项，212 通过、2 跳过；真实 tokenizer 对前版 200 题的 627 个提示检查无超限，最大 3200/3584 token。该预检不是 v6 模型效果验收。
- 项目根目录的 `longtable.py` 已同步为冻结 v6 实现，文件字节摘要与服务器一致；新增 `tests/test_longtable_result_reads.py`。本地完整检查 217 项，213 通过、4 跳过。学生训练应在采集产物通过最终验收后开始，并沿用同版本提示。
- v6 已完整结束并独立重验全部 200 道训练题：158 条执行并一致重放，全部尝试的可重放率 79%，执行成功子集一致率 100%；114 道不同来源题的答案通过官方检查，生成 346 条监督记录。达到本次扩大采集的数量与重放率门槛。v4、v5、v6 使用同一批 200 道题，不能写成 600 道不同题目。
- 最终 v6 归档为本地 `runs/chapter4-scale-v6-20261111/final-v6-verified.zip`，415 个文件已逐一核对摘要。完整 CPU 验收为包内 `run/verification-1794392382438133649/report.json`。真实 Qwen 3B tokenizer 核验全部 346 条记录：最长完整序列 2463 token，提示前缀与训练模板完全一致，无截断；该检查不等于学生训练已经完成。
- 第三章真实二阶复核的脚本和配置位于 `/memory/tableqa/runs/chapter3-second-stage-20261111`。已用本地 32B 对 27 个语义拦截后仍可执行的固定首候选完成复核，`fixture_only=false`，提示不含参考答案或 AI 标签。相同首候选的 AI 误拒率由 17/20（85%）降为 7/20（35%），错误放行由 6/31（19.35%）增为 8/31（25.81%）；27 次调用共 20090 输入 token、135 输出 token、28.60 秒生成耗时（不含加载）。这不是新一轮端到端准确率，不能称为真人误拒率。
- 二阶复核的本地最终归档为 `runs/chapter3-second-stage-20261111/final-second-stage-verified.zip`，105 个文件通过摘要核验；完整覆盖率重算和说明位于该目录的 `real-review/`。两种全放行/全拒绝 fixture 只验证连接，与真实结果分开保存。采集、独立验收和二阶复核均已退出，继续工作时先查产物，不要重复加载模型。
