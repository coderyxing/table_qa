# 思维链引导与高效表格问答研究工程

实现两条共用执行器的路线：依赖约束与证据保留的表格操作链；保守压缩、状态依赖分块与LoRA序列蒸馏。

**当前可运行：** 无GPU的执行器/协议测试、合成演示、三数据集适配、模型推理入口、训练记录构建、LoRA训练入口、实验清单、泛化划分、预算工具、机制报告及案例审核清单。

**服务器验证进度：** 7B容量、172来源损失平衡及一次性200条确认均已完成；确认集原配置WikiTQ/TabFact为28%/54%，所选7B操作模型＋3B失败回退为34%/76%。WikiTQ差值区间仍跨0，提升混合容量和恢复工程，尚非正式test或创新模块独立收益。见[后续实验与边界](docs/next-experiments-2026-11-06.md)。

**面向Chain-of-Table的改进：** 已核验原论文算法与指标，A3仍为适配基线。新增默认关闭的严格数量转换组件和训练集配对诊断；本机/GPU12已验证140项测试。完整候选创新、公式、代码入口、对照公平性与后续实验见[改进方案](docs/chain-of-table-improvement.md)。新工具的测试通过不表示准确率已经提升。

**语义能力继续验证：** 新增有字符跨度记录的抽取工具；8表三组诊断没有证明提分，结果和96条合成训练候选见[语义诊断](docs/semantic-capability-results.md)。已固定并核验Chain-of-Table源码，发现其原有子串检查不能当作本文创新；TabFact本地文本后端通过原源码回放及两条真实训练输入诊断（1完成、1精确标签格式失败），详见[复现核验](docs/chain-of-table-reproduction-audit.md)。本机与GPU12均通过162项测试。

**受控学生训练：** R/M两组3B各完成1epoch/38步、608记录和37106监督token，分层预算匹配、全部192条评测及完成审计已核验。P/R/M的WikiTQ为20%/20%/10%、TabFact为50%/60%/60%（各20条）；合成新实例为50%/16.7%/100%（24条）。合成提升未转化为真实任务提升，**不采用M**，详见[完整结果及负向案例](docs/curriculum-student-control.md)。本轮实验使用冻结的167项测试版本，条件义务原型未参与。

**FeTaQA无人工审核路线：** 使用512条官方答案的普通B0监督，固定修复版权重在额外100题开发集上，BLEU为21.978→25.612，ROUGE-1/2/L为0.523/0.312/0.427→0.632/0.409/0.520。更小12题的较大增幅与旧无效输出也完整保留，不据此声称超过Chain-of-Table。自动指标不等于事实审核，见[协议与结果](docs/fetaqa-without-human-review.md)。当前184项测试在GPU12全部通过；本机180项通过、4项可选库测试跳过。

**训练修复：** 实测发现旧Trainer累积16时梯度未取平均；现已显式归一化，并通过梯度、日志与实际参数更新的完整批次等价检查。旧检查点结果保留，但正式实验需统一修复版重跑；详见[问题及版本边界](docs/gradient-accumulation-fix.md)。

**问题条件原型：** 新增默认不接入推理的跨块条件义务验证器；严格操作绑定、条件抽取完整性和行域检查仍有适用边界，见[原型说明](docs/condition-contract-prototype.md)。进一步查新与已有思想的界线见[文献核验补充](docs/novelty-boundaries-2026-11-06.md)。

七类实验的实际完成范围与论文证据缺口见[覆盖状态](docs/experiment-coverage-status.md)，开发诊断不计作正式主实验完成。

**连续执行（2026-11-07）：** 已启动相同684来源的修复版B0/B2/B5多任务队列，随后自动接续32B教师五组内核消融，再采集512条新的短答/验证训练轨迹并弱筛选。官方判分、审计与材料收集自动接续，运行中尚无完整阶段结果。离线172条机制检查显示保守压缩及自适应分块代理收益仍很小，未据此宣称创新成立。队列、预算和边界见[本轮连续实验](docs/continuous-experiments-2026-11-07.md)。

## 本地验证（只需Python 3.10+）

```bash
python -m unittest discover -s tests -v
python -m tableqa demo --output runs/demo
python -m tableqa manifest runs/manifest.jsonl
python -m tableqa report runs/demo/examples.jsonl runs/demo/predictions.jsonl runs/demo/report
```

demo使用明确标记的回放模型和合成样本，不下载模型、不调用API、不代表准确率实验。

## 双A40服务器部署

先上传本仓库，在Linux下运行`bash scripts/preflight.sh`检查两卡、互联、内存和磁盘。保留现有驱动，按其兼容情况安装PyTorch；不假定`nvidia-smi`显示12.2就安装了Toolkit。环境建议Python 3.11。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train,metrics]'
python -m pip freeze > runs-environment.lock.txt
```

可选依赖限定在保守的Transformers 4.x/PEFT API范围，不强制升级到最新版。完整GPU环境需实际验证并保存freeze；当前范围不是已验证的锁文件。模型与数据下载需服务器网络和存储。

32B BF16仅权重约64GB，还需KV cache和缓冲，先以200条和小batch实测。教师阶段独占两卡；3B/7B学生各用一张卡。正式学生计时停止其他干扰任务，不启用响应缓存。

## 数据准备

从作者仓库获取数据并保留版本：WikiTableQuestions官方TSV及对应CSV；TabFact split JSON及`all_csv`；FeTaQA split JSON/JSONL。此工程不抓取网页或猜测标签。转换例子：

```bash
python -m tableqa adapt wikitq RAW/training.tsv data/wtq_train.jsonl --split train --table-root RAW
python -m tableqa adapt tabfact RAW/train_examples.json data/tabfact_train.jsonl --split train --table-root RAW/all_csv
python -m tableqa adapt fetaqa RAW/train.jsonl data/fetaqa_train.jsonl --split train
python -m tableqa audit data/all_splits.jsonl runs/audit.json
```

`all_splits.jsonl`由三任务规范化记录合并，需保留dataset/id/split。表格包含`headers`和二维`rows`，重复表头映射为不同内部列ID。原始值保留，不将未知值转为0。

## 教师生成与学生训练

实施前在配置加入确切`revision` commit hash；若使用cache必须固定revision。元数据保留配置、输入hash和软件版本。不要把密钥写入配置；API后端从`TABLEQA_API_KEY`读取。

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m tableqa predict data/train.jsonl configs/teacher.json runs/teacher_train.jsonl
python -m tableqa build-records data/train.jsonl runs/teacher_train.jsonl runs/b5_records.jsonl --mode B5 --max-block 4
CUDA_VISIBLE_DEVICES=0 python -m tableqa train runs/b5_records.jsonl configs/student_train.json runs/student-b5
CUDA_VISIBLE_DEVICES=0 python -m tableqa predict data/test.jsonl configs/student_predict.json runs/b5_test.jsonl
python -m tableqa report data/test.jsonl runs/b5_test.jsonl runs/b5_report --freeform-metrics
```

第二次预测同路径需`--resume`，配置或数据hash变化将被拒绝。测试参考答案由预测层白名单隔离，不发送模型。输出保留失败，不因异常删除样本。

短答案和标签轨迹用弱匹配筛选。FeTaQA操作监督需先导出独立双人审核材料，再合并两人的完成记录；仅手填`trajectory_review_approved:true`不能通过审核。批准绑定原始样本、预测与操作链哈希，任何变动需重新审核；分歧与不确定项不自动入池。详见[审核操作](docs/analysis-protocol.md#双人审核)。完整数据训练前检查实际接受量与分布。

B0可复用同一接受样本池，B1需单独A1教师公开理由，不能仅将操作链改名。`build-records --mode B1`读取对应预测responses的rationale。训练器支持`initial_adapter`继续源任务检查点及`resume_from_checkpoint`恢复训练。

训练默认按epoch保存；可传`--validation source_validation_records.jsonl`按验证loss选择检查点。用`build-records --validation-records`显式生成验证记录，该文件保持split=validation，只能来自当前协议允许的源验证集，禁止把目标任务验证集用于零样本调参。无验证文件时只保存训练过程，不宣称已完成验证选型。

等训练token预算在训练配置加入`target_token_budget`（每epoch目标token上限），平衡block/answer后再裁剪完整样本，实际计数保存在训练run.json。固定相同epoch并报告剩余预算。

当前训练模板显式采用`sampling_policy:coverage`，保证预算裁剪前保留每条原始记录，再扩充数量较少的监督类型。旧replacement策略仅为复核旧试跑保留。检查`training_exposure.json`及实际源ID覆盖；设置token预算后仍可能丢失部分原始样本，不能仅凭输入池相同声称严格等样本训练。

## 泛化、预算和分析

```bash
python -m tableqa split data/all_splits.jsonl runs/leave_fetaqa --target fetaqa
python -m tableqa split data/wtq_splits.jsonl runs/leave_sports --topic sports --annotations topics.json
python -m tableqa budget runs/b5_records.jsonl Qwen/Qwen2.5-3B-Instruct runs/b5_budget.jsonl --target-tokens 100000
python -m tableqa cases runs/b5_test.jsonl runs/case_review.jsonl
python -m tableqa compare data/test.jsonl runs/b2_test.jsonl runs/b5_test.jsonl runs/paired.json
```

`manifest`生成的规格需要按实验协议执行，不是所有行直接复用同一全任务检查点。完整分组、少样本、审核和参数操作见[研究设计](docs/research-design.md)。

报告目录包含HTML、JSON、机制统计和官方评测输入。WTQ内置分数仅诊断，必须接作者官方评测；FeTaQA自动指标也不能替代人工事实一致性审核。文献登记表见[对比方法核验表](docs/literature.csv)。

## 工程约束

- 核心操作不运行模型生成的任意Python/SQL；使用白名单JSON。
- 删列、聚合不可逆，通过不可变快照恢复。
- 自由文本答案不等于经过语义证明的答案，结果明确标记`not_proven`。
- 长输入报错并计入失败，当前不提供未经验证的自动长表截断。
- 参数、提示和测试协议冻结后再跑正式测试；不按测试成绩调参。

HF接口实现参考[LoRA文档](https://huggingface.co/docs/peft/en/package_reference/lora)和[Trainer文档](https://huggingface.co/docs/transformers/en/main_classes/trainer)。
