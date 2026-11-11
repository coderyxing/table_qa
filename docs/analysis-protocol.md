# 开发结果分析与人工审核协议

## 当前产物与适用范围

截至2026年11月5日，600条A4/A0队列均完成，仍是**训练集开发实验**，不是论文测试成绩。先固定每任务200条样本，再分别运行A4和A0；不根据回答正确性或长度筛选输入。代码保存在独立frozen-code中，新增工具没有修改该批次。完整结果、实际抽样覆盖差异和后续控制见`development-results-2026-11-05.md`。

已经完成的同30条调试样本对照：A0直接回答的WikiTQ为5/10，A4-v5为4/10；TabFact为10/10与6/10。FeTaQA token-F1分别约0.494与0.421，不能替代事实一致性。调试样本小、已经参与接口迭代，结果只定位限制；目前没有证据宣称操作链优于直接回答。

本地实测副本位于`runs/server-verification/2026-11-05-analysis`，服务器原始报告位于`runs/analysis/teacher-direct-vs-chain`。副本不包含模型权重或凭据；完整预测保留在服务器。

## 完整结果与配对统计

```bash
python scripts/analyze_runs.py SAMPLES A_PREDICTIONS OUTDIR \
  --predictions-b B_PREDICTIONS \
  --official-a WTQ_A/scores.json --official-b WTQ_B/scores.json
```

工具按dataset+id校验完整覆盖，拒绝未知ID、重复ID和混合方法。失败仍保留在任务分数分母和端到端耗时内。报告区分运行成功率、操作链可执行率、直接回答格式成功率；合法输出但答案错误单列，不把它们记成执行失败。

输出为`analysis.json`、带UTF-8 BOM的`summary.csv`和`report.md`。配对差值为B−A，默认按相同样本重采样2000次，得到百分位区间。该区间描述当前样本重采样的变化，不代替三个训练随机种子的均值/标准差，也不能消除开发集反复调试的偏差。

报告的A/B延迟比只是观测比值。正式效率实验必须固定硬件、精度、后端、warm-up、生成限制与样本集；保存质量与失败率，不能靠快速失败宣称加速。有响应缓存命中时不输出加速比。

## 官方WikiTQ判分与输入绑定

```bash
python scripts/evaluate_wtq_official.py SAMPLES PREDICTIONS WTQ_OUT \
  --evaluator data/raw/wikitq-official/WikiTableQuestions-master/evaluator.py \
  --tagged data/raw/wikitq-official/WikiTableQuestions-master/tagged/data
```

使用作者v1.0.2算法和CoreNLP targetCanon，经Python 3.12的lib2to3语法转换运行，并保留原作者许可。称为“官方算法的Python3语法端口”，不称未经改动的官方可执行程序。源文件、转换文件、tagged文件和输入预测均有哈希，另做数值、日期、字符等4项自检。

分析器校验逐样本官方分数与当前样本/预测的结构哈希，拒绝复用其他版本输出的旧分数。medium-development旧包装器早于此绑定功能，现已用新脚本另目录重新判分。复核其他旧队列时也需显式重新判分；保留原文件，不能补写伪造哈希。

CPU后处理`medium_analysis_stage.sh`已完成：用独立冻结副本在`runs/medium-analysis`重新判分、生成完整配对报告和200条FeTaQA双人审核材料。结果与主队列旧判分分目录保留，不把部分预测补成完整报告。

WikiTQ未提供官方分数时只报告DIAGNOSTIC_normalized_set_match。FeTaQA token-F1始终明确标为诊断值，不能写成事实一致性。不同评测方式不混用配对比较。

## 运行中只读检查

```bash
python scripts/analyze_runs.py runs/medium-development/samples.jsonl \
  runs/medium-development/A4-predictions.jsonl runs/analysis/medium-partial --allow-partial
```

只读取已完成JSONL行，不修改预测文件；输出标记partial、缺失数和各任务完成/计划数。600条队列按任务顺序运行，早期仅WikiTQ有输出，不能据此计算三任务最终均值或判断样本质量。完整配对比较禁止allow-partial。

## 无损输入表示候选

`serialization.py`只把状态rows中的列ID字典转换为与columns对齐的数组，保留行列ID、原始值、重复表头、顺序和其他状态字段，并检查精确往返一致。该候选是控制输入开销的工程变量，不额外宣称第三项论文创新。

使用本地3B tokenizer、相同系统提示与完整初始消息，对固定600条训练样本审计：

|任务（各200条）|字典表示总token|数组表示总token|减少比例|初始输入超过4096条数：前→后|
|---|---:|---:|---:|---:|
|WikiTQ|589193|444584|24.54%|32→13|
|TabFact|351954|290805|17.37%|0→0|
|FeTaQA|366904|293811|19.92%|3→0|

这是输入token测量，**不是回答质量不变的证据**；修复提示也可能增加上下文。计数适用于此次tokenizer与提示版本，不外推其他模型。

`serialization_followup.sh`已完成：从固定600条中按seed2027每任务抽10条，不看标签；使用原A4输出为配对字典基线，数组候选保持相同32B、贪心解码、4096输入限制、1024生成限制及执行配置。运行前检查core/inference/models/experiments字节一致。失败7→4条，但小样本不能证明总体质量保持。两个运行不同时发生，因此延迟只作观测，不构成受控加速结论。

## 双人审核

```bash
python scripts/human_review.py export SAMPLES PREDICTIONS REVIEW_DIR --purpose train_filter
# 两位审核人独立完成REVIEW_DIR/reviewer1.jsonl和reviewer2.jsonl后：
python scripts/human_review.py merge REVIEW_DIR/packet.json \
  REVIEW_DIR/reviewer1.jsonl REVIEW_DIR/reviewer2.jsonl REVIEW_RESULT_DIR
```

导出`review.html`、完整`packet.json`和两份初始值为null的JSONL。HTML包括标题、完整表格、问题、参考答案、模型答案、原始证据坐标、操作和失败消息。黄色单元格只是执行器记录的来源，不表示人工认可。原始highlighted_cells按数据集坐标单列展示，不将其冒充经转换的证据标注。

两位审核人完成前不互看判断。每行填写真实的reviewer标识、decision（approve/reject/uncertain）、六项checks与notes：

|检查项|填写标准|
|---|---|
|answer_faithful|答案所有事实均有表格/题目上下文支持，无张冠李戴或错误比较|
|answer_complete|覆盖题目要求的主要事实；不得仅因与参考文本不同就拒绝|
|evidence_valid|追踪单元格能支持实际筛选、比较、聚合及答案；来源存在不等于语义充分|
|evidence_complete|推导所需依据未被遗漏，拒绝只看最终答案猜测证据|
|operations_semantically_valid|操作实现题意，包含正确条件、聚合范围、顺序和数量单位|
|source_annotation_conflict|源表、问题、参考答案或原始标注之间是否有待核实冲突|

approve要求前五项true、冲突false；reject须填写全部检查项及原因；uncertain可保留无法判断项为null，但必须写出疑点。两人不一致和uncertain保留待复核，不能多数表决或自动改成批准。源码不能验证两名标识是否来自独立真人，该限制记录在报告中。

合并时校验完整覆盖、两人不同标识、packet与item哈希、检查项逻辑和执行重放。仅两人一致approve且可重放的轨迹进入`approved-samples.jsonl`。批准附带样本、整条预测及操作链哈希；更换表格、答案、预测或操作均需重新审核。只手填布尔标志不会通过。

train_filter只用于训练集；validation_filter只用于当前协议允许的验证集，并保持split=validation，构建记录仍须显式allow_validation。evaluation用于测试/验证事实分析，**不输出训练批准样本**。所有测试样本仍参加正常评测，不能以人工审核筛掉困难问题。

审核报告的decision_agreement只描述两人决定的一致程度，不是模型准确率。开发队列全量审核用于训练筛选；正式论文事实评估另按预先固定的随机样本和盲化方案执行，不能用定向案例集估计总体事实一致性。

## 下一道验收

600条A4/A0、后续容量/恢复、预算匹配课程对照均已完成，最新状态以next-experiments-2026-11-06.md为准。FeTaQA暂无人工审核者，学生训练不自动纳入自由文本操作链；普通官方答案B0监督与自动指标路线可继续，见fetaqa-without-human-review.md。接口、预算及审核规则稳定后，再冻结正式三任务、多种子主实验和消融；目前不据开发结果写“全面优于基线”。
