# 梯度累积归一化：实测问题、修复与旧结果边界

2026-11-06检查当前实际安装的Transformers 4.46.3/Accelerate 1.1.1时，发现原自定义Trainer只在compute_loss中计算每样本目标token平均，没有在反传前除以累积微批次数。这个版本的Trainer由自身循环管理累积，实际Accelerator.backward的divisor为1；原生training_step对自定义损失的反传没有替我们完成所需的平均。

若完整有效批次含G个相同大小微批次，所需目标为

\[
L=\frac1G\sum_{g=1}^{G}L_g,\qquad
\nabla L=\frac1G\sum_g\nabla L_g.
\]

旧实现累积的是\(\sum_g\nabla L_g\)，相对目标放大G倍。对当前模型存在的loss_kwargs参数，日志也会将未归一化值累加。训练/验证损失曲线不能直接混用这些旧值。

## 用实际运行库验证

scripts/audit_gradient_accumulation.py读取本工程冻结的旧training.py嵌套Trainer，以同一个CPU小模型、相同全批次损失为参考；不是仅做手写代数测试。本机留存runs/fetaqa-transfer-local/gradient-accumulation-audit-20261106.json，包含旧源码哈希、当前模块哈希、运行库与测量。

|累积微批次G|旧梯度/完整批次参考|修复后梯度/参考|修复后最大绝对梯度误差|
|---|---:|---:|---:|
|2|2.0|1.0|2.98e-8|
|16|16.0|1.0|8.94e-8|

另有tests/test_trainer_gradient.py两个实际Trainer集成检查：使用不等目标长度、不同样本权重、block/answer系数及答案权重，验证累积梯度与日志损失；再比较实际训练循环下micro=1/G=2和micro=2/G=1的一步SGD参数更新。均在服务器通过，不把无torch的本地跳过视作已运行。

## 现行实现

training.make_typed_trainer_class保留每样本目标token归一化、类型系数与样本权重，覆盖training_step，显式向Accelerator.backward传入

\[
L_g\cdot\frac{G_{\mathrm{Accelerator}}}{G}.
\]

Accelerator自己的除数被抵消，实际反传为\(L_g/G\)。返回日志为detach(L_g)/G；eval的compute_loss保持原目标值。token数量kwargs不充当样本平均的分母，避免把每样本均值改成全批次token均值。

该实现限定普通单设备Accelerate，拒绝Apex、DeepSpeed及多GPU；原训练入口也要求只暴露一张GPU。训练数据先对齐完整有效批次，frozen模式禁止额外裁剪或填充；实际梯度归一化版本、配置累积步数与Accelerator除数记录在statistics和曝光摘要中。

## 旧实验如何处理

所有旧快照、检查点和输出保留。R/M共享旧实现、相同累积16，源ID/监督token匹配和原始预测仍是真实记录；但它们对应旧训练行为，不能作为现行平均交叉熵公式的实现证据。A—G等使用旧训练入口的检查点也要保留这一版本限制。

不能只将旧loss除以16后声称完成修复，或假定学习率除以16就与新优化过程相同；梯度裁剪、Adam及数值行为需实际重跑检查。后续论文主实验应统一使用修复版，关键消融和预算对照也需在同版重跑，不跨版本将变化归因给创新模块。

FeTaQA答案监督v1保留为旧实现探索；v2采用显式平均重新训练相同512条来源。初版启动脚本SD评测漏传适配器，原SD输出标为无效；训练本身完成且权重可用，另用verify_fetaqa_answer_pilot.py按相同输入/参数实际激活LoRA重评，结果只认verified-evaluation目录。该入口问题也已在主脚本修正并加入加载检查，不复写旧文件。
