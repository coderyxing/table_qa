# 配置文件阅读说明

配置采用标准 JSON，字段解释集中在本说明及对应加载入口中。

| 文件 | 入口及作用 |
|---|---|
| teacher.json | 通用 CLI 的 predict 命令。model_config 指定教师底座、设备分片和生成预算，settings 指定依赖、证据和修复开关 |
| student_predict.json | 通用 CLI 的学生预测。adapter 指定已训练权重，benchmark 开启同硬件预热和计时检查 |
| student_train.json | train 命令使用的 LoRA、采样、学习率、序列上限和梯度累积配置 |
| chain_of_table_protocol.json | 上游方法的冻结源码、许可、运行边界与已完成核验记录 |

model_config 的 backend 可以为 local、api 或 replay。local 的 model 是底座路径或模型 ID，adapter 是可选 LoRA 路径。revision 用于固定底座版本。max_input_tokens 控制完整提示上限，max_new_tokens 控制生成上限。API 密钥通过 key_env 指定的环境变量读取。replay 的 responses 是测试或轨迹重放使用的固定响应。

mode 对应实验方法。settings 只接受推理数据类中的字段。max_block 控制单次可提交的操作数，max_operations 控制整题操作预算，max_repairs 控制有限修复次数。dependency、evidence 及工具能力开关需要与监督记录和线上提示保持一致。

训练配置中 max_seq_length 是提示加目标的总长度。micro_batch 乘 gradient_accumulation_steps 得到有效批次大小。lora_rank 和 lora_alpha 控制适配器容量及缩放，learning_rate、max_epochs 控制优化过程。sampling_policy=coverage 在预算裁剪前保留原始记录覆盖，实际曝光还需读取 training_exposure.json。target_token_budget 按完整记录约束每轮目标 token，补齐累积批次后的真实计数随训练产物保存。

双章专用入口 tableqa.thesis 的 --config 直接接收生成后端参数对象，例如 backend、model 和 max_new_tokens，区别于通用 predict 配置外层的 model_config。运行前以对应命令的 --help 与加载函数核对格式。
