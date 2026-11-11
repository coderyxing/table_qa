# GPU12部署与执行记录

## 当前状态：2026年11月5日

已通过用户提供的ED25519公开指纹核验并用专用SSH密钥登录GPU12；项目为`/root/tableqa-research`。私钥在本机OneDrive之外，仓库不保存密码或私钥。

|项目|实际状态|
|---|---|
|服务器|Ubuntu20.04，RAM约251GiB，双A40各46068MiB，拓扑SYS，无NVLink|
|Python|项目venv基于已有Anaconda Python3.12.7，系统Python3.8未修改|
|运行时|torch2.5.1+cu124、transformers4.46.3、peft0.13.2、accelerate1.1.1|
|驱动|保留535.230.02；CUDA/BF16实测通过|
|模型|Qwen2.5-3B-Instruct及32B-Instruct均已逐文件SHA256校验，32B共17个权重分片|
|GPU验证|32B双卡BF16均衡推理、3B单卡LoRA训练及adapter重载通过|
|工程测试|本机与GPU12均通过131项标准库测试|
|正式论文实验|未完成；现有成绩均为工程/训练集开发结果|

实际环境锁保存在`runs/setup/environment.lock.txt`。不因nvidia-smi显示的CUDA版本推断已安装Toolkit。7B权重现已逐文件核验，LoRA运行待验证；1.5B尚未下载。

## 2026年11月6日准确率优先阶段

新教师60条整链/交互与搜索学生控制均已完成，激进删除搜索没有提升WikiTQ验证成绩，未作为默认。新`accuracy-development-v1`在200条按整表隔离的验证输入上比较4K/8K、1/2次修复、失败答案回退，另200条确认样本保留。A/C分别在GPU1/0运行，A后接B；这轮不是匹配时延基准。

7B四个权重分片SHA256核验完成。推理控制结束后，GPU0执行`scale-7b-accuracy`，GPU1执行`balanced-accuracy`。后者同172条审核来源、同采样顺序与20754目标token/epoch，仅改变任务/标签损失权重。所有调优组完成后，`accuracy-confirmation`冻结选择，在保留验证集做一次确认；正式test不动。

阶段标记均在同名runs目录stage.txt，日志在runs/setup同名文件。具体变量、公式、数据使用边界和验收见`accuracy-improvement.md`。131项测试通过不表示已验证新准确率提升。

## 现行队列

`runs/medium-development`固定每任务200条训练样本，seed2026，排除先前6条/30条调试样本。不按长度或标签正确性预筛输入。该批次代码复制到独立`frozen-code`并固定PYTHONPATH，运行中不改提示或执行器。

执行顺序：A4教师操作链600条 → A0直接回答同600条 → 质量审核与判分 → FeTaQA审核材料 → B2/B5开发训练（rank32、3epoch、seed42）→ 源验证集每个已覆盖任务20条推理。

该主批次已完成，状态为`development_complete_freeform_review_still_pending`。旧B2/B5共同源池149条，但replacement平衡实际覆盖146/132条，不能称严格等样本或等token；已增加coverage策略并另目录排队重跑。FeTaQA未人工批准前不纳入其操作链训练，不表示三任务正式实验结束。

状态来源：

- `runs/medium-development/stage.txt`：阶段；异常为failed_or_interrupted或interrupted。
- `runs/medium-development/progress.json`：完成数、进程存活及UTC时间。
- `runs/setup/medium-stage.log`：批次实际日志。
- `runs/setup/medium-progress.log`：只读进度记录，不发送外部通知。

应结合阶段、进程和实际输出检查状态，不能只看后台PID。`predict --resume`要求同配置、同输入与同冻结代码；改过协议后需新目录，不覆盖旧输出。

`serialization_followup.sh`的30条无损数组表示对照已完成，目录`runs/serialization-followup`，日志`runs/setup/serialization-followup.log`。失败7→4条，WikiTQ2/10→2/10、TabFact5/10→6/10；小样本不能证明总体质量保持或受控加速。

CPU后处理`medium_analysis_stage.sh`已完成官方算法重新判分、配对分析与200条FeTaQA双人审核材料；目录`runs/medium-analysis`，日志`runs/setup/medium-analysis.log`。自动批准0条。

后续开发控制按顺序单独加锁、冻结代码并运行：`student-answer-control`先补同池149条的未微调3B/B0答案监督对照，随后`coverage-student-control`以coverage策略重跑B2/B5、强制实际149条来源全部覆盖。分别查看同名目录stage.txt与`runs/setup`同名日志。前置失败或等待超12小时退出，避免双卡教师与学生冲突。它们使用相同40条源验证集，仍不等token、不构成正式论文结论。

以上两组控制现已完成，实际149条来源一致；coverage B2/B5的WikiTQ30%/35%、TabFact50%/65%，样本差值区间均跨0。新阶段为`interactive-teacher`：60条新源训练样本，32B整链/双操作交互对照；随后`optimized-student-control`单卡跑搜索压缩的固定分块控制。二者均独立冻结并加锁，日志在`runs/setup`同名文件，均为开发实验。

CPU候选审计`runs/optimization-audit`已完成：操作549→435，全部149条签名一致；状态成本分块与忽略状态的目标相同，未据此重复训练或宣称自适应收益。代码、公式与实验限制见`optimization-candidate.md`。

## 已完成的开发试跑

|目录/版本|结果与解释|
|---|---|
|gpu-smoke-initial、gpu-smoke|真实合成LoRA训练和重载通过；重复操作导致预算超限，问答未成功|
|real-data-pilot|8条真实训练样本的3B直接回答，只验证接口|
|supervised-pilot-v3|24条真实答案监督样本，每任务8条，1epoch/6步；loss约4.956。9条验证输出可解析，样本不足以形成结论|
|teacher-pilot|初始6条32B操作链可执行但语义错误存在，不能把可执行等同答案正确|
|teacher-pilot-fresh30|初版每任务10条；20/30可执行，7条accepted_weak、7条review_required、16条rejected；WTQ官方算法1/10、TabFact6/10|
|teacher-direct-fresh30|同30条32B A0：WTQ5/10、TabFact10/10|
|teacher-interface-v5-fresh30|同30条v5复测：26/30可执行，10条accepted_weak、10条review_required、10条rejected；WTQ4/10、TabFact6/10|
|paired-distillation-fair-state|B2/B5同7条WTQ/TabFact弱筛选样本的训练与验证完成；目标token不同、单seed、小样本，不宣称质量或效率优势|
|analysis/teacher-direct-vs-chain|上述30条A0/v5-A4的官方分数、失败和配对统计；属于已参与调试的开发样本|
|analysis/serialization-audit|固定600条的无损表示离线token测量；不包含模型回答质量结论|
|reviews/interface-v5-fresh30|10条FeTaQA静态审核页面、完整packet及两份空白独立模板；自动批准0条|
|medium-development|600条A4/A0均完成；WikiTQ26.0%/53.5%、TabFact52.0%/76.5%；旧149源池B2/B5训练和40条验证完成，抽样覆盖差异已单列|
|medium-analysis|完整判分、修复诊断、预算覆盖、压缩/边界审计与PNG/PDF/SVG开发图表完成|

旧失败结果和旧版本目录均保留。早期带膨胀状态历史的B2工程试跑不能作为现行强基线。早期Trainer尾梯度累积导致epoch=0.91，已显式对齐完整批次并记录padding和实际目标token；后续完成完整epoch。

现行操作提示版本tableqa-v5，状态断言、末行选择、比较表达式及JSON停止语义见`operations.md`。这里不把小样本接口修复改称正式方法收益。

## 数据与评测

|任务|规范化train|validation|test|
|---|---:|---:|---:|
|WikiTQ|11321|2810|4344|
|FeTaQA|7326|1001|2003|
|TabFact|92283|12792|12779|

WikiTQ开发采用官方random-split-1-train/dev，测试为pristine-unseen-tables，不是full-training实验。联合训练池对TabFact固定seed42取10000条训练源样本，再清除跨集合相同表格，联合清理后28403/16536/19126，去除311条。仅匹配表头/值及行列重排，不能保证识别全部语义近似表格。原始官方规范化文件保留不变。

WikiTQ使用原作者v1.0.2算法的Python3语法端口，保留原始/转换源码哈希、CoreNLP targetCanon及逐样本分数。新分析器要求分数绑定样本/预测哈希。主批次旧包装器已在medium-analysis另目录重新判分与汇总。FeTaQA自动指标配置和人工事实审核仍待完成，原始层次表头/日期等复杂案例还需复核。

FeTaQA操作监督现要求通过`human_review.py`合并双人审核及哈希批准；不能仅手填批准布尔值。测试审核用途不能生成训练批准样本。完整规则与命令见`analysis-protocol.md`。

## 部署历史与恢复依据

2026-11-06校内网/VPN恢复后，先读取已有完成标记与审计，没有重启或重复旧任务。runs/curriculum-student-v1已完成全部192条评测及收尾，R/M新权重与旧代码快照均保留；M因真实WikiTQ回归未采用。runs/fetaqa-transfer-v1的12题/四配置自动评测已完成并复算。新增runs/fetaqa-answer-supervision-v1/v2使用独立快照与新LoRA、512条官方FeTaQA训练答案；v2修复梯度累积归一化。初版SD漏载适配器的评测无效，实际加载重评结果在各自verified-evaluation。固定v2权重的额外100题开发复查也已完成，见runs/fetaqa-answer-development100-v1；所有结果不更改人工批准门槛。

主工程已在runs/main-update-fetaqa-20261106/before.tar.gz备份后同步，source.tar.gz保存同步源码，deployment.json记录源码哈希及184项全通过的测试。既有实验快照均未被改写。

批次脚本必须显式指向自己的冻结PYTHONPATH，并从脚本文件路径启动；用python -或python -m时，当前目录可能优先于PYTHONPATH。对旧快照做复算应先cd到其code目录或检查实际导入的tableqa.__file__，避免误导入服务器主工程旧模块。所有实际训练/推理脚本均以独立文件启动，未受后处理交互命令目录优先级影响。

最初因服务器无项目环境，先核验主机并创建独立venv，未升级驱动或覆盖现有环境。HF网络不可用时，使用ModelScope官方Qwen仓库；下载记录逐文件revision及SHA256，不执行仓库中的远程代码。32B第3分片曾遇到curl35并停止，后保留已校验文件、使用IPv4/HTTP1.1分段重试完成，不能把旧下载PID当作持续工作。

相关历史日志保存在`runs/setup`：bootstrap-mirror、download-3b-resume、teacher-stage和teacher-stage-resume等。早期安装脚本的默认cu121候选不等于本机最终cu124实测环境；恢复时以实际环境锁和GPU加载检查为准，避免重复安装、重复下载或启动冲突的教师任务。

正式主实验、七类扩展实验、三seed、人工主题标注、强方法文献/检查点核验及1.5B/7B验证仍未完成，继续按研究协议验收，不提前写正向论文结论。
