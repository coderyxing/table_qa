# 可执行操作接口与语义

当前提示版本为`tableqa-v6`。这是接口定义，不是模型已正确掌握这些操作的声明。所有生成参数都经过白名单执行器处理，不运行任意Python/SQL。

## 状态与引用

原表列为c0、c1等，行ID为r0、r1等；重复显示表头仍有独立列ID。排序和筛选保留行ID，分组结果使用g0等派生ID，并保存原单元格来源。当前状态只展示仍可访问列的原表头。原表不可变，回滚使用快照；聚合和删除本身不宣称可逆。

## 表格操作

|操作|必要字段|执行语义|
|---|---|---|
|project|columns|按给定顺序保留列；拒绝删除计划后续仍读取的列|
|filter|column、cmp、value、type|eq/ne/gt/ge/lt/le、contains、is_null、is_number、in/not_in|
|sort|column、type、descending|稳定排序，number或text；不猜测日期格式|
|select|mode、n或column|first/last按当前行序取前/后N行；extreme按数值列取min/max，保留全部并列|
|group_aggregate|by、aggregations|count统计行数（包含空值行），sum/mean/min/max需严格数值；不是SQL COUNT(col)的非空计数语义|
|derive|as、expression|创建新列，禁止覆盖已有列|
|answer|columns、value或assert|终止执行；不得在其后继续操作|

数值允许千位逗号和百分号；百分号转换为比例。NaN、Inf、模糊文本及空值不会悄悄变成0。`is_number`是模型显式选择的筛选操作，不能把它删除的值默认为无关证据。in/not_in要求列表，按指定数值或文本类型匹配。

数值筛选默认对无法解析的单元格报错。筛选计划可显式设置 `invalid_number: "skip"`，将无法解析的单元格视为不匹配，适用于数值比较与 in/not_in。比较常量及候选列表仍须是合法数值。该选项随操作保存在轨迹中，以便审核被排除的行。

重试使用同一表状态和有长度上限的错误反馈，生成新请求时替换前次反馈。实际失败输出保留在预测 responses 中，修复提示省去完整失败输出，以减少上下文开销。输入仍由模型后端按 token 上限检查。

FeTaQA原始多层表头保留在输入矩阵，当前适配器没有未经审核的自动合并规则。部分第二层表头可能仍作为数据行展示，处理它们必须是明确、可追踪的操作。这属于当前覆盖限制。

## 安全表达式

叶节点为`{"col":"c1"}`或`{"literal":5}`，也可使用原始常量。二元算术add/sub/mul/div；二元比较eq/ne/gt/ge/lt/le，type为number/text；比较输出布尔值。and/or要求两个布尔值；if要求布尔条件和两个分支，只执行被选中的分支。来源依赖保守地包括所有分支引用的列。

```json
{"op":"derive","as":"difference","expression":{"op":"sub","args":[{"col":"c1"},{"col":"c2"}]}}
```

## 事实验证必须基于执行状态

操作链模式中不允许直接给出常量标签`answer.value=1`。下例在筛选后检查是否还有支持行；空结果输出0。

```json
{"op":"answer","assert":{"kind":"row_count","cmp":"gt","value":0}}
```

单行聚合值比较：

```json
{"op":"answer","assert":{"kind":"cell","column":"total","cmp":"eq","value":1,"type":"number"}}
```

单行双列比较：

```json
{"op":"answer","assert":{"kind":"expression","expression":{"op":"gt","args":[{"col":"c1"},{"col":"c2"}],"type":"number"}}}
```

这些机制只证明断言与执行状态一致，不能证明模型选对了支持列、正确理解了原问题。自然语言答案依然需要事实审阅，所有结果保留`semantic_verification=not_proven`。

## 生成与审核

本地模型在第一个完整JSON对象后停止。若某个token同时含有多余字符，解析文本截到对象结束位置，原始文本及`json_stop_overhang`另存；token成本按实际生成计数。API/回放输入仍使用严格JSON解析，不能借此运行表达式字符串或任意代码。

`audit-trajectories`会重放程序并核对答案和证据。短答案禁止使用不经执行的常量答案，事实验证必须包含assert。参考答案匹配只构成弱监督，不证明推理过程正确。FeTaQA没有人工批准就不会自动进入操作链训练。

## 比较组的状态表示

- B2：完整操作链、逐步调用，使用当前状态与操作历史。
- B3：压缩链、逐步调用，状态接口与B2一致。
- B4：压缩分块、完整中间状态历史，作为历史表示消融。
- B5：压缩分块、当前状态与操作历史。
- B5_no_blocks等价于B3；匹配种子/预算时复用结果，不将其重复计为独立实验。

初始状态不重复附带原始表格。此前携带完整状态历史的B2工程试跑保留原目录，但不作为强基线的正式结果。

训练器会把记录数显式对齐到完整梯度累积批次，并记录padding及实际目标token。token预算在对齐后仍必须满足上限。仅支持单GPU训练，使用CUDA_VISIBLE_DEVICES选择一张卡。
