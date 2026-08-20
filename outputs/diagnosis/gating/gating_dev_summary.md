# Task 4 冻结 BGE 置信度门控（dev-only）

## 协议

- 仅使用 dev 的冻结 Flat/UniTable 精确检索结果；未训练、未重新编码、未读取 test。
- 门控对每个 query 统一选择整套 Flat 或 UniTable 三层结果。
- 综合可靠性 R = 0.40×S-TEDS + 0.25×cell-count consistency + 0.15×valid_html + 0.20×confidence。
- 层级权重：Table 0.2、Row 0.3、Cell 0.5；公平性使用冻结 5 客户端划分。

## 结果

- 网格配置数：40；dev query 数：883。
- Flat baseline weighted NDCG@10：0.145597802。
- 最佳门控：g008 / s_teds，UniTable query=2 (0.23%)。
- 最佳 weighted NDCG@10：0.145597802，相对 Flat +0.000000000。
- Row Recall@5 变化：+0.000000000；Cell Recall@5 变化：+0.000000000。
- worst-client weighted NDCG@10 变化：+0.000000000。
- 判定：未达到预注册门槛；不得据任务4结果运行test。

## 边界

本结果只用于 dev 方案比较。无论任务4单项结果如何，均不得直接进入正式 test；须继续完成任务5，并由任务6统一冻结或输出 fail decision。
