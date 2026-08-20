# Task 5 冻结 BGE Flat/UniTable 分数后融合（dev-only）

## 协议与边界

- 仅使用883条dev query的冻结Flat/UniTable Top-10候选及分数；未训练、未重新编码、未读取test。
- 按精确evidence_id对齐，候选池为两路Top-10并集；Table平均池大小10.0793，Row/Cell均为20.0000/20.0000。
- 支持z-score、min-max和rank_score；每种方式搜索alpha=0.0,0.1,...,1.0，共33项。
- 缺失候选分数置于该分支归一化最低分以下1单位。MRR/MAP是截断融合池诊断值，不冒充全语料精确指标。

## 结果

- Flat endpoint weighted NDCG@10：0.101138660。
- 原冻结全语料Flat参考值：0.145597802；因Row/Cell表示池翻倍，不与融合池绝对值直接比较。
- 最佳配置：f016，normalization=min-max，alpha=0.5。
- 最佳 weighted NDCG@10：0.115483709，相对Flat +0.014345050。
- Row Recall@5变化：-0.029197080；Cell Recall@5变化：-0.046715328。
- worst-client weighted NDCG@10变化：+0.004261190。
- 通过池内端点数值筛查的配置数：1/33；候选：f006（z-score，alpha=0.6）。判定：存在池内端点口径的数值候选；仍须任务6审查口径、稳定性并冻结。

## 决策边界

本结果只用于dev方案选择。必须由任务6结合任务4、稳定性和预注册条件统一冻结；当前禁止执行正式test。
