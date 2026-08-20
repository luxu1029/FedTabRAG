# Task 6 dev-only 结构策略选择与冻结报告

## 协议

- 选择只读取任务4/5的dev网格、最佳配置和SHA256清单；未读取test、未训练、未重新编码。
- 所有候选统一对比canonical Flat dev：weighted NDCG@10=0.145597802，Row Recall@5=0.327007299，Cell Recall@5=0.236496350，worst-client=0.099247224。
- 门槛：Δweighted NDCG≥0.003；ΔRow或ΔCell Recall@5≥0.01；Δworst-client≥-0.005。

## 候选审查

- none：不提升，未过门槛。
- gating最佳 g008：weighted NDCG=0.145597802，Δ=+0.000000000，未过门槛。
- fusion最高 f016：池内weighted NDCG=0.115483709；相对canonical Flat Δ=-0.030114093，且Top-10 union口径与全语料Flat不完全可比，未过门槛。
- gating_plus_fusion：没有独立dev网格；不伪造组合指标，记为不可评估/未通过。

## 冻结决定

FAIL_NO_DEV_STRATEGY：不得进入任务7正式test。

未生成可供任务7读取的frozen_config.json。已生成fail_decision.json并冻结失败原因。后续不得运行正式test；按工作安排应停止本轮结构策略test放行，并进入负结果整理/问答端方向。
