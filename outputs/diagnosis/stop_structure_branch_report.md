# 停止当前结构联邦分支报告

## 冻结决定

任务6已冻结为 `FAIL_NO_DEV_STRATEGY`，通过候选数为0，`formal_test_authorized=false`，且未生成 `frozen_config.json`。因此跳过任务7正式test和任务8最小联邦验证。从本报告起，不再继续扩大当前结构联邦训练预算，不再试探学习率、训练轮数、候选队列大小、损失权重或聚合策略。

## 证据汇总

- 主实验：Flat weighted global NDCG@10=0.256154；UniTable=0.229914；Full(R+U)=0.226521。Flat保持最强总体检索结果。
- 三seed均值±标准差：Flat 0.253853±0.002729；UniTable 0.228222±0.002067；Full(R+U) 0.226747±0.000191。排序未随seed改变。
- 结构质量：UniTable S-TEDS=0.922466，Valid HTML=1.000000，cell-count exact=0.977332；高结构质量没有转化为总体检索收益。
- 结构扰动：5/10/20/30%下weighted NDCG绝对下降分别为 0.002179, 0.003841, 0.004595, 0.008947。模型使用了结构信号，但当前信号有效性不足。
- 任务4门控：最佳gating相对canonical Flat Δweighted NDCG=+0.000000，Row/Cell Recall@5变化均为0，未过门槛。
- 任务5融合：Top-10 union中最高配置f016虽有池内增益，但Row/Cell Recall下降；唯一池内数值候选也因表示池口径不可比和统一canonical Flat审查失败，不能放行test。
- 任务6统一审查：none、gating、fusion、gating_plus_fusion通过数均为0，最终 `FAIL_NO_DEV_STRATEGY`。
- QA-100：Flat/UniTable numerical EM均为0.07，Full(R+U)为0.05；所有方法均未输出program，Program Accuracy与Execution Accuracy继续为NA。

## 停止原因与表述边界

当前证据支持：在本项目的受控渲染FinQA、oracle cell text保持不变、冻结BGE及既定联邦预算条件下，UniTable结构序列化和当前FedTabRAG结构训练/聚合方案没有超过Flat强基线。不得外推为真实PDF端到端表格理解普遍无效，也不得宣称结构预测质量低；相反，现有结果更指向表示、候选粒度、evidence ID映射及下游数值推理瓶颈。

本分支作为可复现负结果冻结。后续工作转向：Flat Top-k内轻量证据重排序、数值操作数抽取、显式program generation和执行验证。不得用现有test或固定QA-100重新调参。
