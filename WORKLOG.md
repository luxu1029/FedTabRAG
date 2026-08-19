# FedTabRAG 工作日志

> 更新时间：2026-08-19（Asia/Shanghai）  
> 信息来源：仓库内现有代码、配置、日志、冻结结果、SHA256 清单，以及两份实验安排文档。  
> 状态说明：本目录当前不是 Git 仓库，无法通过 commit/branch 还原历史；下述结论只记录有落盘产物支持的事实。

## 1. 当前结论

- 10 天服务器实验（Day 1–10）已执行并冻结，最后一批主实验产物生成于 2026-08-03。
- 主结果中，`FedE4RAG-Flat` 仍是最强检索基线；`FedE4RAG-UniTable` 和 `FedTabRAG-Full(R+U)` 未超过 Flat。
- 三个 seed（42、7、2026）没有改变排序。Full(R+U) 方差最小，但均值仍低于 Flat。
- UniTable 的结构识别质量较高（S-TEDS 0.922466），但没有转化为检索收益。后续分桶结果显示，高 S-TEDS 桶仍落后 Flat，更支持“结构序列化/编码链路问题”，而非单纯的结构识别精度不足。
- Day 8 的 R/U 聚合消融未通过门槛，因此当时的决定是暂停昂贵训练，先诊断 row candidate、多正例权重和客户端采样偏差。
- 下一阶段诊断任务 1、2 已完成；任务 3 已导出 90 个案例，但人工标注和总结尚未完成；任务 4–9 尚无对应实现或结果目录。
- 目前最合理的恢复点：先完成人工案例标注，再执行冻结 BGE 的 dev-only 门控与分数融合，不应直接扩大联邦训练预算。

## 2. 主实验结果

主表来源：`outputs/day10/tables/main_results.csv`。

| 方法 | Table NDCG@10 | Row NDCG@10 | Cell NDCG@10 | Cell Recall@5 | 加权全局 NDCG@10 | QA-100 数值 EM |
|---|---:|---:|---:|---:|---:|---:|
| FedE4RAG-Flat | 0.314504 | 0.442569 | 0.120965 | 0.435185 | **0.256154** | 0.07 |
| FedE4RAG-UniTable | 0.310078 | 0.329254 | 0.138244 | 0.493056 | 0.229914 | 0.07 |
| FedTabRAG-Full (R+U) | 0.341971 | 0.307879 | 0.131527 | 0.454861 | 0.226521 | 0.05 |

解释边界：Full 在 Table NDCG 上较高，UniTable 在 Cell 指标上有局部优势，但项目预注册的加权全局指标仍由 Flat 明显领先；不能据局部指标宣称总体方法优于 Flat。

### 多种子稳定性

来源：`outputs/day10/tables/supplementary_seeds.csv`。

| 方法 | seed 42 | seed 7 | seed 2026 | 三 seed 均值 ± 标准差 |
|---|---:|---:|---:|---:|
| FedE4RAG-Flat | 0.256154 | 0.255386 | 0.250019 | **0.253853 ± 0.002729** |
| FedE4RAG-UniTable | 0.229914 | 0.225312 | 0.229439 | 0.228222 ± 0.002067 |
| FedTabRAG-Full (R+U) | 0.226521 | 0.226989 | 0.226732 | 0.226747 ± 0.000191 |

### 结构质量、鲁棒性和成本

- UniTable test：S-TEDS `0.922466`、Valid HTML `1.0`、Cell-count exact rate `0.977332`。
- Full(R+U) 结构扰动下的加权 NDCG 绝对下降：5% `0.002179`、10% `0.003841`、20% `0.004595`、30% `0.008947`。
- 训练成本：Flat `1846.39s / 1436 MiB`；UniTable `1868.45s / 1415 MiB`；Full(R+U) `5737.81s / 14750 MiB`。三者 LoRA 参数和总通信字节相同。
- 冻结结果包：`outputs/day10/FedTabRAG_day10_frozen_results.tar.gz`；清单位于 `outputs/day10/sha256_manifest.json` 和 `outputs/day10/package_sha256.txt`。
- 冻结包不包含模型/checkpoint 权重和扰动语料副本；详见 `outputs/day10/freeze_metadata.json`。

## 3. 已完成的实验阶段

### Day 1–4：数据、结构解析、证据与零样本检索

- FinQA 数据审计、渲染和完整性检查已完成，审计位于 `outputs/audit/`。
- UniTable smoke 和全量结构预测/评测已完成，结构指标位于 `outputs/unitable/`。
- Flat、oracle structure、predicted structure 三套语料以及表/行/单元格 evidence map 已生成。
- 冻结 BGE 的三分支零样本检索已完成。Day 4 当时结论为 `NO_GO_PENDING_FLAT_ROW_GRANULARITY_CONTROL`：GT 和 UniTable 未显示稳定结构优势，且 Flat 与结构分支存在候选粒度不对等风险。
- 相关决策：`outputs/retrieval/go_no_go.json`。

### Day 5–7：联邦基线、层次损失、难负例与 KD

- 完成 5 客户端 company 原子划分，映射位于 `data/processed/federated/company_5clients.json`。
- 完成 Flat、UniTable 两个公平 FedAvg 基线，公平性审计通过，汇总位于 `outputs/federated/day5_summary.json`。
- 实现并运行 hierarchy、hierarchy+hard negative、hierarchy+hard+KD 消融。
- 相关实现：`src/federated/train_fedrag.py`、`train_fedtabrag.py`、`fedtabrag_losses.py`、`local_queue.py`。

### Day 7.5–8：本地队列与 R/U 聚合

- 完成本地候选队列、dev 逐轮选模、R-only、U-only、R+U 和恢复 smoke。
- seed 42 的所有候选均在 dev 第 4 轮冻结并各进行一次正式 test。
- 最好 test 加权 NDCG 为 U-only `0.226578`；相对 FedAvg 仅 `+0.000161`，未达到预注册门槛。
- `outputs/federated/day8_final_summary.json` 的最终门控为 `passed_day8_gate: false`。

### Day 9：100 题检索与问答消融

- 固定 seed 42 的 100 题样本，完成 10 种方法的检索和本地 Qwen2.5:7b 问答。
- 最佳 Retrieval Recall@5：U-only 与 R+U 均为 `0.61`；最佳 MRR@5：UniTable `0.372167`。
- 数值 EM 整体较低：Flat/UniTable `0.07`，GT `0.06`，聚合分支多为 `0.05`，层次训练分支为 `0.02`。
- 没有任何方法输出 program，因此 Program/Execution Accuracy 为 `NA`，不得补写假设值。
- 汇总：`outputs/day9/ablation_summary.json`；联合错误分析：`outputs/day9/joint_error_analysis.json`。

### Day 10：多种子、鲁棒性和结果冻结

- 补跑 seed 7 和 2026 的 Flat、UniTable、R+U。
- 完成 5/10/20/30% 结构扰动鲁棒性实验。
- 生成主表、补充种子表、消融表、效率表、结构表、收敛图和严格缺失审计。
- strict-missing 共记录 9 项缺失，不做插值或假设填充，见 `outputs/day10/strict_missing_audit.json`。

## 4. 下一阶段诊断进度

计划来源：`FedTabRAG_下一阶段实验工作安排.docx`。

| 任务 | 状态 | 现有产物 / 下一动作 |
|---|---|---|
| 1. Flat vs UniTable 逐 query 配对 | **完成** | dev 883、test 1147 条，无缺失/重复；`outputs/diagnosis/pairwise_*.csv` |
| 2. 结构质量与任务属性分桶 | **完成** | 6 类 dev/test 分桶及摘要；`outputs/diagnosis/buckets/` |
| 3. 三类人工案例各 30 条 | **部分完成** | 3 个 XLSX 已导出；90 条人工标签仍为空，尚缺 `manual_case_analysis_summary.md` |
| 4. 冻结 BGE 置信度门控 | **未开始** | 尚无 `evaluate_confidence_gating.py` 或 `outputs/diagnosis/gating/` |
| 5. Flat/UniTable 分数后融合 | **未开始** | 尚无 `evaluate_score_fusion.py` 或 `outputs/diagnosis/fusion/` |
| 6. dev-only 选择与冻结方案 | **未开始** | 必须等待任务 4、5；若无方案过门槛，应冻结 fail decision 并停止 test |
| 7. 冻结方案正式 test 一次 | **未开始/受门槛阻塞** | 只有任务 6 PASS 后才允许执行一次 |
| 8. 最小联邦验证 | **未开始/受门槛阻塞** | 只有冻结策略超过 Flat 后才允许启动 |
| 9. 停止结构分支并转问答端 | **待决策** | 若任务 4–8 失败，整理负结果并转数值推理、重排序或 program generation |

### 已完成诊断的关键事实

- dev 配对总体 mean delta NDCG（UniTable - Flat）：`-0.003463`。
- test 配对总体：`+0.000107`，只允许作描述性解释，不可用于调参。
- dev 高 S-TEDS 桶 q4：delta `-0.005826`；低 S-TEDS 桶 q1：`-0.003853`。
- cell-count mismatch 桶只有 11 条，delta `-0.005784`；exact 桶为 `-0.003434`。
- hybrid 问题存在小幅局部正值 `+0.000631`，样本证据不足以直接冻结门控规则。
- 摘要：`outputs/diagnosis/diagnosis_summary.md`。

## 5. 预注册门槛与实验纪律

冻结 BGE 下的 gating/fusion 只有同时满足以下条件，才允许进入一次正式 test 或新的联邦训练：

1. dev 加权 NDCG@10 相对冻结 Flat 至少 `+0.003`；
2. Row 或 Cell Recall@5 至少提升 `0.01`；
3. worst-client 相对 Flat 的下降不超过 `0.005`；
4. 至少两个 seed 方向一致，且收益不能只来自单个结构质量桶；
5. 阈值、融合权重和最终方案只能由 dev 冻结，test 不参与选择。

必须继续遵守：相同问题、候选集合、gold evidence 和 row-major 单元格文本多重集合；保留配置、命令、日志和输入/输出 SHA256；缺失项写 `NA`；不得重复使用 test 调参。

## 6. 推荐恢复顺序

1. 人工填写以下三个文件中的 `error_type`、`manual_note`、`whether_structure_helpful`：
   - `outputs/diagnosis/manual_cases_flat_win.xlsx`
   - `outputs/diagnosis/manual_cases_unitable_win.xlsx`
   - `outputs/diagnosis/manual_cases_both_fail.xlsx`
2. 基于人工标注撰写 `outputs/diagnosis/manual_case_analysis_summary.md`，完成任务 3。
3. 实现任务 4，只读取 dev，运行冻结 BGE 置信度门控网格。
4. 实现任务 5，只读取 dev，按 evidence ID 对齐后运行 z-score/min-max/rank-score 融合。
5. 执行任务 6，严格按预注册门槛生成 frozen config 或 fail decision。
6. 只有任务 6 PASS 才执行任务 7 的一次正式 test；只有任务 7 超过 Flat 才执行任务 8。
7. 若未过门槛，直接执行任务 9，停止扩大结构联邦预算并转向 QA 数值推理/证据重排序。

## 7. 环境、测试与复现入口

### 当前测试健康度

2026-08-19 验证：

```bash
env PYTHONPATH=. conda run -n fedrag-test pytest -q tests
```

结果：`22 passed, 2 warnings`。

直接运行 `conda run -n fedrag-test pytest -q` 会在收集阶段失败：本项目未配置根目录包路径，并会误收集 `third_party/` 中依赖 `flgo`/`llms` 的测试。后续应继续使用上面的限定命令，或另行补充 pytest 配置；这不代表项目自身 22 个测试失败。

### 常用只读汇总

```bash
sed -n '1,80p' outputs/day10/tables/main_results.csv
sed -n '1,80p' outputs/day10/tables/supplementary_seeds.csv
sed -n '1,240p' outputs/federated/day8_final_summary.json
sed -n '1,240p' outputs/day9/ablation_summary.json
sed -n '1,240p' outputs/diagnosis/diagnosis_summary.md
```

### 关键代码入口

- 数据与渲染：`src/data/`
- 表结构与证据：`src/table/`
- 索引与检索：`src/retrieval/`
- 联邦训练与评测：`src/federated/`
- 诊断与结果合并：`src/evaluation/`
- Day 8–10 编排与冻结：`scripts/`
- 实验配置：`configs/`

## 8. 已知风险与未决事项

- **无 Git 历史**：当前目录没有 `.git` 仓库，冻结元数据中的 `git_commit` 也是 `NA`。应在不破坏现有产物的前提下补建版本管理或保存外部快照。
- **任务 3 依赖人工判断**：Excel 中的人工列不能由脚本自动伪造；完成前不应宣称人工错误归因已通过。
- **候选粒度公平性**：Day 4 已指出 Flat pseudo-row 与结构化多 row candidates 的粒度差异，后续融合/门控必须显式说明候选对齐方法。
- **test 污染风险**：test 已用于既有主实验的一次性正式评测和任务 1–3 的描述/人工解释；任务 4–6 的参数搜索必须完全隔离 test。
- **QA 瓶颈明显**：100 题数值 EM 最高仅 0.07，且无 program 输出。如果结构门控失败，转向 QA 端比继续堆叠结构训练更符合当前证据。

## 9. 新会话恢复提示词

```text
请先读取 WORKLOG.md、FedTabRAG_下一阶段实验工作安排.docx、
outputs/day10/tables/main_results.csv、outputs/federated/day8_final_summary.json
和 outputs/diagnosis/diagnosis_summary.md，再检查当前文件状态。
严格遵守 dev-only 选模、test 一次正式评测、strict missing 和 SHA256 冻结规则。
从 WORKLOG.md 的“推荐恢复顺序”继续，不要重复已完成的 Day 1–10 或诊断任务 1–2。
```
