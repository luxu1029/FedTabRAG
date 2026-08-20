# 下一阶段问答端实验计划

## 冻结输入与目标

- 固定QA-100只用于既有结果审计和错误解释，不再用于调参。
- 已导出468个“方法×query”的gold Top-5命中但数值答案错误案例，覆盖72个query；其中Flat优先案例47条。
- 当前所有10种方法的program_predicted均为false；Program Accuracy与Execution Accuracy保持NA，直至新实验真实产生并执行program。

## 优先级

1. **数值错误审计**：先人工复核Flat的47条案例，区分操作数缺失、符号/百分比缩放、错误运算、答案格式和Gold异常。不得使用QA-100标签选择模型参数。
2. **dev-only证据重排序**：在FinQA dev的Flat Top-k内开发轻量reranker，特征限定为query-evidence语义、公司/年份一致性、数值覆盖和表头—数值邻接；冻结后test只运行一次。
3. **数值操作数抽取**：在dev上评估operand precision/recall、单位与百分比归一化，再评估conditional numerical EM（gold evidence已命中条件下）。
4. **program generation**：定义受限算子集合与可执行语法；报告Program Exact Match和Execution Accuracy前，必须真实输出program并通过解析/执行审计。当前历史值继续为NA。
5. **停止/继续门槛**：只有dev上的conditional numerical EM、operand recall或重排序Recall有预注册且稳定提升，才冻结一次test方案；否则记录负结果，不回到结构联邦超参搜索。

## 推荐评测与数据隔离

- 开发集：FinQA dev；所有阈值、reranker权重、prompt与程序语法只在dev选择。
- 正式测试：冻结配置后一次；固定QA-100只能作为既有测试子集报告，不重复选参。
- 主要指标：Retrieval Recall@5、conditional numerical EM、operand precision/recall、answer-format validity。
- Program指标：在确有program输出后报告Program Exact Match、parse success和Execution Accuracy；缺失继续写NA。
- 公平性与复现：保存输入/输出SHA256、命令、配置、日志、失败样本和strict missing，不插值、不补造。

## 当前案例分布

- 各方法gold-hit但错误数：{"fedavg_queue": 56, "flat": 47, "gt_zero_shot": 46, "hierarchy": 21, "hierarchy_hard": 39, "hierarchy_hard_kd": 39, "r_only": 56, "r_plus_u": 57, "u_only": 57, "unitable": 50}。
- 错误类别总计：{"format_matching_failure": 179, "number_extraction_failure": 26, "operation_failure": 140, "structure_localization_failure": 123}。
