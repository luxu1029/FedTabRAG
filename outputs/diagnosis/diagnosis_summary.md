# Flat vs UniTable 分桶诊断摘要（任务2）

## 协议边界

- S-TEDS四分位边界只由dev计算，并原样应用到test。
- dev四分位边界：Q25=0.909091，Q50=0.933333，Q75=0.947368。
- 表格大小边界预注册为small≤20、medium≤50、large>50个单元格。
- test只用于描述性解释；本脚本不搜索、不选择、不冻结门控阈值或融合权重。

## 总体配对结果

- dev：n=883，mean delta_ndcg=-0.003463。
- test：n=1147，mean delta_ndcg=0.000107（仅解释）。

## 关键dev诊断

- q1_low：n=245，delta_ndcg=-0.003853，UniTable win rate=0.0571。
- q4_high：n=206，delta_ndcg=-0.005826，UniTable win rate=0.0922。
- 高S-TEDS桶仍低于Flat，支持优先检查结构序列化/编码问题。
- cell-count mismatch桶delta=-0.005784，exact桶delta=-0.003434。错位样本退化更明显，可作为后续门控诊断特征。

## 结论边界

本任务只定位差异来源，不据此选择阈值，也不允许test影响任务4-6的方案选择。门控是否有效必须由后续冻结BGE的dev网格实验单独验证。
