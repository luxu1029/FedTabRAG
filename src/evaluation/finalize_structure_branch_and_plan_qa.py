#!/usr/bin/env python3
"""Freeze the structure-branch negative result and export QA-next cases."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import re
import statistics
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


LOG = logging.getLogger("finalize_structure_branch_and_plan_qa")
NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
CASE_FIELDS = (
    "priority", "method", "variant", "query_id", "sample_id", "company",
    "program_type", "table_size_bin", "structure_quality_bin", "s_teds",
    "question", "gold_answer", "prediction", "parsed_gold", "parsed_prediction",
    "error_category", "gold_hit", "first_gold_rank", "gold_evidence_ids",
    "retrieved_evidence_ids", "retrieved_evidence", "gold_program",
    "program_predicted", "program_accuracy", "execution_accuracy",
    "recommended_track", "manual_review_note",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day9-root", type=Path, default=Path("outputs/day9"))
    parser.add_argument("--day10-root", type=Path, default=Path("outputs/day10"))
    parser.add_argument("--diagnosis-root", type=Path, default=Path("outputs/diagnosis"))
    parser.add_argument("--methods", type=Path, default=Path("configs/day9_methods.json"))
    parser.add_argument("--finqa-test", type=Path, default=Path("data/raw/finqa/test.json"))
    parser.add_argument("--evidence-map", type=Path,
                        default=Path("data/processed/evidence_map.jsonl"))
    parser.add_argument("--qa-output-dir", type=Path, default=Path("outputs/qa_next"))
    parser.add_argument("--log-file", type=Path,
                        default=Path("outputs/logs/diagnosis_task9.log"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique(rows: list[dict[str, Any]], key: str, source: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        value = str(row[key])
        if value in result:
            raise ValueError(f"Duplicate {key}={value} in {source}")
        result[value] = row
    return result


def classify(retrieval: dict[str, Any], qa: dict[str, Any], positive_by_level: dict[str, Any],
             program: str) -> str:
    if qa["numerical_em"]:
        return "success"
    if not retrieval["gold_hit"]:
        return "retrieval_failure"
    retrieved_ids = {row["evidence_id"] for row in retrieval["retrieved"]}
    fine_positive = set(positive_by_level.get("row", [])) | set(
        positive_by_level.get("cell", []))
    if fine_positive and not (retrieved_ids & fine_positive):
        return "structure_localization_failure"
    if qa["parsed_prediction"] is None:
        return "format_matching_failure"
    operands = NUMBER.findall(program)
    context = " ".join(row["text"] for row in retrieval["retrieved"])
    normalized_context = context.replace(",", "")
    if operands and not all(token.replace(",", "") in normalized_context for token in operands):
        return "number_extraction_failure"
    return "operation_failure"


def recommended_track(category: str) -> str:
    return {
        "structure_localization_failure": "evidence_reranking",
        "number_extraction_failure": "numeric_operand_extraction",
        "operation_failure": "program_generation_or_numeric_reasoning",
        "format_matching_failure": "answer_format_and_numeric_parser",
    }.get(category, "manual_review")


def scalar(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def excel_col(number: int) -> str:
    result = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result
    return result


def clean_xml(value: str) -> str:
    return "".join(char for char in value if char in "\t\n\r" or ord(char) >= 32)[:32767]


def write_xlsx(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    """Write one standards-compliant XLSX sheet using only the Python stdlib."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    all_rows = [dict(zip(CASE_FIELDS, CASE_FIELDS)), *rows]
    xml_rows = []
    for row_number, row in enumerate(all_rows, 1):
        cells = []
        style = ' s="1"' if row_number == 1 else ' s="2"'
        for column_number, field in enumerate(CASE_FIELDS, 1):
            reference = f"{excel_col(column_number)}{row_number}"
            value = escape(clean_xml(scalar(row.get(field, ""))))
            cells.append(f'<c r="{reference}" t="inlineStr"{style}><is><t xml:space="preserve">{value}</t></is></c>')
        xml_rows.append(f'<row r="{row_number}">{"".join(cells)}</row>')
    last = f"{excel_col(len(CASE_FIELDS))}{len(all_rows)}"
    widths = "".join(
        f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>'
        for index, width in enumerate(
            (12, 18, 20, 42, 34, 12, 16, 14, 18, 12, 55, 16, 16, 22, 22,
             30, 10, 14, 55, 55, 80, 45, 16, 18, 18, 36, 35), 1)
    )
    sheet = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<dimension ref="A1:{last}"/><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>
<cols>{widths}</cols><sheetData>{''.join(xml_rows)}</sheetData><autoFilter ref="A1:{excel_col(len(CASE_FIELDS))}{len(all_rows)}"/></worksheet>'''
    files = {
        "[Content_Types].xml": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/></Types>''',
        "_rels/.rels": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>''',
        "xl/workbook.xml": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="qa_numeric_error_cases" sheetId="1" r:id="rId1"/></sheets></workbook>''',
        "xl/_rels/workbook.xml.rels": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>''',
        "xl/styles.xml": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/><color rgb="FFFFFFFF"/></font></fonts><fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill></fills><borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>''',
        "xl/worksheets/sheet1.xml": sheet,
    }
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    temporary.replace(path)


def atomic_text(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(args.log_file, mode="a", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter); stream_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])

    fail_path = args.diagnosis_root / "frozen_selection/fail_decision.json"
    failure = load_json(fail_path)
    if failure.get("decision") != "FAIL_NO_DEV_STRATEGY" or failure.get(
            "formal_test_authorized") is not False:
        raise ValueError("Task 9 trigger condition is not frozen")
    if (args.diagnosis_root / "frozen_selection/frozen_config.json").exists():
        raise ValueError("A frozen pass config exists; refusing Task 9 stop decision")
    LOG.info("Task 9 trigger verified: %s", failure["decision"])

    registry = load_json(args.methods)
    if args.seed != int(registry["protocol"]["seed"]):
        raise ValueError("Seed differs from frozen Day-9 protocol")
    sample = load_json(args.day9_root / "sample_ids.json")
    frozen_ids = [row["query_id"] for row in sample["samples"]]
    sample_meta = {row["query_id"]: row for row in sample["samples"]}
    if len(frozen_ids) != 100 or len(set(frozen_ids)) != 100:
        raise ValueError("Frozen QA sample must contain 100 unique queries")
    finqa_rows = load_json(args.finqa_test)
    finqa = {f"test:{item['id']}:{index}": item for index, item in enumerate(finqa_rows)}
    evidence_maps = unique([row for row in jsonl(args.evidence_map)
                            if row.get("split") == "test" and row["query_id"] in set(frozen_ids)],
                           "query_id", args.evidence_map)
    if set(evidence_maps) != set(frozen_ids):
        raise ValueError("Evidence-map coverage differs from frozen QA sample")

    cases = []
    per_method = {}
    category_counts = Counter()
    all_program_flags = []
    for method, spec in registry["methods"].items():
        retrieval_path = args.day9_root / "retrieval" / f"{method}.jsonl"
        qa_path = args.day9_root / "qa" / f"{method}.jsonl"
        retrieval = unique(jsonl(retrieval_path), "query_id", retrieval_path)
        qa = unique(jsonl(qa_path), "query_id", qa_path)
        if set(retrieval) != set(frozen_ids) or set(qa) != set(frozen_ids):
            raise ValueError(f"{method}: frozen query coverage mismatch")
        selected = 0
        for query_id in frozen_ids:
            qrow, rrow = qa[query_id], retrieval[query_id]
            if bool(qrow["gold_hit"]) != bool(rrow["gold_hit"]):
                raise ValueError(f"{method}/{query_id}: gold-hit mismatch")
            all_program_flags.append(bool(qrow["program_predicted"]))
            if not rrow["gold_hit"] or qrow["numerical_em"]:
                continue
            item = finqa[query_id]
            qa_gold = item["qa"]
            positive = evidence_maps[query_id]["variants"][spec["variant"]][
                "positive_evidence_ids"]
            category = classify(rrow, qrow, positive, str(qa_gold.get("program") or ""))
            if category in ("success", "retrieval_failure"):
                raise ValueError(f"Invalid selected category {method}/{query_id}: {category}")
            metadata = sample_meta[query_id]
            retrieved_compact = [
                {"rank": row["rank"], "evidence_id": row["evidence_id"],
                 "level": row["level"], "score": row["score"], "text": row["text"]}
                for row in rrow["retrieved"]
            ]
            cases.append({
                "priority": "flat_primary" if method == "flat" else "cross_method",
                "method": method, "variant": spec["variant"], "query_id": query_id,
                "sample_id": metadata["sample_id"],
                "company": metadata["sample_id"].split("/", 1)[0],
                "program_type": metadata["program_type"],
                "table_size_bin": metadata["table_size_bin"],
                "structure_quality_bin": metadata["structure_quality_bin"],
                "s_teds": metadata["s_teds"], "question": qrow["question"],
                "gold_answer": qrow["gold_answer"], "prediction": qrow["prediction"],
                "parsed_gold": qrow["parsed_gold"],
                "parsed_prediction": qrow["parsed_prediction"],
                "error_category": category, "gold_hit": True,
                "first_gold_rank": rrow["first_gold_rank"],
                "gold_evidence_ids": rrow["gold_evidence_ids"],
                "retrieved_evidence_ids": qrow["retrieved_evidence_ids"],
                "retrieved_evidence": retrieved_compact,
                "gold_program": qa_gold.get("program") or "NA",
                "program_predicted": False, "program_accuracy": "NA",
                "execution_accuracy": "NA", "recommended_track": recommended_track(category),
                "manual_review_note": "",
            })
            selected += 1; category_counts[category] += 1
        per_method[method] = selected
    if any(all_program_flags):
        raise ValueError("At least one program was predicted; Program Accuracy may not remain NA")
    if len(cases) != 468 or per_method.get("flat") != 47:
        raise ValueError(f"Unexpected QA-next selection count: total={len(cases)} flat={per_method.get('flat')}")
    cases.sort(key=lambda row: (row["priority"] != "flat_primary", row["method"], row["query_id"]))
    unique_queries = len({row["query_id"] for row in cases})

    main = {row["method"]: row for row in read_csv(args.day10_root / "tables/main_results.csv")}
    seeds = read_csv(args.day10_root / "tables/supplementary_seeds.csv")
    seed_summary = {row["method"]: row for row in seeds if row["row_type"] == "summary"}
    structure = next(row for row in read_csv(args.day10_root / "tables/structure.csv")
                     if row["method"] == "UniTable")
    robustness = read_csv(args.day10_root / "tables/robustness.csv")
    gating = load_json(args.diagnosis_root / "gating/gating_dev_best_config.json")
    fusion = load_json(args.diagnosis_root / "fusion/fusion_dev_best_config.json")
    ablation = load_json(args.day9_root / "ablation_summary.json")

    report_path = args.diagnosis_root / "stop_structure_branch_report.md"
    negative_path = args.diagnosis_root / "negative_result_summary.csv"
    xlsx_path = args.qa_output_dir / "qa_numeric_error_cases.xlsx"
    plan_path = args.qa_output_dir / "next_stage_qa_plan.md"
    manifest_path = args.qa_output_dir / "task9_sha256_manifest.json"
    flat, unitable, full = (main["FedE4RAG-Flat"], main["FedE4RAG-UniTable"],
                            main["FedTabRAG-Full(R+U)"])
    report = f"""# 停止当前结构联邦分支报告

## 冻结决定

任务6已冻结为 `FAIL_NO_DEV_STRATEGY`，通过候选数为0，`formal_test_authorized=false`，且未生成 `frozen_config.json`。因此跳过任务7正式test和任务8最小联邦验证。从本报告起，不再继续扩大当前结构联邦训练预算，不再试探学习率、训练轮数、候选队列大小、损失权重或聚合策略。

## 证据汇总

- 主实验：Flat weighted global NDCG@10={float(flat['weighted_global_ndcg@10']):.6f}；UniTable={float(unitable['weighted_global_ndcg@10']):.6f}；Full(R+U)={float(full['weighted_global_ndcg@10']):.6f}。Flat保持最强总体检索结果。
- 三seed均值±标准差：Flat {float(seed_summary['FedE4RAG-Flat']['mean_across_seeds']):.6f}±{float(seed_summary['FedE4RAG-Flat']['std_across_seeds']):.6f}；UniTable {float(seed_summary['FedE4RAG-UniTable']['mean_across_seeds']):.6f}±{float(seed_summary['FedE4RAG-UniTable']['std_across_seeds']):.6f}；Full(R+U) {float(seed_summary['FedTabRAG-Full(R+U)']['mean_across_seeds']):.6f}±{float(seed_summary['FedTabRAG-Full(R+U)']['std_across_seeds']):.6f}。排序未随seed改变。
- 结构质量：UniTable S-TEDS={float(structure['s_teds']):.6f}，Valid HTML={float(structure['valid_html_rate']):.6f}，cell-count exact={float(structure['cell_count_exact_rate']):.6f}；高结构质量没有转化为总体检索收益。
- 结构扰动：5/10/20/30%下weighted NDCG绝对下降分别为 {', '.join(f"{float(row['absolute_drop_vs_0']):.6f}" for row in robustness if row['perturbation_percent'] != '0')}。模型使用了结构信号，但当前信号有效性不足。
- 任务4门控：最佳gating相对canonical Flat Δweighted NDCG={float(gating['metrics']['delta_weighted_ndcg_vs_flat']):+.6f}，Row/Cell Recall@5变化均为0，未过门槛。
- 任务5融合：Top-10 union中最高配置{fusion['config']['config_id']}虽有池内增益，但Row/Cell Recall下降；唯一池内数值候选也因表示池口径不可比和统一canonical Flat审查失败，不能放行test。
- 任务6统一审查：none、gating、fusion、gating_plus_fusion通过数均为0，最终 `FAIL_NO_DEV_STRATEGY`。
- QA-100：Flat/UniTable numerical EM均为0.07，Full(R+U)为0.05；所有方法均未输出program，Program Accuracy与Execution Accuracy继续为NA。

## 停止原因与表述边界

当前证据支持：在本项目的受控渲染FinQA、oracle cell text保持不变、冻结BGE及既定联邦预算条件下，UniTable结构序列化和当前FedTabRAG结构训练/聚合方案没有超过Flat强基线。不得外推为真实PDF端到端表格理解普遍无效，也不得宣称结构预测质量低；相反，现有结果更指向表示、候选粒度、evidence ID映射及下游数值推理瓶颈。

本分支作为可复现负结果冻结。后续工作转向：Flat Top-k内轻量证据重排序、数值操作数抽取、显式program generation和执行验证。不得用现有test或固定QA-100重新调参。
"""
    atomic_text(report_path, report, args.overwrite)

    negative_rows = [
        {"scope": "main", "method": name, "metric": "weighted_global_ndcg@10",
         "value": row["weighted_global_ndcg@10"], "decision_use": "Flat strongest"}
        for name, row in (("FedE4RAG-Flat", flat), ("FedE4RAG-UniTable", unitable),
                          ("FedTabRAG-Full(R+U)", full))
    ] + [
        {"scope": "three_seed", "method": name, "metric": "mean_weighted_ndcg@10",
         "value": row["mean_across_seeds"], "decision_use": "ranking stable across seeds"}
        for name, row in seed_summary.items()
    ] + [
        {"scope": "task6", "method": "structure_strategy", "metric": "passing_candidates",
         "value": "0", "decision_use": "FAIL_NO_DEV_STRATEGY"},
        {"scope": "qa100", "method": "all_methods", "metric": "program_accuracy",
         "value": "NA", "decision_use": "no method predicted a program"},
        {"scope": "qa_next", "method": "all_methods", "metric": "gold_hit_wrong_pairs",
         "value": str(len(cases)), "decision_use": f"{unique_queries} unique queries"},
    ]
    atomic_csv(negative_path, negative_rows, args.overwrite)
    write_xlsx(xlsx_path, cases, args.overwrite)

    plan = f"""# 下一阶段问答端实验计划

## 冻结输入与目标

- 固定QA-100只用于既有结果审计和错误解释，不再用于调参。
- 已导出{len(cases)}个“方法×query”的gold Top-5命中但数值答案错误案例，覆盖{unique_queries}个query；其中Flat优先案例{per_method['flat']}条。
- 当前所有10种方法的program_predicted均为false；Program Accuracy与Execution Accuracy保持NA，直至新实验真实产生并执行program。

## 优先级

1. **数值错误审计**：先人工复核Flat的{per_method['flat']}条案例，区分操作数缺失、符号/百分比缩放、错误运算、答案格式和Gold异常。不得使用QA-100标签选择模型参数。
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

- 各方法gold-hit但错误数：{json.dumps(per_method, ensure_ascii=False, sort_keys=True)}。
- 错误类别总计：{json.dumps(dict(sorted(category_counts.items())), ensure_ascii=False)}。
"""
    atomic_text(plan_path, plan, args.overwrite)

    inputs = [fail_path, args.day10_root / "tables/main_results.csv",
              args.day10_root / "tables/supplementary_seeds.csv",
              args.day10_root / "tables/structure.csv",
              args.day10_root / "tables/robustness.csv",
              args.diagnosis_root / "gating/gating_dev_best_config.json",
              args.diagnosis_root / "fusion/fusion_dev_best_config.json",
              args.day9_root / "sample_ids.json", args.day9_root / "ablation_summary.json",
              args.methods, args.finqa_test, args.evidence_map]
    inputs += [args.day9_root / subdir / f"{method}.jsonl"
               for method in registry["methods"] for subdir in ("retrieval", "qa")]
    outputs = [report_path, negative_path, xlsx_path, plan_path]
    manifest = {
        "schema_version": "1.0", "task": 9, "trigger": "FAIL_NO_DEV_STRATEGY",
        "qa_filter": "gold_hit == true AND numerical_em == false",
        "selected_method_query_pairs": len(cases), "selected_unique_queries": unique_queries,
        "flat_priority_cases": per_method["flat"], "program_accuracy": "NA",
        "execution_accuracy": "NA", "test_used_for_tuning": False,
        "inputs": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                   for path in inputs],
        "outputs": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                    for path in outputs],
    }
    atomic_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                args.overwrite)
    LOG.info("Task 9 complete pairs=%d unique_queries=%d flat=%d", len(cases), unique_queries,
             per_method["flat"])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        LOG.exception("Task 9 finalization failed")
        sys.exit(2)
