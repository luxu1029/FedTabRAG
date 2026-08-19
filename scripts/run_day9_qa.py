#!/usr/bin/env python3
"""Run resumable, deterministic Ollama QA for one frozen Day-9 retrieval method."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import math
import re
import time
import urllib.request
from pathlib import Path


NUMBER = re.compile(r"[-+]?\(?\$?[0-9][0-9,]*(?:\.[0-9]+)?\)?%?")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--methods", type=Path, default=Path("configs/day9_methods.json"))
    parser.add_argument("--retrieval-dir", type=Path, default=Path("outputs/day9/retrieval"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/day9/qa"))
    parser.add_argument("--prompt", type=Path, default=Path("configs/day9_qa_prompt.txt"))
    parser.add_argument("--finqa-test", type=Path, default=Path("data/raw/finqa/test.json"))
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/generate")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def parse_number(text: str):
    match = NUMBER.search(text.strip())
    if not match:
        return None
    token = match.group(0)
    percent = token.endswith("%")
    negative = token.startswith("(") and token.endswith(")")
    cleaned = token.replace("$", "").replace(",", "").replace("%", "")
    cleaned = cleaned.replace("(", "").replace(")", "")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return {"value": -value if negative else value, "percent": percent, "token": token}


def numerical_em(prediction: str, gold: str) -> tuple[bool, str, dict | None, dict | None]:
    pred_text, gold_text = prediction.strip().lower(), str(gold).strip().lower()
    if gold_text in {"yes", "no"}:
        exact = pred_text.strip(" .") == gold_text
        return exact, "boolean_exact", None, None
    pred_num, gold_num = parse_number(pred_text), parse_number(gold_text)
    if pred_num is None:
        return False, "no_numeric_prediction", pred_num, gold_num
    if gold_num is None:
        exact = pred_text.strip(" .") == gold_text.strip(" .")
        return exact, "normalized_string_exact", pred_num, gold_num
    tolerance = 1e-4 * max(1.0, abs(gold_num["value"]))
    exact = abs(pred_num["value"] - gold_num["value"]) <= tolerance
    return exact, "numeric_tolerance_1e-4", pred_num, gold_num


def request_ollama(url: str, payload: dict, timeout: float, retries: int) -> dict:
    body = json.dumps(payload).encode()
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2 ** (attempt - 1))
    raise AssertionError("unreachable")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    registry = json.loads(args.methods.read_text())
    protocol = registry["protocol"]
    if args.method not in registry["methods"] or args.seed != protocol["seed"]:
        raise ValueError("method/seed differs from frozen protocol")
    retrieval_path = args.retrieval_dir / f"{args.method}.jsonl"
    retrieval = list(jsonl(retrieval_path))
    if args.limit is not None:
        retrieval = retrieval[:args.limit]
    if any(row["top_k"] != protocol["top_k"] for row in retrieval):
        raise ValueError("retrieval top-k differs from protocol")
    prompt_template = args.prompt.read_text()
    raw_finqa = json.loads(args.finqa_test.read_text())
    gold = {f"test:{item['id']}:{index}": item["qa"]
            for index, item in enumerate(raw_finqa)}
    suffix = f".limit{args.limit}" if args.limit is not None else ""
    output = args.output_dir / f"{args.method}{suffix}.jsonl"
    state_path = args.output_dir / f"{args.method}{suffix}.state.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed = {row["query_id"] for row in jsonl(output)} if output.exists() else set()
    mode = "a" if output.exists() else "w"
    started = time.monotonic()
    with output.open(mode) as handle:
        for index, row in enumerate(retrieval, 1):
            if row["query_id"] in completed:
                continue
            evidence = "\n".join(
                f"[{item['rank']}] {item['text']}" for item in row["retrieved"])
            prompt = prompt_template.format(evidence=evidence, question=row["question"])
            payload = {
                "model": protocol["llm"], "prompt": prompt, "stream": False,
                "keep_alive": -1,
                "options": {"temperature": protocol["temperature"],
                            "num_predict": protocol["max_output_tokens"],
                            "seed": args.seed},
            }
            requested_at = dt.datetime.now(dt.timezone.utc).isoformat()
            response = request_ollama(args.ollama_url, payload, args.timeout, args.retries)
            prediction = str(response.get("response", "")).strip()
            exact, rule, pred_num, gold_num = numerical_em(
                prediction, gold[row["query_id"]]["answer"])
            record = {
                "method": args.method, "query_id": row["query_id"],
                "gold_hit": row["gold_hit"], "question": row["question"],
                "gold_answer": gold[row["query_id"]]["answer"],
                "prediction": prediction, "numerical_em": exact,
                "evaluation_rule": rule, "parsed_prediction": pred_num,
                "parsed_gold": gold_num, "program_predicted": False,
                "retrieved_evidence_ids": [x["evidence_id"] for x in row["retrieved"]],
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "request_time_utc": requested_at,
                "llm": protocol["llm"], "temperature": protocol["temperature"],
                "max_output_tokens": protocol["max_output_tokens"],
                "raw_response": response,
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            if index % 10 == 0:
                logging.info("method=%s completed=%d/%d", args.method, index, len(retrieval))
    rows = list(jsonl(output))
    if len(rows) != len(retrieval):
        raise ValueError("QA output coverage mismatch")
    state = {
        "method": args.method, "queries": len(rows), "seed": args.seed,
        "top_k": protocol["top_k"], "llm": protocol["llm"],
        "temperature": protocol["temperature"],
        "max_output_tokens": protocol["max_output_tokens"],
        "prompt_sha256": sha256(args.prompt),
        "retrieval_sha256": sha256(retrieval_path),
        "numerical_em": sum(row["numerical_em"] for row in rows) / len(rows),
        "program_accuracy": None, "program_accuracy_reason": "no program predicted",
        "errors": sum("error" in row.get("raw_response", {}) for row in rows),
        "elapsed_seconds_current_invocation": time.monotonic() - started,
        "output_sha256": sha256(output), "resume_supported": True,
    }
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    logging.info("finished method=%s numerical_em=%.4f", args.method,
                 state["numerical_em"])


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Day-9 QA failed")
        raise
