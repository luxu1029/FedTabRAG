#!/usr/bin/env python3
"""Encode a JSONL corpus with a frozen BGE model into a resumable memmap."""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


LOG = logging.getLogger("build_index")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def model_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "state.json"
    embeddings_path = args.output / "embeddings.npy"
    count = count_lines(args.corpus)
    fingerprint = model_fingerprint(args.model)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    model = AutoModel.from_pretrained(str(args.model), local_files_only=True)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dimension = int(model.config.hidden_size)
    dtype = np.float16 if args.fp16 else np.float32
    expected = {
        "model": str(args.model.resolve()),
        "model_fingerprint": fingerprint,
        "corpus": str(args.corpus.resolve()),
        "corpus_count": count,
        "dimension": dimension,
        "dtype": np.dtype(dtype).name,
        "max_length": args.max_length,
        "pooling": "cls",
        "normalize": args.normalize,
        "seed": args.seed,
    }
    processed = 0
    if args.resume and state_path.exists() and embeddings_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"Resume state mismatch for {key}: {state.get(key)} != {value}")
        processed = int(state.get("processed", 0))
        embeddings = np.lib.format.open_memmap(embeddings_path, mode="r+")
    else:
        embeddings = np.lib.format.open_memmap(
            embeddings_path, mode="w+", dtype=dtype, shape=(count, dimension)
        )
        atomic_json(state_path, {**expected, "processed": 0, "complete": False})

    started = time.perf_counter()
    batch_texts: List[str] = []
    batch_indices: List[int] = []

    def encode_batch() -> None:
        nonlocal processed, batch_texts, batch_indices
        if not batch_texts:
            return
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=args.fp16 and device.type == "cuda",
            ):
                output = model(**encoded).last_hidden_state[:, 0]
            if args.normalize:
                output = F.normalize(output.float(), p=2, dim=1)
        values = output.detach().cpu().numpy().astype(dtype, copy=False)
        embeddings[batch_indices[0] : batch_indices[-1] + 1] = values
        embeddings.flush()
        processed = batch_indices[-1] + 1
        atomic_json(state_path, {**expected, "processed": processed, "complete": False})
        if processed % (args.batch_size * 20) == 0 or processed == count:
            LOG.info("Progress %d/%d", processed, count)
        batch_texts, batch_indices = [], []

    with args.corpus.open("r", encoding="utf-8") as handle:
        row = 0
        for line in handle:
            if not line.strip():
                continue
            if row < processed:
                row += 1
                continue
            item = json.loads(line)
            batch_texts.append(str(item["text"]))
            batch_indices.append(row)
            row += 1
            if len(batch_texts) >= args.batch_size:
                encode_batch()
        encode_batch()
    if processed != count:
        raise RuntimeError(f"Encoded {processed}, expected {count}")
    atomic_json(
        state_path,
        {**expected, "processed": processed, "complete": True, "elapsed_seconds": time.perf_counter() - started},
    )
    LOG.info("Complete count=%d output=%s", count, embeddings_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        LOG.error("Interrupted; rerun with --resume")
        sys.exit(130)
    except Exception:
        LOG.exception("Fatal index build error")
        sys.exit(2)

