#!/usr/bin/env python3
"""Run UniTable's structure-only model on a rendered-table manifest."""

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import tokenizers as tk
from PIL import Image
from torch import Tensor, nn
from torchvision import transforms


LOG = logging.getLogger("run_unitable_structure")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unitable-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--max-decode-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--flush-every", type=int, default=1)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    values: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return values


def read_existing(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    entries = read_jsonl(path)
    output: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        key = str(entry.get("sample_key", ""))
        if not key:
            raise ValueError(f"Existing output has entry without sample_key: {path}")
        output[key] = entry
    return output


def atomic_write(path: Path, entries: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def build_model(
    unitable_root: Path, checkpoint: Path, device: torch.device
) -> Tuple[tk.Tokenizer, nn.Module, List[int], int]:
    root = str(unitable_root.resolve())
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)

    from src.model import Decoder, Encoder, EncoderDecoder, ImgLinearBackbone
    from src.trainer.utils import VALID_HTML_TOKEN

    vocab_path = unitable_root / "vocab" / "vocab_html.json"
    vocab = tk.Tokenizer.from_file(str(vocab_path))
    d_model = 768
    dropout = 0.2
    backbone = ImgLinearBackbone(d_model=d_model, patch_size=16)
    encoder = Encoder(
        d_model=d_model,
        nhead=12,
        dropout=dropout,
        activation="gelu",
        norm_first=True,
        nlayer=12,
        ff_ratio=4,
    )
    decoder = Decoder(
        d_model=d_model,
        nhead=12,
        dropout=dropout,
        activation="gelu",
        norm_first=True,
        nlayer=4,
        ff_ratio=4,
    )
    model = EncoderDecoder(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        vocab_size=vocab.get_vocab_size(),
        d_model=d_model,
        padding_idx=vocab.token_to_id("<pad>"),
        max_seq_len=784,
        dropout=dropout,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    state = torch.load(str(checkpoint), map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError("Structure checkpoint is not a state_dict")
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    whitelist = [
        token_id
        for token in VALID_HTML_TOKEN
        for token_id in [vocab.token_to_id(token)]
        if token_id is not None
    ]
    eos_id = vocab.token_to_id("<eos>")
    if eos_id is None:
        raise ValueError("HTML vocabulary does not contain <eos>")
    return vocab, model, whitelist, eos_id


def image_transform(size: int):
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.86597056, 0.88463002, 0.87491087],
                std=[0.20686628, 0.18201602, 0.18485524],
            ),
        ]
    )


def decode_structure(
    model: nn.Module,
    image: Tensor,
    prefix: Sequence[int],
    max_decode_len: int,
    eos_id: int,
    whitelist: Sequence[int],
) -> Tuple[List[int], List[float]]:
    from src.utils import subsequent_mask

    model.eval()
    with torch.inference_mode():
        memory = model.encode(image)
        context = torch.tensor(
            prefix, dtype=torch.int32, device=image.device
        ).repeat(image.shape[0], 1)
        probabilities: List[float] = []
        all_ids = set(range(model.generator.out_features))
        blacklist = list(all_ids.difference(set(whitelist)))

        for _ in range(max_decode_len):
            causal_mask = subsequent_mask(context.shape[1]).to(image.device)
            logits = model.decode(memory, context, causal_mask, None)
            logits = model.generator(logits)[:, -1, :]
            logits[..., blacklist] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_probs, next_tokens = probs.topk(1)
            probabilities.append(float(next_probs[0, 0].detach().cpu()))
            context = torch.cat([context, next_tokens], dim=1)
            if int(next_tokens[0, 0]) == eos_id:
                break
    return [int(value) for value in context[0].detach().cpu().tolist()], probabilities


def structure_tokens(decoded: str) -> List[str]:
    from src.utils import html_str_to_token_list

    return html_str_to_token_list(decoded)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.batch_size != 1:
        raise ValueError("This structure-only runner currently requires --batch-size 1")
    if args.flush_every <= 0:
        raise ValueError("--flush-every must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    manifest = read_jsonl(args.manifest)
    if args.limit is not None:
        manifest = manifest[: args.limit]
    existing = read_existing(args.output) if args.resume else {}
    checkpoint_hash = sha256(args.checkpoint)
    vocab_path = args.unitable_root / "vocab" / "vocab_html.json"
    vocab_hash = sha256(vocab_path)
    LOG.info(
        "Loading structure-only checkpoint=%s sha256=%s device=%s",
        args.checkpoint,
        checkpoint_hash,
        device,
    )
    vocab, model, whitelist, eos_id = build_model(
        args.unitable_root, args.checkpoint, device
    )
    transform = image_transform(args.image_size)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    results: List[Dict[str, Any]] = []
    failures = 0
    skipped = 0
    started = time.perf_counter()
    for index, item in enumerate(manifest, 1):
        key = str(item.get("sample_key", ""))
        old = existing.get(key)
        if old and (old.get("status") == "ok" or not args.retry_errors):
            results.append(old)
            skipped += 1
            continue
        sample_started = time.perf_counter()
        base = {
            "sample_key": key,
            "id": item.get("id"),
            "split": item.get("split"),
            "image_path": item.get("image_path"),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "vocab_sha256": vocab_hash,
            "device": str(device),
        }
        try:
            image_path = Path(str(item["image_path"]))
            with Image.open(image_path) as pil_image:
                tensor = transform(pil_image.convert("RGB")).unsqueeze(0).to(device)
            token_ids, token_probs = decode_structure(
                model,
                tensor,
                prefix=[vocab.token_to_id("[html]")],
                max_decode_len=args.max_decode_len,
                eos_id=eos_id,
                whitelist=whitelist,
            )
            decoded = vocab.decode(token_ids, skip_special_tokens=False)
            tokens = structure_tokens(decoded)
            raw_html = "".join(tokens)
            confidence = (
                math.exp(
                    sum(math.log(max(probability, 1e-12)) for probability in token_probs)
                    / len(token_probs)
                )
                if token_probs
                else 0.0
            )
            result = {
                **base,
                "status": "ok",
                "raw_token_ids": token_ids,
                "raw_decoded": decoded,
                "structure_tokens": tokens,
                "html": raw_html,
                "non_empty_slots": sum("[]" in token for token in tokens),
                "confidence": confidence,
                "token_probabilities": token_probs,
                "decode_steps": len(token_probs),
                "ended_with_eos": bool(token_ids and token_ids[-1] == eos_id),
                "elapsed_seconds": time.perf_counter() - sample_started,
                "peak_gpu_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else 0
                ),
                "error": None,
            }
        except Exception as exc:
            failures += 1
            LOG.exception("Inference failed for %s", key)
            result = {
                **base,
                "status": "error",
                "raw_token_ids": [],
                "raw_decoded": "",
                "structure_tokens": [],
                "html": "",
                "non_empty_slots": 0,
                "confidence": 0.0,
                "token_probabilities": [],
                "decode_steps": 0,
                "ended_with_eos": False,
                "elapsed_seconds": time.perf_counter() - sample_started,
                "peak_gpu_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else 0
                ),
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)
        if index % args.flush_every == 0 or index == len(manifest):
            atomic_write(args.output, results)
        LOG.info(
            "Progress %d/%d status=%s steps=%d confidence=%.6f elapsed=%.3fs",
            index,
            len(manifest),
            result["status"],
            result["decode_steps"],
            result["confidence"],
            result["elapsed_seconds"],
        )

    atomic_write(args.output, results)
    LOG.info(
        "Completed total=%d failures=%d skipped=%d elapsed=%.1fs",
        len(results),
        failures,
        skipped,
        time.perf_counter() - started,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        LOG.error("Interrupted; rerun with --resume")
        sys.exit(130)
    except Exception:
        LOG.exception("Fatal structure inference error")
        sys.exit(2)

