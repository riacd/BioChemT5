#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterator

from .smiles_tokenizer import BASE_SPECIAL_TOKENS, SENTINEL_TOKENS, SmilesTokenizer, smiles_tokenize, strip_atom_maps


def _iter_line_chunks(path: Path, chunk_size: int, max_records: int | None) -> Iterator[list[str]]:
    chunk: list[str] = []
    seen = 0
    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            if not line.strip():
                continue
            chunk.append(line)
            seen += 1
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
            if max_records is not None and seen >= max_records:
                break
    if chunk:
        yield chunk


def _empty_stats() -> dict[str, Any]:
    return {
        "records": 0,
        "mapped_records": 0,
        "rsmiles_records": 0,
        "rsmiles_views": 0,
        "canonical_reaction_tokens": 0,
        "mlm_reaction_tokens": 0,
        "rsmiles_view_reaction_tokens": 0,
        "tokenizer_training_texts": 0,
        "tokenizer_training_tokens": 0,
        "json_errors": 0,
        "canonical_len_hist": Counter(),
        "mlm_len_hist": Counter(),
        "rsmiles_view_len_hist": Counter(),
    }


def _update_len(stats: dict[str, Any], hist_key: str, total_key: str, length: int) -> None:
    stats[hist_key][length] += 1
    stats[total_key] += length


def _tokenize_for_vocab(text: str) -> list[str]:
    return smiles_tokenize(strip_atom_maps(text))


def _process_chunk(args: tuple[list[str], bool]) -> tuple[Counter[str], dict[str, Any]]:
    lines, include_rsmiles_views = args
    counter: Counter[str] = Counter()
    stats = _empty_stats()

    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            stats["json_errors"] += 1
            continue

        stats["records"] += 1
        rxn = str(record.get("rxn") or "")
        canonical_tokens = _tokenize_for_vocab(rxn)
        counter.update(canonical_tokens)
        _update_len(stats, "canonical_len_hist", "canonical_reaction_tokens", len(canonical_tokens))
        stats["tokenizer_training_texts"] += 1
        stats["tokenizer_training_tokens"] += len(canonical_tokens)

        mapped_rxn = record.get("mapped_rxn")
        if isinstance(mapped_rxn, str) and mapped_rxn:
            stats["mapped_records"] += 1
            mlm_tokens = _tokenize_for_vocab(mapped_rxn)
            counter.update(mlm_tokens)
            stats["tokenizer_training_texts"] += 1
            stats["tokenizer_training_tokens"] += len(mlm_tokens)
        else:
            mlm_tokens = canonical_tokens
        _update_len(stats, "mlm_len_hist", "mlm_reaction_tokens", len(mlm_tokens))

        views = record.get("rsmiles_views")
        if include_rsmiles_views and isinstance(views, list) and views:
            stats["rsmiles_records"] += 1
            for view in views:
                if not isinstance(view, dict):
                    continue
                forward_input = view.get("forward_input")
                forward_target = view.get("forward_target")
                if not isinstance(forward_input, str) or not isinstance(forward_target, str):
                    continue
                view_tokens = _tokenize_for_vocab(f"{forward_input}>>{forward_target}")
                counter.update(view_tokens)
                stats["rsmiles_views"] += 1
                stats["tokenizer_training_texts"] += 1
                stats["tokenizer_training_tokens"] += len(view_tokens)
                _update_len(stats, "rsmiles_view_len_hist", "rsmiles_view_reaction_tokens", len(view_tokens))

    return counter, stats


def _merge_stats(base: dict[str, Any], addon: dict[str, Any]) -> None:
    for key, value in addon.items():
        if key.endswith("_hist"):
            base[key].update(value)
        else:
            base[key] += value


def _quantiles(hist: Counter[int], quantile_values: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99)) -> dict[str, int | None]:
    total = sum(hist.values())
    if total <= 0:
        return {f"p{int(q * 100)}": None for q in quantile_values}
    out: dict[str, int | None] = {}
    cumulative = 0
    targets = {q: max(1, int(total * q + 0.999999)) for q in quantile_values}
    pending = list(quantile_values)
    for length in sorted(hist):
        cumulative += hist[length]
        while pending and cumulative >= targets[pending[0]]:
            q = pending.pop(0)
            out[f"p{int(q * 100)}"] = length
    for q in pending:
        out[f"p{int(q * 100)}"] = max(hist)
    return out


def _length_summary(hist: Counter[int], total_tokens: int) -> dict[str, Any]:
    count = sum(hist.values())
    summary = {
        "count": count,
        "total_tokens": total_tokens,
        "avg_tokens": (total_tokens / count) if count else 0.0,
        "min_tokens": min(hist) if hist else None,
        "max_tokens": max(hist) if hist else None,
    }
    summary.update(_quantiles(hist))
    return summary


def _build_tokenizer(counter: Counter[str], vocab_size: int) -> SmilesTokenizer:
    token_to_id: dict[str, int] = {}
    for token in BASE_SPECIAL_TOKENS + SENTINEL_TOKENS:
        token_to_id.setdefault(token, len(token_to_id))
    for token, _count in counter.most_common():
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)
        if len(token_to_id) >= vocab_size:
            break
    return SmilesTokenizer(token_to_id)


def build_tokenizer_and_stats(args: argparse.Namespace) -> dict[str, Any]:
    input_jsonl = Path(args.input_jsonl)
    started = time.time()
    total_counter: Counter[str] = Counter()
    total_stats = _empty_stats()
    num_workers = max(1, int(args.num_workers))
    chunks = _iter_line_chunks(input_jsonl, int(args.chunk_size), args.max_records)
    last_report = 0

    def maybe_report() -> None:
        nonlocal last_report
        if args.log_every <= 0:
            return
        records = int(total_stats["records"])
        if records - last_report < args.log_every:
            return
        last_report = records
        print(
            json.dumps(
                {
                    "records": records,
                    "rsmiles_views": total_stats["rsmiles_views"],
                    "elapsed_seconds": round(time.time() - started, 3),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if num_workers == 1:
        for counter, stats in map(lambda chunk: _process_chunk((chunk, args.include_rsmiles_views)), chunks):
            total_counter.update(counter)
            _merge_stats(total_stats, stats)
            maybe_report()
    else:
        with Pool(processes=num_workers) as pool:
            work = ((chunk, args.include_rsmiles_views) for chunk in chunks)
            for counter, stats in pool.imap_unordered(_process_chunk, work, chunksize=1):
                total_counter.update(counter)
                _merge_stats(total_stats, stats)
                maybe_report()

    tokenizer = _build_tokenizer(total_counter, int(args.vocab_size))
    tokenizer.save(args.out_vocab_json)

    specials = set(BASE_SPECIAL_TOKENS + SENTINEL_TOKENS)
    non_special_vocab_size = sum(1 for token in tokenizer.token_to_id if token not in specials)
    included_tokens = set(tokenizer.token_to_id)
    oov_token_types = [token for token in total_counter if token not in included_tokens]
    oov_token_count = sum(total_counter[token] for token in oov_token_types)

    stats_payload = {
        "input_jsonl": str(input_jsonl),
        "input_size_bytes": input_jsonl.stat().st_size if input_jsonl.exists() else None,
        "include_rsmiles_views": bool(args.include_rsmiles_views),
        "max_records": args.max_records,
        "num_workers": num_workers,
        "chunk_size": int(args.chunk_size),
        "elapsed_seconds": round(time.time() - started, 3),
        "records": total_stats["records"],
        "mapped_records": total_stats["mapped_records"],
        "rsmiles_records": total_stats["rsmiles_records"],
        "rsmiles_views": total_stats["rsmiles_views"],
        "json_errors": total_stats["json_errors"],
        "vocab_path": str(args.out_vocab_json),
        "requested_vocab_size": int(args.vocab_size),
        "actual_vocab_size": len(tokenizer),
        "reserved_special_tokens": len(specials),
        "non_special_vocab_size": non_special_vocab_size,
        "observed_token_types": len(total_counter),
        "oov_token_types_after_vocab_cap": len(oov_token_types),
        "oov_token_count_after_vocab_cap": oov_token_count,
        "top_tokens": total_counter.most_common(int(args.top_k)),
        "canonical_reaction": _length_summary(
            total_stats["canonical_len_hist"], total_stats["canonical_reaction_tokens"]
        ),
        "mlm_reaction": _length_summary(total_stats["mlm_len_hist"], total_stats["mlm_reaction_tokens"]),
        "rsmiles_view_reaction": _length_summary(
            total_stats["rsmiles_view_len_hist"], total_stats["rsmiles_view_reaction_tokens"]
        ),
        "tokenizer_training_corpus": {
            "texts": total_stats["tokenizer_training_texts"],
            "total_tokens": total_stats["tokenizer_training_tokens"],
            "avg_tokens_per_text": (
                total_stats["tokenizer_training_tokens"] / total_stats["tokenizer_training_texts"]
                if total_stats["tokenizer_training_texts"]
                else 0.0
            ),
            "avg_tokens_per_record": (
                total_stats["tokenizer_training_tokens"] / total_stats["records"] if total_stats["records"] else 0.0
            ),
        },
    }

    out_stats = Path(args.out_stats_json)
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    out_stats.write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return stats_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--out-vocab-json", required=True)
    parser.add_argument("--out-stats-json", required=True)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=max(1, min(64, os.cpu_count() or 1)))
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=100000)
    parser.add_argument("--include-rsmiles-views", action="store_true")
    args = parser.parse_args()
    stats = build_tokenizer_and_stats(args)
    print(json.dumps({key: stats[key] for key in ("records", "actual_vocab_size", "elapsed_seconds")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
