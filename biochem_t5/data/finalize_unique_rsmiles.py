#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, TextIO


def _open_text(path: str | Path, mode: str = "rt") -> TextIO:
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")  # type: ignore[return-value]
    return path.open(mode, encoding="utf-8")


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as writer:
        json.dump(payload, writer, ensure_ascii=False, indent=2, sort_keys=True)
        writer.write("\n")


def _view_key(view: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(view.get("forward_input") or ""),
        str(view.get("forward_target") or ""),
        str(view.get("retro_input") or ""),
        str(view.get("retro_target") or ""),
    )


def dedupe_record_views(record: dict[str, Any], max_views: int = 20) -> tuple[dict[str, Any], int]:
    views = record.get("rsmiles_views")
    if not isinstance(views, list):
        record["rsmiles_views"] = []
        record["rsmiles_view_count"] = 0
        record["unique_rsmiles_view_count"] = 0
        return record, 0

    seen: set[tuple[str, str, str, str]] = set()
    unique_views: list[dict[str, Any]] = []
    duplicate_count = 0
    for view in views:
        if not isinstance(view, dict):
            duplicate_count += 1
            continue
        key = _view_key(view)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        out_view = dict(view)
        out_view["aug_id"] = len(unique_views)
        unique_views.append(out_view)
        if len(unique_views) >= max_views:
            break

    old_count = int(record.get("rsmiles_view_count") or len(views))
    duplicate_count += max(0, old_count - len(views))
    record["rsmiles_views"] = unique_views
    record["rsmiles_view_count"] = len(unique_views)
    record["unique_rsmiles_view_count"] = len(unique_views)
    if record.get("rsmiles_status") == "ok" and not unique_views:
        record["rsmiles_status"] = "empty_unique_rsmiles"
    return record, duplicate_count


def cmd_finalize(args: argparse.Namespace) -> int:
    started = time.time()
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    status_counts: Counter[str] = Counter()
    view_count_histogram: Counter[str] = Counter()
    total_views_before = 0
    total_views_after = 0
    duplicate_views_removed = 0
    records_with_duplicates = 0
    written = 0

    with _open_text(args.input_jsonl) as reader, out_jsonl.open("w", encoding="utf-8") as writer:
        for line in reader:
            if not line.strip():
                continue
            record = json.loads(line)
            before = int(record.get("rsmiles_view_count") or len(record.get("rsmiles_views") or []))
            record, removed = dedupe_record_views(record, max_views=args.max_views)
            after = int(record.get("rsmiles_view_count") or 0)
            writer.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

            written += 1
            total_views_before += before
            total_views_after += after
            duplicate_views_removed += removed
            records_with_duplicates += int(removed > 0)
            status_counts[str(record.get("rsmiles_status"))] += 1
            view_count_histogram[str(after)] += 1

            if args.limit is not None and written >= args.limit:
                break
            if args.progress_every and written % args.progress_every == 0:
                print(
                    f"[finalize-unique] written={written:,} "
                    f"removed={duplicate_views_removed:,} elapsed={time.time() - started:.1f}s",
                    flush=True,
                )

    stats = {
        "input_jsonl": str(args.input_jsonl),
        "out_jsonl": str(out_jsonl),
        "records_written": written,
        "max_views": args.max_views,
        "status_counts": dict(status_counts),
        "view_count_histogram": dict(sorted(view_count_histogram.items(), key=lambda item: int(item[0]))),
        "total_views_before": total_views_before,
        "total_views_after": total_views_after,
        "duplicate_views_removed": duplicate_views_removed,
        "records_with_duplicate_views": records_with_duplicates,
        "elapsed_s": time.time() - started,
    }
    _write_json(args.out_stats, stats)
    print(f"[finalize-unique done] written={written:,} stats={args.out_stats}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Drop duplicate R-SMILES views without padding back to 20.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-stats", required=True)
    parser.add_argument("--max-views", type=int, default=20)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.set_defaults(func=cmd_finalize)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
