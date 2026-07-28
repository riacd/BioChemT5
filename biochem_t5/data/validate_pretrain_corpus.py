#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = [
    "rxn",
    "rsmiles_status",
    "rsmiles_views",
    "rsmiles_view_count",
    "unique_rsmiles_view_count",
    "ec_levels",
    "reaction_center_changed_atom_maps",
    "reaction_center_neighbor_atom_maps",
    "reaction_center_atom_maps",
]
VIEW_FIELDS = ("forward_input", "forward_target", "retro_input", "retro_target")


def validate_record(record: dict[str, Any], line_no: int, errors: Counter[str], first_errors: list[dict[str, Any]]) -> int:
    for key in REQUIRED_FIELDS:
        if key not in record:
            errors[f"missing_{key}"] += 1
            if len(first_errors) < 20:
                first_errors.append({"line_no": line_no, "error": f"missing_{key}", "rxn_id": record.get("rxn_id")})

    views = record.get("rsmiles_views") or []
    if not isinstance(views, list):
        errors["views_not_list"] += 1
        if len(first_errors) < 20:
            first_errors.append({"line_no": line_no, "error": "views_not_list", "rxn_id": record.get("rxn_id")})
        views = []

    if record.get("rsmiles_view_count") != len(views):
        errors["view_count_mismatch"] += 1
        if len(first_errors) < 20:
            first_errors.append(
                {
                    "line_no": line_no,
                    "error": "view_count_mismatch",
                    "rxn_id": record.get("rxn_id"),
                    "declared": record.get("rsmiles_view_count"),
                    "actual": len(views),
                }
            )
    if record.get("unique_rsmiles_view_count") != len(views):
        errors["unique_count_mismatch"] += 1
    if len(views) > 20:
        errors["view_count_gt_20"] += 1
    if record.get("rsmiles_status") == "ok" and not views:
        errors["ok_without_views"] += 1
    if record.get("rsmiles_status") != "ok" and views:
        errors["non_ok_with_views"] += 1
    if not isinstance(record.get("ec_levels"), dict):
        errors["ec_levels_not_dict"] += 1

    seen: set[tuple[str, str, str, str]] = set()
    for idx, view in enumerate(views):
        if not isinstance(view, dict):
            errors["view_not_dict"] += 1
            continue
        key = tuple(str(view.get(field) or "") for field in VIEW_FIELDS)
        if any(not part for part in key):
            errors["empty_view_field"] += 1
        if key in seen:
            errors["duplicate_view_key"] += 1
            if len(first_errors) < 20:
                first_errors.append({"line_no": line_no, "error": "duplicate_view_key", "rxn_id": record.get("rxn_id"), "view_idx": idx})
        seen.add(key)

    return len(views)


def cmd_validate(args: argparse.Namespace) -> int:
    started = time.time()
    path = Path(args.input_jsonl)
    errors: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    center_status_counts: Counter[str] = Counter()
    center_source_counts: Counter[str] = Counter()
    view_histogram: Counter[str] = Counter()
    first_errors: list[dict[str, Any]] = []
    records = 0

    with path.open("r", encoding="utf-8") as reader:
        for line_no, line in enumerate(reader, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors["json_parse_error"] += 1
                if len(first_errors) < 20:
                    first_errors.append({"line_no": line_no, "error": "json_parse_error", "message": str(exc)})
                continue

            records += 1
            view_count = validate_record(record, line_no, errors, first_errors)
            status_counts[str(record.get("rsmiles_status"))] += 1
            center_status_counts[str(record.get("reaction_center_status"))] += 1
            center_source_counts[str(record.get("reaction_center_source"))] += 1
            view_histogram[str(view_count)] += 1

            if args.limit is not None and records >= args.limit:
                break
            if args.progress_every and records % args.progress_every == 0:
                print(json.dumps({"scanned": records, "elapsed_s": round(time.time() - started, 1)}, sort_keys=True), flush=True)

    summary = {
        "input_jsonl": str(path),
        "records": records,
        "elapsed_s": round(time.time() - started, 3),
        "errors": dict(errors),
        "first_errors": first_errors,
        "rsmiles_status_counts": dict(status_counts),
        "reaction_center_status_counts": dict(center_status_counts),
        "reaction_center_source_counts": dict(center_source_counts),
        "view_count_histogram": dict(sorted(view_histogram.items(), key=lambda item: int(item[0]))),
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if not errors else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the compact BiochemT5 pretraining corpus.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=500000)
    parser.set_defaults(func=cmd_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
