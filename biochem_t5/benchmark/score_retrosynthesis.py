from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .data import write_json
from .original.score_topk import canonicalize_smiles


TOP_K_VALUES = tuple(range(1, 11))


def _read_predictions(path: str | Path) -> tuple[dict[str, dict[str, Any]], int]:
    predictions: dict[str, dict[str, Any]] = {}
    duplicate_products = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid predictions JSONL at line {line_number}") from exc
            product = canonicalize_smiles(payload.get("product", ""))
            if not product:
                raise ValueError(f"Invalid prediction product at line {line_number}")
            if product in predictions:
                duplicate_products += 1
                continue
            predictions[product] = payload
    return predictions, duplicate_products


def _read_test_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {"product_smiles", "substrate_smiles"}
        if not required.issubset(fieldnames):
            missing = ", ".join(sorted(required - fieldnames))
            raise ValueError(f"CSV {path} is missing required columns: {missing}")
        for index, row in enumerate(reader):
            product = canonicalize_smiles(row.get("product_smiles", ""))
            if not product:
                raise ValueError(f"Invalid or empty product at row {index}")
            target = canonicalize_smiles(row.get("substrate_smiles", ""))
            if not target:
                raise ValueError(f"Invalid or empty target at row {index}")
            rows.append({"id": index, "product": product, "target": target})
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _candidate_strings(payload: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for candidate in payload.get("fused_candidates", []):
        if isinstance(candidate, str):
            result.append(candidate)
        elif isinstance(candidate, dict):
            result.append(str(candidate.get("smiles", "")))
    return result


def _canonical_candidates(payload: dict[str, Any], top_k: int) -> tuple[list[str], int]:
    predictions: list[str] = []
    seen: set[str] = set()
    invalid_predictions = 0
    for value in _candidate_strings(payload)[:top_k]:
        prediction = canonicalize_smiles(value)
        if not prediction:
            invalid_predictions += 1
            continue
        if prediction in seen:
            continue
        seen.add(prediction)
        predictions.append(prediction)
    return predictions, invalid_predictions


def score(
    predictions_path: str | Path,
    test_csv: str | Path,
    output: str | Path,
    *,
    details_output: str | Path | None = None,
    require_exact_products: bool = True,
) -> dict[str, Any]:
    test_rows = _read_test_rows(test_csv)
    predictions, duplicate_products = _read_predictions(predictions_path)
    target_products = {row["product"] for row in test_rows}
    prediction_products = set(predictions)
    missing = sorted(target_products - prediction_products)
    extra = sorted(prediction_products - target_products)
    if require_exact_products and (missing or extra or duplicate_products):
        raise ValueError(
            "Prediction product set mismatch: "
            f"missing={len(missing)}, extra={len(extra)}, duplicate={duplicate_products}"
        )

    canonical_by_product: dict[str, list[str]] = {}
    invalid_by_product: dict[str, int] = {}
    for product, payload in predictions.items():
        candidates, invalid_count = _canonical_candidates(payload, max(TOP_K_VALUES))
        canonical_by_product[product] = candidates
        invalid_by_product[product] = invalid_count

    hits = {top_k: 0 for top_k in TOP_K_VALUES}
    invalid_predictions = 0
    details: list[dict[str, Any]] = []
    for row in test_rows:
        candidates = canonical_by_product.get(row["product"], [])
        invalid_predictions += invalid_by_product.get(row["product"], 0)
        matched_rank = next(
            (
                rank
                for rank, prediction in enumerate(candidates, start=1)
                if prediction == row["target"]
            ),
            None,
        )
        for top_k in TOP_K_VALUES:
            hits[top_k] += int(matched_rank is not None and matched_rank <= top_k)
        detail = {
            "id": row["id"],
            "product": row["product"],
            "target": row["target"],
            "matched_rank": matched_rank,
        }
        for top_k in TOP_K_VALUES:
            detail[f"top_{top_k}_hit"] = int(
                matched_rank is not None and matched_rank <= top_k
            )
        details.append(detail)

    total = len(test_rows)
    metrics: dict[str, Any] = {
        "num_test_entries": total,
        "num_test_products": len(target_products),
        "prediction_columns_used": [f"pred_{top_k}" for top_k in TOP_K_VALUES],
        "invalid_predictions": invalid_predictions,
        "exact_match": {
            f"top_{top_k}": hits[top_k] / total for top_k in TOP_K_VALUES
        },
        "hit_counts": {f"top_{top_k}": hits[top_k] for top_k in TOP_K_VALUES},
        "missing_products": len(missing),
        "extra_products": len(extra),
        "duplicate_products": duplicate_products,
        "missing_product_values": missing,
        "extra_product_values": extra,
    }
    output = Path(output)
    write_json(output, metrics)
    details_path = Path(details_output) if details_output else output.with_suffix(".details.jsonl")
    details_path.parent.mkdir(parents=True, exist_ok=True)
    with details_path.open("w", encoding="utf-8") as handle:
        for item in details:
            handle.write(json.dumps(item, sort_keys=True) + "\n")

    print(f"Samples: {total}")
    print("Prediction columns used: " + ", ".join(metrics["prediction_columns_used"]))
    print(f"Invalid predictions skipped: {invalid_predictions}")
    for top_k in TOP_K_VALUES:
        accuracy = metrics["exact_match"][f"top_{top_k}"]
        print(f"Top-{top_k} Accuracy: {accuracy * 100:.3f}% ({hits[top_k]}/{total})")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Score retrosynthesis predictions")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--details-output")
    parser.add_argument("--allow-product-mismatch", action="store_true")
    args = parser.parse_args()
    score(
        args.predictions,
        args.test_csv,
        args.output,
        details_output=args.details_output,
        require_exact_products=not args.allow_product_mismatch,
    )


if __name__ == "__main__":
    main()
