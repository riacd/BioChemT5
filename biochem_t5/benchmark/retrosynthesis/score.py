"""Score retrosynthesis predictions with unified top-k semantics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger

from biochem_t5.benchmark.common import write_json


RDLogger.DisableLog("rdApp.*")


TOP_K_VALUES = tuple(range(1, 11))


def canonicalize_smiles(smiles: str) -> str:
    if smiles is None:
        return ""
    smiles = str(smiles).strip()
    if not smiles:
        return ""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""

    for atom in mol.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")

    try:
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return ""

    fragments = [fragment for fragment in canonical.split(".") if fragment]
    fragments.sort()
    return ".".join(fragments)


def get_prediction_columns(
    fieldnames: list[str],
    prediction_prefix: str,
    top_k: int,
) -> list[str]:
    columns = []
    for index in range(1, top_k + 1):
        name = f"{prediction_prefix}{index}"
        if name in fieldnames:
            columns.append(name)
    return columns


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


def _rows_for_topk_scorer(
    test_rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for test_row in test_rows:
        candidates = _candidate_strings(predictions.get(test_row["product"], {}))[:top_k]
        row = dict(test_row)
        for rank in range(1, top_k + 1):
            row[f"pred_{rank}"] = candidates[rank - 1] if rank <= len(candidates) else ""
        rows.append(row)
    return rows


def _score_topk_rows(
    rows: list[dict[str, Any]],
    prediction_prefix: str = "pred_",
    top_k: int = 10,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Canonicalize, deduplicate, and score row-wise top-k predictions."""
    if not rows:
        raise ValueError("No rows provided for scoring")

    fieldnames = list(rows[0].keys())
    prediction_cols = get_prediction_columns(fieldnames, prediction_prefix, top_k)
    if not prediction_cols:
        raise ValueError(
            f"No prediction columns found with prefix '{prediction_prefix}'. "
            f"Expected columns like {prediction_prefix}1, {prediction_prefix}2, ..."
        )

    hits = [0 for _ in range(len(prediction_cols))]
    invalid_predictions = 0
    detail_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        sample_id = row.get("id", idx)
        target = canonicalize_smiles(row.get("target", ""))
        if not target:
            raise ValueError(f"Invalid or empty target at row {idx} (id={sample_id})")

        seen = set()
        canonical_predictions = []
        for col in prediction_cols:
            prediction = canonicalize_smiles(row.get(col, ""))
            if not prediction:
                invalid_predictions += 1
                continue
            if prediction in seen:
                continue
            seen.add(prediction)
            canonical_predictions.append(prediction)

        matched_rank = None
        for rank_idx, prediction in enumerate(canonical_predictions):
            if prediction == target:
                matched_rank = rank_idx + 1
                break

        if matched_rank is not None:
            for k_idx in range(matched_rank - 1, len(hits)):
                hits[k_idx] += 1

        detail = {
            "id": sample_id,
            "product": row["product"],
            "target": target,
            "matched_rank": matched_rank,
        }
        for k_idx in range(len(prediction_cols)):
            detail[f"top_{k_idx + 1}_hit"] = int(
                matched_rank is not None and matched_rank <= k_idx + 1
            )
        detail_rows.append(detail)

    total = len(rows)
    metrics = {
        "prediction_columns_used": prediction_cols,
        "invalid_predictions": invalid_predictions,
        "exact_match": {
            f"top_{index}": hit_count / total
            for index, hit_count in enumerate(hits, start=1)
        },
        "hit_counts": {
            f"top_{index}": hit_count
            for index, hit_count in enumerate(hits, start=1)
        },
    }
    return metrics, detail_rows


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

    scoring_rows = _rows_for_topk_scorer(test_rows, predictions, max(TOP_K_VALUES))
    topk_metrics, details = _score_topk_rows(
        scoring_rows,
        top_k=max(TOP_K_VALUES),
    )
    total = len(test_rows)
    metrics: dict[str, Any] = {
        "scoring_implementation": "biochem_t5/benchmark/retrosynthesis/score.py",
        "num_test_entries": total,
        "num_test_products": len(target_products),
        **topk_metrics,
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
    print(f"Invalid predictions skipped: {metrics['invalid_predictions']}")
    for top_k in TOP_K_VALUES:
        accuracy = metrics["exact_match"][f"top_{top_k}"]
        hit_count = metrics["hit_counts"][f"top_{top_k}"]
        print(f"Top-{top_k} Accuracy: {accuracy * 100:.3f}% ({hit_count}/{total})")
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
