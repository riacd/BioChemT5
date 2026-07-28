#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger


RDLogger.DisableLog("rdApp.*")


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
        cano = Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return ""

    fragments = [frag for frag in cano.split(".") if frag]
    fragments.sort()
    return ".".join(fragments)


def load_rows(input_path: Path):
    if input_path.suffix.lower() == ".jsonl":
        with input_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    with input_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def get_prediction_columns(fieldnames, prediction_prefix, top_k):
    cols = []
    for i in range(1, top_k + 1):
        name = f"{prediction_prefix}{i}"
        if name in fieldnames:
            cols.append(name)
    return cols


def main():
    parser = argparse.ArgumentParser(description="Compute unified top-k accuracy from per-sample prediction lists.")
    parser.add_argument("--input", required=True, help="CSV or JSONL file containing one row per test sample.")
    parser.add_argument("--target-col", default="target", help="Ground-truth column name.")
    parser.add_argument("--id-col", default="id", help="Sample id column name.")
    parser.add_argument("--prediction-prefix", default="pred_", help="Prediction column prefix, e.g. pred_ -> pred_1..pred_10.")
    parser.add_argument("--top-k", type=int, default=10, help="Maximum K to evaluate.")
    parser.add_argument("--save-details", default="", help="Optional path to save per-sample detailed CSV.")
    args = parser.parse_args()

    input_path = Path(args.input)
    rows = list(load_rows(input_path))
    if not rows:
        raise ValueError(f"No rows found in {input_path}")

    fieldnames = list(rows[0].keys())
    prediction_cols = get_prediction_columns(fieldnames, args.prediction_prefix, args.top_k)
    if not prediction_cols:
        raise ValueError(
            f"No prediction columns found with prefix '{args.prediction_prefix}' in {input_path}. "
            f"Expected columns like {args.prediction_prefix}1, {args.prediction_prefix}2, ..."
        )

    hits = [0 for _ in range(len(prediction_cols))]
    invalid_predictions = 0
    detail_rows = []

    for idx, row in enumerate(rows):
        sample_id = row.get(args.id_col, idx)
        target_raw = row.get(args.target_col, "")
        target = canonicalize_smiles(target_raw)
        if not target:
            raise ValueError(f"Invalid or empty target at row {idx} (id={sample_id})")

        seen = set()
        predictions = []
        for col in prediction_cols:
            pred = canonicalize_smiles(row.get(col, ""))
            if not pred:
                invalid_predictions += 1
                continue
            if pred in seen:
                continue
            seen.add(pred)
            predictions.append(pred)

        matched_rank = None
        for rank_idx, pred in enumerate(predictions):
            if pred == target:
                matched_rank = rank_idx + 1
                break

        if matched_rank is not None:
            for k_idx in range(matched_rank - 1, len(hits)):
                hits[k_idx] += 1

        detail = {
            "id": sample_id,
            "target": target,
            "matched_rank": matched_rank if matched_rank is not None else "",
        }
        for k_idx in range(len(prediction_cols)):
            detail[f"top_{k_idx + 1}_hit"] = int(matched_rank is not None and matched_rank <= (k_idx + 1))
        detail_rows.append(detail)

    total = len(rows)
    print(f"Samples: {total}")
    print(f"Prediction columns used: {', '.join(prediction_cols)}")
    print(f"Invalid predictions skipped: {invalid_predictions}")
    for k_idx, hit_count in enumerate(hits, start=1):
        print(f"Top-{k_idx} Accuracy: {hit_count / total * 100:.3f}% ({hit_count}/{total})")

    if args.save_details:
        out_path = Path(args.save_details)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detail_rows)
        print(f"Saved details to {out_path}")


if __name__ == "__main__":
    main()
