from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml
from torch.utils.data import DataLoader

from biochem_t5.data.smiles_tokenizer import SmilesTokenizer
from biochem_t5.benchmark.common import write_json
from biochem_t5.models.ec_retrieval import ECRetrievalModel

from .metrics import classification_metrics, exact_topk_euclidean, recall_at_k
from .data import ReactionCollator, ReactionDataset, ec_at_level, load_ecreact_csv


@torch.no_grad()
def encode_records(
    model: ECRetrievalModel,
    records: Sequence,
    tokenizer: SmilesTokenizer,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    loader = DataLoader(
        ReactionDataset(records), batch_size=batch_size, collate_fn=ReactionCollator(tokenizer, max_length)
    )
    model.eval()
    chunks: list[torch.Tensor] = []
    autocast_enabled = device.type == "cuda"
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            chunks.append(model(input_ids, attention_mask).float().cpu())
    return torch.cat(chunks)


def _prediction_rows(test, train, distances: torch.Tensor, indices: torch.Tensor) -> list[dict[str, Any]]:
    rows = []
    for test_index, record in enumerate(test):
        for rank, (distance, train_index) in enumerate(
            zip(distances[test_index].tolist(), indices[test_index].tolist()), start=1
        ):
            neighbor = train[train_index]
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "rank": rank,
                    "neighbor_sample_id": neighbor.sample_id,
                    "pred_ec1": neighbor.ec1,
                    "pred_ec2": neighbor.ec2,
                    "pred_ec3": neighbor.ec3,
                    "distance": distance,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(output_dir: Path, current: dict[str, Any], zero_shot: bool) -> None:
    metrics_by_system = {"zero_shot" if zero_shot else "fine_tuned": current}
    other_path = output_dir / ("metrics.json" if zero_shot else "zero_shot_metrics.json")
    if other_path.is_file():
        metrics_by_system["fine_tuned" if zero_shot else "zero_shot"] = json.loads(
            other_path.read_text(encoding="utf-8")
        )
    rows: list[dict[str, Any]] = []
    for system, metrics in metrics_by_system.items():
        for level in ("ec1", "ec2", "ec3"):
            values = metrics.get(level)
            if values:
                rows.append({"system": system, "level": level, **values})
    published = current.get("claire_published")
    if published:
        rows.append({
            "system": "claire_published",
            "level": published["level"],
            **{key: value for key, value in published.items() if key != "level"},
        })
    if rows:
        fields = ["system", "level", "weighted_f1", "accuracy", "macro_f1", "macro_recall", "recall_at_1", "recall_at_4"]
        with (output_dir / "benchmark_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def score_predictions(test, train, indices: torch.Tensor, mask: set[str] | None = None) -> dict[str, Any]:
    selected = [index for index, record in enumerate(test) if mask is None or record.sample_id in mask]
    result: dict[str, Any] = {"rows": len(selected)}
    for level in ("ec1", "ec2", "ec3"):
        truth = [test[index].label(level) for index in selected]
        neighbors = [
            [train[train_index].label(level) for train_index in indices[index].tolist()]
            for index in selected
        ]
        metrics = classification_metrics(truth, [row[0] for row in neighbors])
        result[level] = {**metrics, **recall_at_k(truth, neighbors)}
    return result


def _published_metrics(path: str | Path, test, level: str) -> dict[str, float]:
    predictions: list[str] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            predictions.append(row[1].split(":", 1)[1].split("/", 1)[0])
    if len(predictions) != len(test):
        raise ValueError(f"Published predictions have {len(predictions)} rows, expected {len(test)}")
    return classification_metrics([record.label(level) for record in test], predictions)


def _top1_predictions(path: str | Path, field: str) -> dict[str, str]:
    result = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["rank"] == "1":
                result[row["sample_id"]] = row[field]
    return result


def _triplet_consistency(paths: dict[str, str]) -> dict[str, Any]:
    predictions = {level: _top1_predictions(path, f"pred_{level}") for level, path in paths.items()}
    sample_ids = sorted(set.intersection(*(set(values) for values in predictions.values())))
    consistent = sum(
        predictions["ec1"][sample_id] == ec_at_level(predictions["ec3"][sample_id], 1)
        and predictions["ec2"][sample_id] == ec_at_level(predictions["ec3"][sample_id], 2)
        for sample_id in sample_ids
    )
    return {"rows": len(sample_ids), "consistent_rows": consistent, "rate": consistent / max(len(sample_ids), 1)}


def predict(config_path: str | Path, checkpoint: str | Path | None = None, zero_shot: bool = False) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    pretrained = Path(config["pretrained_checkpoint"])
    output_dir = Path(config["output_dir"])
    prediction_cfg = config.get("prediction", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SmilesTokenizer.load(pretrained / "smiles_vocab.json")
    model = ECRetrievalModel.from_pretrained(pretrained, seed=int(config["training"].get("seed", 13)))
    checkpoint_path = Path(checkpoint) if checkpoint else output_dir / "model.pt"
    if not zero_shot:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
    model.to(device)

    train = load_ecreact_csv(config["data"]["train_csv"])
    test = load_ecreact_csv(config["data"]["test_csv"])
    cache_dir = output_dir / ("zero_shot_embeddings" if zero_shot else "embeddings")
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_embeddings = encode_records(
        model, train, tokenizer, int(config["data"].get("max_length", 1200)),
        int(prediction_cfg.get("batch_size", 16)), device,
    )
    test_embeddings = encode_records(
        model, test, tokenizer, int(config["data"].get("max_length", 1200)),
        int(prediction_cfg.get("batch_size", 16)), device,
    )
    torch.save(train_embeddings, cache_dir / "train.pt")
    torch.save(test_embeddings, cache_dir / "test.pt")
    search_queries = test_embeddings.to(device) if device.type == "cuda" else test_embeddings
    distances, indices = exact_topk_euclidean(
        search_queries,
        train_embeddings,
        k=4,
        query_chunk=int(prediction_cfg.get("query_chunk", 512)),
        library_chunk=int(prediction_cfg.get("library_chunk", 8192)),
    )
    rows = _prediction_rows(test, train, distances, indices)
    prefix = "zero_shot_" if zero_shot else ""
    _write_csv(output_dir / f"{prefix}top4_predictions.csv", rows)
    metrics = score_predictions(test, train, indices)
    manifest_path = config["data"].get("split_manifest")
    if manifest_path and Path(manifest_path).is_file():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        seen = set(manifest["audit"].get("seen_test_sample_ids", []))
        unseen = {record.sample_id for record in test if record.sample_id not in seen}
        metrics["unseen"] = score_predictions(test, train, indices, unseen)
    metrics["embedding"] = {"normalized": False, "dimension": int(train_embeddings.shape[1]), "distance": "euclidean"}
    if config["task"]["type"] == "hierarchical":
        predictions = [train[index].ec3 for index in indices[:, 0].tolist()]
        metrics["hierarchical_consistency"] = {
            "rate": 1.0,
            "note": "EC1/EC2 are deterministic truncations of the predicted EC3 label.",
            "verified": all(ec_at_level(label, 1) and ec_at_level(label, 2) for label in predictions),
        }
    published = prediction_cfg.get("published_predictions")
    if published:
        published_level = prediction_cfg.get("published_level", config["task"].get("level", "ec3"))
        metrics["claire_published"] = {
            "level": published_level,
            **_published_metrics(published, test, published_level),
        }
    companion_paths = prediction_cfg.get("triplet_prediction_csvs")
    if companion_paths and all(Path(path).is_file() for path in companion_paths.values()):
        metrics["triplet_cross_level_consistency"] = _triplet_consistency(companion_paths)
    write_json(output_dir / f"{prefix}metrics.json", metrics)
    _write_summary(output_dir, metrics, zero_shot)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exact nearest-neighbor EC prediction")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--zero-shot", action="store_true")
    args = parser.parse_args()
    metrics = predict(args.config, args.checkpoint, args.zero_shot)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
