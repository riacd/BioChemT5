from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_cosine_schedule_with_warmup

from biochem_t5.data.smiles_tokenizer import SmilesTokenizer
from biochem_t5.losses.ec_retrieval import gather_with_gradient, hierarchical_loss
from biochem_t5.models.ec_retrieval import ECRetrievalModel

from .ec_metrics import classification_metrics, exact_topk_euclidean
from .ecreact import (
    HierarchicalBatchSampler,
    ReactionCollator,
    ReactionDataset,
    TripletCollator,
    TripletDataset,
    load_ecreact_csv,
    write_json,
)


def _fallback_counts(loader: DataLoader) -> dict[str, int]:
    sampler = getattr(loader, "batch_sampler", None)
    counts = getattr(sampler, "fallback_counts", None)
    return dict(counts) if counts is not None else {}


def _distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    return rank, local_rank, world_size


def _unwrap(model: torch.nn.Module) -> ECRetrievalModel:
    return model.module if isinstance(model, DistributedDataParallel) else model  # type: ignore[return-value]


def _labels_all_ranks(local: dict[str, list[str]]) -> dict[str, list[str]]:
    if not dist.is_initialized():
        return local
    gathered: list[dict[str, list[str]] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    return {level: sum((item[level] for item in gathered if item is not None), []) for level in local}


def _move(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return batch["input_ids"].to(device), batch["attention_mask"].to(device)


@torch.no_grad()
def _encode(
    model: ECRetrievalModel,
    records: Sequence,
    collator: ReactionCollator,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    loader = DataLoader(ReactionDataset(records), batch_size=batch_size, collate_fn=collator)
    model.eval()
    chunks = []
    for batch in loader:
        input_ids, attention_mask = _move(batch, device)
        chunks.append(model(input_ids, attention_mask).float().cpu())
    model.train()
    return torch.cat(chunks)


@torch.no_grad()
def _validate(
    model: ECRetrievalModel,
    library: Sequence,
    validation: Sequence,
    level: str,
    collator: ReactionCollator,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    library_embeddings = _encode(model, library, collator, device, batch_size)
    validation_embeddings = _encode(model, validation, collator, device, batch_size)
    search_queries = validation_embeddings.to(device) if device.type == "cuda" else validation_embeddings
    _distances, indices = exact_topk_euclidean(search_queries, library_embeddings, k=1)
    truth = [record.label(level) for record in validation]
    predictions = [library[index].label(level) for index in indices[:, 0].tolist()]
    return classification_metrics(truth, predictions)


def _make_loader(
    records: Sequence,
    config: dict[str, Any],
    tokenizer: SmilesTokenizer,
    rank: int,
    world_size: int,
    epoch_seed: int,
) -> DataLoader:
    training = config["training"]
    max_length = int(config["data"].get("max_length", 1200))
    workers = int(training.get("num_workers", 0))
    task = config["task"]["type"]
    if task == "triplet":
        dataset = TripletDataset(records, config["task"]["level"], seed=epoch_seed)
        sampler = DistributedSampler(dataset, world_size, rank, shuffle=True, seed=epoch_seed) if world_size > 1 else None
        return DataLoader(
            dataset,
            batch_size=int(training.get("per_device_triplets", 8)),
            sampler=sampler,
            shuffle=sampler is None,
            collate_fn=TripletCollator(tokenizer, max_length),
            num_workers=workers,
        )
    if task == "hierarchical":
        sampler = HierarchicalBatchSampler(
            records,
            batches_per_epoch=training.get("batches_per_epoch"),
            seed=epoch_seed + rank * 100_003,
        )
        return DataLoader(
            ReactionDataset(records), batch_sampler=sampler,
            collate_fn=ReactionCollator(tokenizer, max_length), num_workers=workers
        )
    raise ValueError(f"Unknown task type: {task}")


def _optimizer(model: ECRetrievalModel, config: dict[str, Any]):
    training = config["training"]
    return torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": float(training.get("encoder_lr", 2e-5))},
            {"params": model.projection.parameters(), "lr": float(training.get("projection_lr", 1e-4))},
        ],
        weight_decay=float(training.get("weight_decay", 0.01)),
    )


def _save(
    path: Path, model: torch.nn.Module, optimizer, scheduler, state: dict[str, Any], config: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **_unwrap(model).checkpoint_payload(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "state": state,
        "config": config,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(config_path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    rank, local_rank, world_size = _distributed()
    training = config["training"]
    seed = int(training.get("seed", 13))
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    pretrained = Path(config["pretrained_checkpoint"])
    output_dir = Path(config["output_dir"])
    tokenizer = SmilesTokenizer.load(pretrained / "smiles_vocab.json")
    records = load_ecreact_csv(config["data"]["train_csv"])
    manifest = json.loads(Path(config["data"]["split_manifest"]).read_text(encoding="utf-8"))
    selection_train = [records[index] for index in manifest["split"]["train_indices"]]
    validation = [records[index] for index in manifest["split"]["validation_indices"]]
    collator = ReactionCollator(tokenizer, int(config["data"].get("max_length", 1200)))
    level = config["task"].get("level", "ec3")
    selection_level = level if config["task"]["type"] == "triplet" else "ec3"
    max_epochs = int(training.get("max_epochs", 10))
    patience = int(training.get("patience", 2))
    bf16 = bool(training.get("bf16", True)) and device.type == "cuda"
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else contextlib.nullcontext()

    def new_model() -> ECRetrievalModel:
        model = ECRetrievalModel.from_pretrained(pretrained, seed=seed)
        if bool(training.get("gradient_checkpointing", True)):
            model.encoder.gradient_checkpointing_enable()
        model.to(device)
        return model

    model = new_model()
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
    loader = _make_loader(selection_train, config, tokenizer, rank, world_size, seed)
    optimizer = _optimizer(_unwrap(model), config)
    total_steps = max(1, len(loader) * max_epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(total_steps * 0.05), total_steps)
    state: dict[str, Any] = {"phase": "selection", "epoch": 0, "best_epoch": 0, "best_waf1": -1.0, "bad_epochs": 0, "history": []}
    resume = training.get("resume_from")
    resume_checkpoint = None
    if resume:
        resume_checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        state.update(resume_checkpoint["state"])
        if state["phase"] == "complete":
            return state
        if state["phase"] == "selection":
            _unwrap(model).load_state_dict(resume_checkpoint["model"])
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
            scheduler.load_state_dict(resume_checkpoint["scheduler"])

    triplet_loss = torch.nn.TripletMarginLoss(margin=float(config["task"].get("margin", 1.0)))
    selection_start = int(state["epoch"]) if state["phase"] == "selection" else max_epochs
    for epoch in range(selection_start, max_epochs):
        model.train()
        loss_sum = 0.0
        for batch in loader:
            input_ids, attention_mask = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast:
                embeddings = model(input_ids, attention_mask)
                if config["task"]["type"] == "triplet":
                    anchor, positive, negative = embeddings.reshape(batch["triplet_count"], 3, -1).unbind(1)
                    loss = triplet_loss(anchor, positive, negative)
                else:
                    embeddings = gather_with_gradient(embeddings)
                    local_labels = {
                        ec_level: [record.label(ec_level) for record in batch["records"]]
                        for ec_level in ("ec1", "ec2", "ec3")
                    }
                    loss = hierarchical_loss(
                        embeddings,
                        _labels_all_ranks(local_labels),
                        float(config["task"].get("temperature", 0.07)),
                        config["task"].get("weights"),
                    )
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("max_grad_norm", 1.0)))
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach())
        if dist.is_initialized():
            dist.barrier()
        metrics = None
        if rank == 0:
            metrics = _validate(
                _unwrap(model), selection_train, validation, selection_level, collator, device,
                int(training.get("validation_batch_size", 16)),
            )
        score = torch.tensor(metrics["weighted_f1"] if metrics else 0.0, device=device)
        if dist.is_initialized():
            dist.broadcast(score, 0)
        improved = float(score) > float(state["best_waf1"])
        state["epoch"] = epoch + 1
        state["history"].append({
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(len(loader), 1),
            "sampler_fallback_counts": _fallback_counts(loader),
            **(metrics or {}),
        })
        if improved:
            state.update({"best_epoch": epoch + 1, "best_waf1": float(score), "bad_epochs": 0})
        else:
            state["bad_epochs"] += 1
        if rank == 0:
            _save(output_dir / "latest.pt", model, optimizer, scheduler, state, config)
            if improved:
                _save(output_dir / "best_selection.pt", model, optimizer, scheduler, state, config)
        if int(state["bad_epochs"]) >= patience:
            break

    best_epoch_tensor = torch.tensor(int(state["best_epoch"]), device=device)
    if dist.is_initialized():
        dist.broadcast(best_epoch_tensor, 0)
        dist.barrier()
    best_epoch = int(best_epoch_tensor)
    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = new_model()
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
    full_loader = _make_loader(records, config, tokenizer, rank, world_size, seed + 1_000_000)
    optimizer = _optimizer(_unwrap(model), config)
    full_steps = max(1, len(full_loader) * best_epoch)
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(full_steps * 0.05), full_steps)
    final_history = list(state.get("final_history", []))
    final_start = 0
    if resume_checkpoint is not None and state["phase"] == "final":
        _unwrap(model).load_state_dict(resume_checkpoint["model"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        scheduler.load_state_dict(resume_checkpoint["scheduler"])
        final_start = int(state.get("final_epoch", 0))
    for epoch in range(final_start, best_epoch):
        model.train()
        loss_sum = 0.0
        for batch in full_loader:
            input_ids, attention_mask = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast:
                embeddings = model(input_ids, attention_mask)
                if config["task"]["type"] == "triplet":
                    anchor, positive, negative = embeddings.reshape(batch["triplet_count"], 3, -1).unbind(1)
                    loss = triplet_loss(anchor, positive, negative)
                else:
                    embeddings = gather_with_gradient(embeddings)
                    labels = {key: [record.label(key) for record in batch["records"]] for key in ("ec1", "ec2", "ec3")}
                    loss = hierarchical_loss(embeddings, _labels_all_ranks(labels), float(config["task"].get("temperature", 0.07)), config["task"].get("weights"))
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("max_grad_norm", 1.0)))
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach())
        final_history.append({
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(len(full_loader), 1),
            "sampler_fallback_counts": _fallback_counts(full_loader),
        })
        state.update({"phase": "final", "final_epoch": epoch + 1, "final_history": final_history})
        if rank == 0:
            _save(output_dir / "latest.pt", model, optimizer, scheduler, state, config)
    state.update({"phase": "complete", "final_history": final_history})
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        _save(output_dir / "model.pt", model, optimizer, scheduler, state, config)
        tokenizer.save(output_dir / "smiles_vocab.json")
        (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        write_json(output_dir / "training_metrics.json", state)
    if dist.is_initialized():
        dist.barrier()
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BioChemT5 for EC retrieval")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
