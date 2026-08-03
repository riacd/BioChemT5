# Benchmark Modules

Benchmark implementations are grouped by task:

- `ecreact/`: ECreact/CLAIRE data preparation, retrieval training, prediction, and metrics.
- `forward_prediction/`: Biochem Bench substrate-to-single-product fine-tuning, prediction, and scoring.
- `retrosynthesis/`: Biochem Bench retrosynthesis data loading, fine-tuning, prediction, and scoring.
- `common.py`: small file, JSON, and SMILES helpers shared by both benchmarks.

Retrosynthesis scoring adapts generated prediction JSONL files to a row-wise
`target` and `pred_1..pred_10` format before canonicalization, deduplication,
and top-k evaluation.

Use the benchmark-specific module paths for new commands:

```bash
python -m biochem_t5.benchmark.ecreact.prepare --help
python -m biochem_t5.benchmark.ecreact.train --help
python -m biochem_t5.benchmark.ecreact.predict --help

python -m biochem_t5.benchmark.retrosynthesis.train --help
python -m biochem_t5.benchmark.retrosynthesis.predict --help
python -m biochem_t5.benchmark.retrosynthesis.score --help

python -m biochem_t5.benchmark.forward_prediction.train --help
python -m biochem_t5.benchmark.forward_prediction.predict --help
python -m biochem_t5.benchmark.forward_prediction.score --help
```

The forward benchmark extends the checkpoint vocabulary append-only from the
configured benchmark splits, preserving every pretrained token ID. It uses `<forward_main>`
to distinguish single-product targets from the full product-side forward
pretraining objective, and applies the retrosynthesis scorer's canonical Top-k
exact-match logic unchanged.
