# BioChemT5

Trainable BioChemT5 ablation repository using Hugging Face PyTorch T5 with chemistry-specific data handling.

Quick smoke run:

```bash
python -m biochem_t5.train_pretrain --config configs/ablations/span_uniform_only.yaml
```

8-GPU run:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m biochem_t5.train_pretrain \
  --config configs/ablations/forward_backward_span.yaml
```

Core docs:

- [BioChemT5 pretraining report](docs/BiochemT5_pretrianing_report.md)
- [BioChem Bench split pretraining](docs/BioChemT5_biochem_bench_split_pretraining.md)
