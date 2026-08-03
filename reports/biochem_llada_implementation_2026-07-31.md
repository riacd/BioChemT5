# BioChemLLaDA implementation and smoke validation

## Scope

This change adds a randomly initialized BioChemLLaDA model family while preserving the existing BioChemT5 path. The production architecture has 24 bidirectional transformer layers, hidden size 768, 12 attention heads, SwiGLU width 3072, RMSNorm, RoPE, tied embeddings, and 2,048-token context. Its exact parameter count is 227,553,285.

The shared tokenizer keeps every existing 301 token ID unchanged and appends `<mask>` as ID 301 only for LLaDA checkpoints. T5 checkpoints remain loadable without metadata; checkpoint family detection falls back to the presence of `t5/` or `llada/`.

## Objective and generation

Pretraining samples forward, retro, and MLM tasks with probabilities 0.35, 0.35, and 0.30. Each sample draws `t` uniformly from `[0.001, 1]`. Tokens are replaced only inside their task-specific noiseable region, and masked-token cross entropy is corrected by each token's actual masking probability before per-sample target normalization.

Reaction-center masking uses base, neighbor, and center weights 1, 2, and 4. A scalar is solved so clipped weighted probabilities retain mean `t`. Conditional targets use the smallest bucket in 64, 128, 256, 512, and 768; EOS fills the remaining canvas and targets above 768 are truncated with a final EOS.

Diffusion generation uses prompt-only length prediction, 64 denoising steps, low-confidence remasking, temperature 0.5, ten batched candidates by default, commit-time mean log probability ranking, canonical SMILES deduplication, and one retry with the next larger bucket if all candidates lack EOS or fail canonicalization.

## Production configs

- `configs/training/llada_220m_uniform_200k.yaml`
- `configs/training/llada_220m_reaction_center_200k.yaml`
- `configs/ablations/llada_tiny_smoke.yaml`
- `configs/ablations/llada_tiny_ddp_smoke.yaml`

Both production configs use the BioChem Bench-clean corpus, global batch 256 on eight GPUs, AdamW with learning rate 2e-4 and betas 0.9/0.95, 5,000 warmup steps, cosine decay to 2e-5, bf16, gradient checkpointing, a fixed 100k checkpoint, validation every 1,000 steps, and early stopping after 20 non-improving validations.

Formal monitoring now records the raw diffusion loss, raw and weighted length loss, cumulative task and length-bucket counts, target/prompt/MLM truncation rates, requested and realized mask rates, and separate base/neighbor/center token probabilities. The cumulative counters are saved in checkpoint metrics and continue across resume. Validation reports the same loss components and diffusion monitoring with its fixed random seed. When EC contrastive loss is disabled, both T5 and LLaDA collators skip EC view tokenization entirely.

## Verification

CPU worker:

- 227,553,285 production parameters verified on a meta device.
- Tiny four-step pretraining completed with checkpoint save and reload.
- 83 executable repository tests passed when excluding the stale `test_ablation_configs.py` file.
- Full suite result: 87 passed and 2 failed. The remaining failures predate LLaDA: one test references a user-deleted EC-only config, and one assumes the documentation directory contains only a single file.

GPU worker, NVIDIA H200:

- Single-GPU diffusion generation produced finite scores and monotonically reduced the mask count to zero.
- Eight-GPU tiny DDP completed one optimizer step with `world_size=8`.
- Eight-GPU resume restored step 1 from the saved model, optimizer, and scheduler state.
- Forward prediction, retrosynthesis, and hierarchical EC retrieval each completed an end-to-end smoke using rows sampled from their original train, validation, and test files.

The reproducible benchmark driver is `tests/run_llada_benchmark_smoke.py`; generated artifacts and the machine-readable summary are under `outputs/BiochemT5/benchmark/llada_smoke/`.

Smoke metrics from the multi-label rerun:

| Task | Rows | Functional result |
| --- | ---: | --- |
| Forward generation | 4 official-split rows | top-1/top-10 exact match 0/0; validity 0% |
| Retrosynthesis | 4 official-split rows | top-1/top-10 exact match 0/0; validity 0% |
| EC1 retrieval | 16 official-split rows | accuracy 62.50%; macro-F1 61.90% |
| EC2 retrieval | 16 official-split rows | accuracy 43.75%; macro-F1 44.44% |
| EC3 retrieval | 16 official-split rows | accuracy 43.75%; macro-F1 37.50% |

The EC sample contains eight EC3 labels selected with a fixed seed. It is more useful as an interface and metric smoke than the original contiguous-prefix sample, which contained only one label.

## Metric interpretation

The LLaDA benchmark run is a functional smoke from a tiny random checkpoint with one downstream update. Its generation accuracy and validity are not performance estimates and must not be compared directly with the trained T5-base results.

For reference, the maintained T5 report records retrosynthesis top-1/top-10 exact match of 41.11%/62.01% on the full-data run, and EC hierarchical accuracy of 96.07%, 92.64%, and 88.32% for EC1, EC2, and EC3. Formal LLaDA values should be added beside those figures only after a production pretraining and downstream fine-tuning run.
