# BioChemT5 实验结果汇总

**汇总日期：** 2026-07-29  
**范围：** 仓库中已保存的预训练、ECreact 分类、Biochem Bench 逆合成和 smoke/审计结果。

## 1. 执行摘要

- 已确认一条正式的 301-token 词表预训练链路：`forward_backward_span_t5_base_100k`，8 卡训练 100,000 steps，验证损失 **0.1093**。
- 在 ECreact 上，`hierarchical` 配置总体最好：EC1/EC2/EC3 accuracy 分别为 **96.07% / 92.64% / 88.32%**；未见类别子集上分别为 **97.74% / 95.10% / 92.39%**。
- 相对 CLAIRE published baseline，`hierarchical` 在 EC1/EC2/EC3 accuracy 上分别提升 **0.23/2.63/2.40 个百分点**；`triplet_ec3` 只在 EC3 上超过 CLAIRE（+0.92 个百分点）。
- 在 Biochem Bench 逆合成上，使用 full-data 划分的正式链路最好：beam-10 exact match 为 **41.11% / 48.78% / 51.46% / 54.89%**（top-1/3/5/10），测试产品数 5,478。
- 使用 updated 数据时，逆合成结果下降到 top-1 **29.31%**。该结果与 full-data 结果不能简单视为严格消融，因为数据划分同时发生了变化。

## 2. 数据与实验链路

### 2.1 预训练语料与词表

现有统计文件显示：

| 项目 | 数值 |
|---|---:|
| canonical reaction records | 4,168,335 |
| RSMILES records | 3,773,931 |
| RSMILES views | 75,478,620 |
| canonical 总 token 数 | 1,138,796,069 |
| formal vocabulary size | 301 |
| 词表截断后的 OOV token 数 | 0 |

来源：[pretrain_aug20_full_vocab_stats.json](/mnt/shared-storage-user/huyutong/BioChemT5/.tmp/pretrain_aug20_full_vocab_stats.json)。

正式预训练 checkpoint 元数据记录了 301-token 词表、4,168,335 条语料、8 卡和 100,000 steps：[checkpoint_metadata.json](/mnt/shared-storage-user/huyutong/BioChemT5/outputs/BiochemT5/production/forward_backward_span_t5_base_100k/checkpoint_metadata.json)。

### 2.2 Biochem Bench 数据规模

正式 full-data 处理后，train/val/test 分别为 **32,384/5,187/12,542 pairs**，对应 **23,700/4,179/5,478 unique products**；过滤后 train-val-test 的产品重叠为 0。来源：[full-data manifest](/mnt/shared-storage-user/huyutong/BioChemT5/outputs/BiochemT5/benchmark/biochem_bench_retrosynthesis_full_data/data_manifest.json)。

updated 数据使用较小的划分：train/val/test unique products 为 **27,401/5,079/4,988**，unique pairs 为 **51,503/7,872/7,783**。来源：[updated-data manifest](/mnt/shared-storage-user/huyutong/BioChemT5/outputs/BiochemT5/benchmark/biochem_bench_retrosynthesis_updated_data/data_manifest.json)。

## 3. 已保存的 100k 预训练结果

| 运行 | 训练步数 | 训练 loss | 验证 loss | 备注 |
|---|---:|---:|---:|---|
| `forward_backward_span_t5_base_100k` | 100,000 | 0.0503 | **0.1093** | formal 301-token vocabulary，8 卡 |
| `biochem_bench_split_20260728_v2/span_reaction_center_t5_base_100k` | 100,000 | 0.0276 | **0.0467** | split 数据链路，8 卡；缺少独立 checkpoint metadata |

正式 full-data 预训练的任务损失分解为 MLM **0.0430**、forward **0.0154**、retro **0.0797**。训练吞吐约 **25,692 tokens/s**，样本吞吐约 **218.4 samples/s**。来源：[formal train_metrics.json](/mnt/shared-storage-user/huyutong/BioChemT5/outputs/BiochemT5/production/forward_backward_span_t5_base_100k/train_metrics.json)。

## 4. ECreact 分类结果

测试集均为 **18,816 rows**；`unseen` 子集为 **15,155 rows**。下表列出各配置在三个 EC 层级上的 accuracy，括号内为 macro-F1。

| 配置 | EC1 | EC2 | EC3 | unseen EC1/EC2/EC3 |
|---|---:|---:|---:|---:|
| `hierarchical` | **96.07% (82.26%)** | **92.64% (78.97%)** | **88.32% (71.53%)** | **97.74% / 95.10% / 92.39%** |
| `triplet_ec1` | 95.58% (80.34%) | 87.79% (67.40%) | 82.46% (58.17%) | 97.20% / 89.14% / 85.12% |
| `triplet_ec2` | 95.55% (79.95%) | 91.49% (76.84%) | 85.04% (65.10%) | 97.14% / 93.76% / 88.32% |
| `triplet_ec3` | 95.60% (79.82%) | 92.08% (76.74%) | 86.85% (67.34%) | 97.23% / 94.46% / 90.64% |

补充指标：

- `hierarchical` 的 best weighted-F1 为 **0.8585**，第 3 个 epoch 取得最佳 checkpoint。
- `triplet_ec1/ec2/ec3` 的 best weighted-F1 分别为 **0.9586/0.9153/0.8494**，最佳 epoch 分别为 3/5/7。
- `triplet_ec3` 的跨层级标签一致率为 **93.51%**（17,595/18,816）。
- `hierarchical` 在 EC3 accuracy 上高于三种 triplet 配置；triplet 配置的目标层级性能随训练目标变化，但未形成对所有层级的统一优势。

详细文件：

- [hierarchical metrics](/mnt/shared-storage-user/huyutong/BioChemT5/outputs/BiochemT5/benchmark/ecreact_t5/hierarchical/seed_13/metrics.json)
- [triplet_ec1 metrics](/mnt/shared-storage-user/huyutong/BioChemT5/outputs/BiochemT5/benchmark/ecreact_t5/triplet_ec1/seed_13/metrics.json)
- [triplet_ec2 metrics](/mnt/shared-storage-user/huyutong/BioChemT5/outputs/BiochemT5/benchmark/ecreact_t5/triplet_ec2/seed_13/metrics.json)
- [triplet_ec3 metrics](/mnt/shared-storage-user/huyutong/BioChemT5/outputs/BiochemT5/benchmark/ecreact_t5/triplet_ec3/seed_13/metrics.json)

### 4.1 与 CLAIRE 的分层对比

下面的 `CLAIRE published` 使用论文配套的官方预测文件和标签，在当前 18,816-row 测试集上重算；EC3 weighted-F1 **0.8607** 与论文报告的 **0.861** 一致。EC1/EC2 数值是同一批官方预测按 EC 前缀截取得到，属于论文结果的分层重算，不是重新训练 CLAIRE。来源：[CLAIRE reproduction notes](/mnt/shared-storage-user/huyutong/BioChemT5/benchmark/ECreact_bench/REPRODUCTION.md)。

表中格式为 **accuracy / weighted-F1**，括号为相对 CLAIRE accuracy 的百分点差值。

#### EC1

| 方法 | Accuracy / weighted-F1 | 相对 CLAIRE |
|---|---:|---:|
| CLAIRE published | 95.84% / 95.80% | 基线 |
| BioChemT5 hierarchical | **96.07% / 96.00%** | **+0.23 pp** |
| BioChemT5 triplet_ec1 | 95.58% / 95.46% | -0.27 pp |
| BioChemT5 triplet_ec2 | 95.55% / 95.49% | -0.29 pp |
| BioChemT5 triplet_ec3 | 95.60% / 95.51% | -0.24 pp |

#### EC2

| 方法 | Accuracy / weighted-F1 | 相对 CLAIRE |
|---|---:|---:|
| CLAIRE published | 90.01% / 89.65% | 基线 |
| BioChemT5 hierarchical | **92.64% / 92.55%** | **+2.63 pp** |
| BioChemT5 triplet_ec1 | 87.79% / 87.63% | -2.22 pp |
| BioChemT5 triplet_ec2 | 91.49% / 91.41% | +1.47 pp |
| BioChemT5 triplet_ec3 | 92.08% / 91.94% | +2.06 pp |

#### EC3

| 方法 | Accuracy / weighted-F1 | 相对 CLAIRE |
|---|---:|---:|
| CLAIRE published | 85.93% / 86.07% | 基线 |
| BioChemT5 hierarchical | **88.32% / 88.04%** | **+2.40 pp** |
| BioChemT5 triplet_ec1 | 82.46% / 82.09% | -3.47 pp |
| BioChemT5 triplet_ec2 | 85.04% / 84.79% | -0.89 pp |
| BioChemT5 triplet_ec3 | 86.85% / 86.50% | **+0.92 pp** |

#### 对比结论

- `hierarchical` 是唯一在三个 EC 层级都超过 CLAIRE 的本地方法；优势在 EC2 和 EC3 最明显，分别为 **+2.63 pp** 和 **+2.40 pp accuracy**。
- `triplet_ec3` 在目标层级 EC3 上超过 CLAIRE，但在 EC1/EC2 上略低于 CLAIRE，说明单层级 triplet 目标不能稳定传递到所有层级。
- `triplet_ec2` 在 EC2 和 EC3 均优于 CLAIRE，但 EC1 略低；`triplet_ec1` 在三个层级均低于 CLAIRE。
- CLAIRE 论文与官方预测文件存在一个需要记录的口径差异：官方文件重算的跨层级一致率为 **90.02%**，论文报告为 **82.01%**。该差异不影响上表各层级 accuracy/weighted-F1，但不应把两者的一致率直接混用。

## 5. Biochem Bench 逆合成结果

### 5.1 Beam-10 测试结果

| 配置 | 使用的预训练模型 | 测试产品数 | exact top-1 | top-3 | top-5 | top-10 | largest-fragment top-10 | 无效 SMILES |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `full_data` | `forward_backward_span_t5_base_100k` | 5,478 | **41.11%** | **48.78%** | **51.46%** | **54.89%** | **62.01%** | 1.94% |
| `updated_data` | `forward_backward_span_t5_base_100k` | 4,988 | 29.31% | 38.38% | 41.53% | 45.46% | 57.33% | 3.28% |

full-data 模型的平均有效候选数为 **9.49**，候选唯一率 **96.80%**；updated-data 分别为 **9.45** 和 **97.76%**。full-data 结果由 formal 301-token 预训练 checkpoint 微调得到，来源元数据：[source_checkpoint_metadata.json](/mnt/shared-storage-user/huyutong/BioChemT5/outputs/BiochemT5/benchmark/biochem_bench_retrosynthesis_full_data/source_checkpoint_metadata.json)。

`full_data` 和 `updated_data` 使用同一个正式预训练模型 `outputs/BiochemT5/production/forward_backward_span_t5_base_100k`。该模型以 forward/retro/MLM = 0.35/0.35/0.30 的多任务目标训练 100,000 steps。

### 5.2 验证损失

| 配置 | 最佳验证 step | 最佳 val loss |
|---|---:|---:|
| `full_data` | 500 | 0.05836 |
| `updated_data` | 2,000 | 0.05833 |

逆合成详细指标：

- [full-data beam-10 metrics](/mnt/shared-storage-user/huyutong/BioChemT5/outputs/BiochemT5/benchmark/biochem_bench_retrosynthesis_full_data/test_beam10_metrics.json)
- [updated-data beam-10 metrics](/mnt/shared-storage-user/huyutong/BioChemT5/outputs/BiochemT5/benchmark/biochem_bench_retrosynthesis_updated_data/test_beam10_metrics.json)

## 6. Smoke、历史与异常记录

这些结果用于验证代码、数据接口或运行链路，不应与正式 benchmark 混合比较：

- ECreact GPU smoke：4 条测试样本，accuracy 为 25%，仅说明流程可运行。
- 逆合成 GPU smoke：1 个测试产品，exact-match top-1/top-10 均为 0；样本量不足以支持质量结论。

## 7. 结论与下一步

1. 当前最可信的主线是 **formal 301-token vocabulary + full-data pretraining + full-data Biochem Bench fine-tuning**；其逆合成 top-10 exact match 达到 **54.89%**。
2. ECreact 上 `hierarchical` 配置表现最稳定，尤其在 EC3 和 unseen 子集上领先，支持继续保留层级联合建模方向。
3. updated-data 逆合成结果明显低于 full-data，但存在数据规模和去重/过滤差异，下一轮应固定数据版本和 checkpoint 后再做严格 ablation。
4. 建议为所有正式 benchmark 保存统一的 `source_checkpoint_metadata.json`，以便复现实验 provenance。
5. 若用于论文表格，建议只纳入 formal 301-token 链路和明确记录数据 manifest 的结果；smoke 结果仅用于工程验证。
