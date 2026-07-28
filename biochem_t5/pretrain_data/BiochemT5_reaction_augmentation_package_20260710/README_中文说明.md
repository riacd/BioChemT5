# BiochemT5 生化反应数据增强代码包

本包用于复现当前项目中 BiochemT5 预训练语料的数据增强流程。它包含聚类中心反应抽取、RXNMapper atom-map、R-SMILES 最多 20 倍增强、reaction center 原子信息补齐、R-SMILES view 去重与最终语料校验相关代码。

本包只包含代码、小样例和统计文件，不包含 60GB 级别的完整 JSONL 训练语料。

## 目录结构

```text
BiochemT5_reaction_augmentation_package_20260710/
  README_中文说明.md
  MANIFEST.md
  requirements_augmentation.txt
  env/
    environment.biochem_t5_py311.yml
  scripts/
    run_augmentation_pipeline.sh
    original_h200_scripts/
  src/
    biochem_t5/data/
      rsmiles_augmentation.py
      reaction_center.py
      finalize_unique_rsmiles.py
      validate_pretrain_corpus.py
      smiles_tokenizer.py
      span_masking.py
  third_party/
    Rsmiles-main/
  examples/
    smoke/
    full_stats/
  tests/
```

## 输入数据要求

### 1. 原始反应 JSONL

`rsmiles_augmentation.py build-centers` 需要输入原始 reaction JSONL。每行是一个 JSON object，至少应包含：

```json
{
  "rxn": "reactants>>products",
  "ecs": ["1.1.1.1"],
  "template_ids": ["..."]
}
```

其中 `rxn` 可以是 `reactants>>products`，也兼容三段式 `reactants>agents>products`。

### 2. 聚类 assignment TSV.GZ

用于从全量反应中抽取聚类中心。TSV header 必须包含：

```text
cluster_id
center_record_id
```

`center_record_id` 对应原始 JSONL 的 1-based 行号。

## 输出数据字段

最终输出 JSONL 每条记录会包含或补齐以下字段：

```text
rxn_id
cluster_id
cluster_size
is_cluster_center
primary_template_id
primary_ec
ec_levels
mapped_rxn
rxnmapper_confidence
rsmiles_status
rsmiles_views
rsmiles_view_count
unique_rsmiles_view_count
reaction_center_version
reaction_center_source
reaction_center_status
reaction_center_changed_atom_maps
reaction_center_neighbor_atom_maps
reaction_center_atom_maps
reaction_center_changed_atom_count
reaction_center_neighbor_atom_count
reaction_center_atom_count
```

`rsmiles_views` 中每个 view 的结构为：

```json
{
  "aug_id": 0,
  "forward_input": "...",
  "forward_target": "...",
  "retro_input": "...",
  "retro_target": "..."
}
```

其中：

- `forward_input` / `forward_target` 用于正向反应预测；
- `retro_input` / `retro_target` 用于逆向反应预测；
- `mapped_rxn` 和 reaction center 字段用于 T5 span mask MLM 的中心原子加权 mask；
- `ec_levels` 用于 EC 层级对比学习采样。

## 环境依赖

建议使用项目中已有 conda 环境：

```bash
/mnt/shared-storage-user/tanpan/miniconda3/envs/vplm/bin/python
```

或根据 `env/environment.biochem_t5_py311.yml` 创建环境。

核心依赖：

```text
python>=3.10
rdkit
rxnmapper
torch
numpy
textdistance
pytest
```

注意：

- R-SMILES 原始源码在 `third_party/Rsmiles-main/`。
- `rsmiles_augmentation.py` 内部对 `textdistance` 做了 fallback；如果环境里没有 `textdistance`，会使用内置 Levenshtein fallback 满足 R-SMILES 源码调用。
- 全量增强推荐 CPU 多进程运行，不需要 GPU。

## 一键 pipeline 脚本

推荐使用：

```bash
bash scripts/run_augmentation_pipeline.sh
```

运行前用环境变量指定输入输出：

```bash
export PYTHON=/mnt/shared-storage-user/tanpan/miniconda3/envs/vplm/bin/python
export INPUT_JSONL=/path/to/reactions.jsonl
export ASSIGNMENTS_TSV_GZ=/path/to/final_assignments.tsv.gz
export OUT_DIR=/path/to/pretrain_corpus
export WORKERS=120
export CHUNK_SIZE=64
export CENTER_CHUNK_SIZE=512

bash scripts/run_augmentation_pipeline.sh
```

默认输出：

```text
$OUT_DIR/cluster_centers.sim095_r3_complete.jsonl
$OUT_DIR/cluster_centers.sim095_r3_complete.stats.json
$OUT_DIR/cluster_centers.rsmiles_aug20.jsonl
$OUT_DIR/cluster_centers.rsmiles_aug20.stats.json
$OUT_DIR/cluster_centers.rsmiles_aug20.centered.jsonl
$OUT_DIR/cluster_centers.rsmiles_aug20.centered.stats.json
$OUT_DIR/cluster_centers.rsmiles_unique20.centered.jsonl
$OUT_DIR/cluster_centers.rsmiles_unique20.centered.stats.json
$OUT_DIR/cluster_centers.rsmiles_unique20.centered.validation.json
```

## 分步运行命令

### Step 1：抽取聚类中心反应

```bash
PYTHONPATH=src \
python src/biochem_t5/data/rsmiles_augmentation.py build-centers \
  --input-jsonl "$INPUT_JSONL" \
  --assignments-tsv-gz "$ASSIGNMENTS_TSV_GZ" \
  --out-jsonl "$OUT_DIR/cluster_centers.sim095_r3_complete.jsonl" \
  --out-stats "$OUT_DIR/cluster_centers.sim095_r3_complete.stats.json" \
  --progress-every 1000000
```

该步骤会写入：

- `rxn_id`
- `cluster_id`
- `cluster_size`
- `is_cluster_center`
- `primary_template_id`
- `primary_ec`
- `ec_levels`

### Step 2：RXNMapper atom-map + R-SMILES 增强

```bash
PYTHONPATH=src \
python src/biochem_t5/data/rsmiles_augmentation.py augment \
  --input-jsonl "$OUT_DIR/cluster_centers.sim095_r3_complete.jsonl" \
  --out-jsonl "$OUT_DIR/cluster_centers.rsmiles_aug20.jsonl" \
  --out-stats "$OUT_DIR/cluster_centers.rsmiles_aug20.stats.json" \
  --rsmiles-repo third_party/Rsmiles-main \
  --augmentation 20 \
  --workers 120 \
  --chunk-size 64 \
  --map-batch-size 16 \
  --resume \
  --progress-every 1000
```

该步骤会：

1. 如果记录没有可用 `mapped_rxn`，先用 RXNMapper 标 atom-map；
2. 检查 atom-map 是否缺失、重复或不平衡；
3. 调用 R-SMILES 源码生成最多 20 个 forward/retro 对齐 view；
4. 写入 `rsmiles_status`、`rsmiles_views`、view 数量和 RXNMapper confidence。

`--augmentation` 大于 20 时会被裁剪到 20。

### Step 3：补齐 reaction center 原子信息

```bash
PYTHONPATH=src \
python src/biochem_t5/data/reaction_center.py \
  --input-jsonl "$OUT_DIR/cluster_centers.rsmiles_aug20.jsonl" \
  --out-jsonl "$OUT_DIR/cluster_centers.rsmiles_aug20.centered.jsonl" \
  --out-stats "$OUT_DIR/cluster_centers.rsmiles_aug20.centered.stats.json" \
  --workers 120 \
  --chunk-size 512 \
  --resume \
  --progress-every 100000
```

reaction center 的标注逻辑是：比较 mapped reaction 中反应物侧和产物侧同一 atom-map 的原子属性与键环境。若环境不同，则该 atom-map 是 changed center；changed center 的 mapped 邻居会写入 neighbor center。

### Step 4：R-SMILES views 去重

```bash
PYTHONPATH=src \
python src/biochem_t5/data/finalize_unique_rsmiles.py \
  --input-jsonl "$OUT_DIR/cluster_centers.rsmiles_aug20.centered.jsonl" \
  --out-jsonl "$OUT_DIR/cluster_centers.rsmiles_unique20.centered.jsonl" \
  --out-stats "$OUT_DIR/cluster_centers.rsmiles_unique20.centered.stats.json" \
  --max-views 20 \
  --progress-every 100000
```

去重 key 为：

```text
(forward_input, forward_target, retro_input, retro_target)
```

不会补齐到 20；一条反应最终有多少个唯一 view 就保留多少个，最多 20。

### Step 5：最终语料校验

```bash
PYTHONPATH=src \
python src/biochem_t5/data/validate_pretrain_corpus.py \
  --input-jsonl "$OUT_DIR/cluster_centers.rsmiles_unique20.centered.jsonl" \
  --out-json "$OUT_DIR/cluster_centers.rsmiles_unique20.centered.validation.json"
```

用于检查 JSONL 格式、关键字段、EC 层级、reaction center 和 R-SMILES view 结构。

## h200 原始脚本

`scripts/original_h200_scripts/` 保留了本项目实际跑全量任务时使用的原始脚本：

```text
run_rsmiles_aug20_h2002.sh
launch_rsmiles_aug20_h2002.sh
run_reaction_center_h2002.sh
launch_reaction_center_h2002.sh
run_finalize_unique_rsmiles_h2001.sh
launch_finalize_unique_rsmiles_h2001.sh
```

这些脚本包含原项目绝对路径，适合在当前集群目录复现；如果把包移动到其他目录，请优先使用 `scripts/run_augmentation_pipeline.sh`。

## 当前全量运行结果摘要

统计文件放在 `examples/full_stats/`。当前项目已完成的全量数据增强结果：

```text
cluster center records: 4,168,335
R-SMILES ok records: 3,773,931
RXNMapper failed records: 394,175
R-SMILES failed records: 229
total R-SMILES views: 75,478,620
unique R-SMILES views: 75,419,255
```

最终用于预训练的语料文件为：

```text
data/BiochemT5/pretrain_corpus/cluster_centers.rsmiles_unique20.centered.jsonl
```

它没有被打包进本交付包，因为大小约 60GB。

## 小样例

`examples/smoke/` 包含小规模输入和输出样例，可以用来检查字段结构。完整全量增强不建议在登录节点运行，应在 CPU 核心充足的计算节点上运行。

## 测试

在包根目录运行：

```bash
PYTHONPATH=src pytest tests -q
```

当前附带的测试覆盖：

- R-SMILES view 去重逻辑；
- reaction center 加权 mask 相关工具；
- SMILES tokenizer 对 atom-map 去除和特殊 token 的处理。

