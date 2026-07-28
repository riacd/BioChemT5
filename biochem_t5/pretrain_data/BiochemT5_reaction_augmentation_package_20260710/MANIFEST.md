# 打包清单

## 核心源码

- `src/biochem_t5/data/rsmiles_augmentation.py`：聚类中心抽取、RXNMapper atom-map、R-SMILES 增强。
- `src/biochem_t5/data/reaction_center.py`：基于 mapped reaction 的反应中心原子推断。
- `src/biochem_t5/data/finalize_unique_rsmiles.py`：R-SMILES views 去重，每条最多保留 20 个唯一 view。
- `src/biochem_t5/data/validate_pretrain_corpus.py`：最终语料结构校验。
- `src/biochem_t5/data/smiles_tokenizer.py`、`span_masking.py`：随包提供，便于测试和解释 reaction center 加权 MLM 相关字段。

## 运行脚本

- `scripts/run_augmentation_pipeline.sh`：可移动的一键 pipeline 脚本。
- `scripts/original_h200_scripts/`：原项目 h200 节点运行脚本，保留绝对路径。

## 第三方源码

- `third_party/Rsmiles-main/`：Root-aligned SMILES 官方源码副本，保留原 LICENSE 和 README。

## 示例与统计

- `examples/smoke/`：小样例输入/输出。
- `examples/full_stats/`：当前项目全量增强的统计 JSON，不含大体积 JSONL。

## 环境

- `env/environment.biochem_t5_py311.yml`：当前项目使用过的 conda 环境导出文件。
- `requirements_augmentation.txt`：增强流程的最小依赖提示。

## 未包含的大文件

以下文件因体积过大未打包：

- `data/BiochemT5/pretrain_corpus/cluster_centers.sim095_r3_complete.jsonl`
- `data/BiochemT5/pretrain_corpus/cluster_centers.rsmiles_aug20.jsonl`
- `data/BiochemT5/pretrain_corpus/cluster_centers.rsmiles_aug20.centered.jsonl`
- `data/BiochemT5/pretrain_corpus/cluster_centers.rsmiles_unique20.centered.jsonl`
- `data/BiochemT5/pretrain_corpus/cluster_centers.rsmiles_unique20.centered.index.pkl`

