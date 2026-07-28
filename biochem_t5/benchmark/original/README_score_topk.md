# `score_topk.py` 使用说明

本文档说明以下脚本的输入格式、使用方法和输出内容：

- `/mnt/shared-storage-user/huangtianming/huangtianming/reaction_balancing/four_source_pipeline/outputs/data_all/score_topk.py`

## 脚本作用

`score_topk.py` 是一个统一的 retrosynthesis benchmark 评分脚本。

它假设每个模型已经输出了自己认为的 `top-k` 候选反应物，然后这个脚本只负责统一评测。

这个脚本不会对模型输出再次重排，也不会使用模型内部打分逻辑。它只做下面几件事：

- 规范化 SMILES
- 去除 atom mapping
- 去除重复预测
- 将预测结果与标准答案比较
- 计算 `Top-1` 到 `Top-k` 准确率

## 输入格式

推荐输入格式为 `CSV` 文件。

要求：

- 每一行对应一个测试样本
- 至少要有 `target` 和 `pred_1`

推荐列名：

- `id`
- `target`
- `pred_1`
- `pred_2`
- `pred_3`
- ...
- `pred_10`

示例：

```csv
id,target,pred_1,pred_2,pred_3,pred_4,pred_5,pred_6,pred_7,pred_8,pred_9,pred_10
0,CCO.CN,CCO.CN,CCN.O,CCOCN,,,,,,,
1,CCC.O,CCCO,CCC.O,CC.CCO,,,,,,,
```

含义：

- `id`：样本编号，可选
- `target`：该测试样本的标准答案反应物 SMILES
- `pred_1` 到 `pred_10`：模型输出的 top-10 候选反应物 SMILES

说明：

- 预测列允许为空
- 允许输入带 atom map 的 SMILES
- 允许分子片段顺序不同
- 脚本会先 canonicalize 再比较

## 也支持 JSONL

如果输入文件后缀是 `.jsonl`，脚本也可以读取。

每一行格式类似：

```json
{"id": 0, "target": "CCO.CN", "pred_1": "CCO.CN", "pred_2": "CCN.O"}
```

不过为了后续检查方便，更推荐使用 `CSV`。

## 基本使用方法

直接运行：

```bash
python /mnt/shared-storage-user/huangtianming/huangtianming/reaction_balancing/four_source_pipeline/outputs/data_all/score_topk.py \
  --input /path/to/predictions.csv
```

默认会读取这些列：

- `target`
- `pred_1`, `pred_2`, ..., `pred_10`

## 主要参数

- `--input`
  输入文件路径，支持 `CSV` 或 `JSONL`

- `--target-col`
  真值列名
  默认值：`target`

- `--id-col`
  样本编号列名
  默认值：`id`

- `--prediction-prefix`
  预测列名前缀
  默认值：`pred_`

- `--top-k`
  评测到多少个候选
  默认值：`10`

- `--save-details`
  可选。保存逐样本详细结果的输出路径

## 自定义列名示例

如果你的文件列名不是：

- `target`
- `pred_1`, `pred_2`, ...

而是：

- `ground_truth`
- `top_1`, `top_2`, ..., `top_10`

那么可以这样运行：

```bash
python /mnt/shared-storage-user/huangtianming/huangtianming/reaction_balancing/four_source_pipeline/outputs/data_all/score_topk.py \
  --input /path/to/predictions.csv \
  --target-col ground_truth \
  --prediction-prefix top_
```

## 输出内容

脚本默认把统计结果打印到终端。

示例：

```text
Samples: 1000
Prediction columns used: pred_1, pred_2, pred_3, pred_4, pred_5, pred_6, pred_7, pred_8, pred_9, pred_10
Invalid predictions skipped: 23
Top-1 Accuracy: 41.200% (412/1000)
Top-2 Accuracy: 55.700% (557/1000)
Top-3 Accuracy: 63.400% (634/1000)
Top-4 Accuracy: 68.000% (680/1000)
Top-5 Accuracy: 71.500% (715/1000)
Top-6 Accuracy: 74.100% (741/1000)
Top-7 Accuracy: 76.300% (763/1000)
Top-8 Accuracy: 78.200% (782/1000)
Top-9 Accuracy: 80.100% (801/1000)
Top-10 Accuracy: 81.900% (819/1000)
```

其中：

- `Samples`：测试样本总数
- `Prediction columns used`：实际参与计算的预测列
- `Invalid predictions skipped`：无效 SMILES 数量
- `Top-k Accuracy`：标准 top-k 准确率

## 保存逐样本详细结果

如果你传入 `--save-details`，脚本会额外生成一个 CSV 文件，记录每个样本的详细命中情况。

示例：

```bash
python /mnt/shared-storage-user/huangtianming/huangtianming/reaction_balancing/four_source_pipeline/outputs/data_all/score_topk.py \
  --input /path/to/predictions.csv \
  --save-details /path/to/predictions_scored.csv
```

输出文件会包含类似字段：

- `id`
- `target`
- `matched_rank`
- `top_1_hit`
- `top_2_hit`
- ...
- `top_10_hit`

含义：

- `matched_rank`：标准答案第一次出现在第几名
- `top_k_hit = 1`：说明标准答案在前 `k` 个候选中
- `top_k_hit = 0`：说明标准答案不在前 `k` 个候选中

## 比较规则

脚本在比较预测结果和真值之前，会做以下规范化：

1. 用 RDKit 解析 SMILES
2. 去除 atom mapping
3. canonicalize
4. 按 `.` 拆开分子片段后排序，再重新拼接

所以只要两个结果在化学上经过规范化后一致，就会被当成命中。

## 推荐 benchmark 工作流

建议对每个模型都统一这样处理：

1. 在测试集上跑推理
2. 把模型原始输出转换成统一表格格式：

```csv
id,target,pred_1,pred_2,...,pred_10
```

3. 用 `score_topk.py` 统一打分

这样你就可以用同一套评测脚本来比较不同模型，例如：

- RSGPT
- MEGAN
- LocalRetro
- R-SMILES

## 总结

这个脚本的输入应该是：

- 每个测试样本一行
- 一列真值 `target`
- 多列模型输出的 `top-k` 预测

这个脚本的输出是：

- `Top-1` 到 `Top-k` 的统一准确率
- 可选的逐样本详细命中结果
