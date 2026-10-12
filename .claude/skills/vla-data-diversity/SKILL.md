---
name: vla-data-diversity
description: 评估示教数据的多样性与覆盖范围：起始姿态、关键帧处构型（物体与放置位置的代理）、静止维姿态（批次/装置差异）、时长与路径、首帧画面；找出聚类、近重复与离群 episode，给出覆盖各簇的留出集，并在模型评估时按"离训练数据多远"解读误差。属于 vla-pipeline 的 S6，结果也供 S8 下游评估使用。
---

# 数据评估：多样性与覆盖

## 回答什么问题

- 数据覆盖了哪些情况：物体摆在哪些位置、手臂从哪里出发、场景和光照怎么变；
- 数据是否分成几批：不同机位、不同采集批次、不同装置状态；
- 有没有几乎重复的 episode，或者孤立的 episode；
- 留出集该怎么选，才能如实回答"模型对没见过的情况好不好"；
- 模型误差是否随"离训练数据的距离"增大。

这些问题决定了评估结论能说到什么程度。比如物体总放在同一区域时，留出集上误差低，只说明模型在这一区域可用。

## 输入

定时版数据集、定稿的 `keyframes.json`、`robot_spec.json`。工具：`vla_tools.diversity`、`vla_tools.coverage_eval`。

## 描述量

| 组 | 含义 |
|---|---|
| `start_pose` | 第 0 帧的关节状态（不含静止维） |
| `pose@<关键帧>` | 每集恰好出现一次的关键帧处的关节状态；抓取处代表物体位置，释放处代表放置位置 |
| `static_pose` | 静止维的平均姿态；不同取值通常对应不同批次或装置状态 |
| `motion` | 真实帧数（不含补帧）与关节路径长度 |
| `image:<相机>` | 首帧缩略图（灰度、标准化）：场景布局、机位、光照 |

## 步骤

```bash
python -m vla_tools.diversity --dataset data/<src>-timed --spec robot_spec.json --keyframes runs/<task>/keyframes.json \
    --cameras <俯视相机列> --holdout <N> --strategy stratified \
    --out runs/<task>/diversity.json --plot runs/<task>/diversity.png --montage runs/<task>/diversity_montage.png
```

1. **逐组读结果**：离散程度、簇的数量与大小、最孤立的几集。
2. **看图解释每一个簇**：在首帧拼图上确认每个簇在画面上对应什么（机位、摆放、批次）。不能解释的簇，不能用来做划分依据。
3. **处理近重复与离群**：近重复的 episode 只保留一份，或者确保它们不分别落在训练集和留出集里；离群的 episode 要判断是数据问题（例如机位变了）还是有价值的少数情况，结论写回 `vla-data-cleaning` 的排除列表或保留理由。
4. **选留出集**：
   - `stratified`：按综合簇的比例从每簇抽取，每簇至少留一集在训练集，用于衡量分布内的泛化；
   - `extrapolation`：取最孤立的几集，用于衡量模型在数据边缘的表现。
   工具会列出留出后没有训练样本的簇；出现这种情况时，换策略或调整留出数量。
5. **记录覆盖结论**：数据覆盖了什么、没覆盖什么，写进 `docs/`，供评估报告引用。

## 在模型评估中使用

下游评估在留出集上得到每个模型的 `eval_<model>.json` 后：

```bash
python -m vla_tools.coverage_eval --diversity runs/<task>/diversity.json --eval baseline=eval_a.json candidate=eval_b.json
```

逐集列出"到最近训练 episode 的距离"与拼接误差、指令偏移，并给出距离与误差的 Spearman 相关。留出集只有几集时，相关系数只是弱证据，主要看逐集表。

## UR3 上的观测（参考量级）

22 集中：
- 静止维姿态分为两簇（14 / 8）；
- 首帧画面分为三簇：第 0 集机位不同，单独一簇；第 16–21 集是另一个批次，摆放偏右；其余 15 集一簇；
- 方块总在盘子右侧的同一区域；
- 没有近重复。

分层留出 4 集（3、4、11、19），它们到最近训练集的距离是中位最近邻距离的 0.72–1.37 倍。

## 停止条件

- 某个簇无法从画面或数据上解释；
- 留出集无法覆盖主要的簇（数据太少或太集中）：如实报告，不给泛化结论。

## 完成标准

- `diversity.json`、分布图、首帧拼图都存在，每个簇都有解释；
- 留出集及其理由已写入，并传给 `vla-action-labels` 的 `--exclude`；
- 覆盖结论写入 `docs/`。
