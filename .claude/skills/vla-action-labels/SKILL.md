---
name: vla-action-labels
description: 设计并构建 VLA 动作块标签：让 H 个点覆盖 span 个录制帧（按时间等步长或按轨迹进度取点），用网格搜索按服务端实际执行的误差选择参数，用标签回放选执行长度，再构建带留出集的训练版和评估版数据集。属于 vla-pipeline 的 S7。
---

# 动作标签

## 触发与输入

想让策略一次预测得更远，或想改变监督点在轨迹上的分布时使用。不改标签时同样走这一步，按 `time`、`span = H` 构建：它与逐帧 chunk 完全相同，后续工具和比较口径保持一致。

输入：定时版数据集、定稿的 `keyframes.json`、`diversity.json` 中的留出集、`robot_spec.json`，以及模型的 `action_horizon`（H）和运控对首点跳变的上限。

工具：`vla_tools.sweep`、`vla_tools.label_replay`、`vla_tools.build_labels`。方法定义见 [方法](references/method.md)。

## 步骤

### 1. 网格搜索

按 `experiment` skill 的做法：先列出全部格子，再全部运行，最后用一张表报告。

```bash
python -m vla_tools.sweep --dataset data/<src>-timed --spec robot_spec.json --keyframes runs/<task>/keyframes.json \
    --horizon <H> --metrics time curvature arc --spans <H> ... --event-density 0 1 \
    --head <计划的执行长度> [--max-jump <运控上限>] [--max-pad <比例>] --out runs/<task>/sweep.json
```

- 网格必须包含 `time` 度量（不需要猜时间的对照）和 `span = H` 的格子（逐帧基线）。
- 各列都是在服务端会执行的结果上测得的：标签点按服务端假定的时间放置，插值成逐帧目标，再与录制比较。列的含义见工具说明。
- **可接受**：前 head 帧误差中位 ≤ 1 帧运动量，指令边沿偏移中位 ≤ 1 帧，所有边沿都保留；给定时还要满足跳变和补齐比例的上限。
- **选择**：先取最大的可接受 span，再在同一 span 内取整个 chunk 误差最小的格子。脚本另外输出每个 span 的最佳格子作为候选清单；进入训练对比的是候选清单，而不是单个最优。

### 2. 执行长度

```bash
python -m vla_tools.label_replay --dataset data/<src>-timed --spec robot_spec.json --keyframes runs/<task>/keyframes.json \
    --config runs/<task>/label_config.json --execute 2 4 6 8 10 20
```

把标签当成完美模型的输出：每 e 帧取一次标签，按服务端规则插值，执行前 e 帧后拼接，与录制比较。得到的是该配置下任何模型都去不掉的误差下限。选择误差和边沿偏移都不随 e 明显增大的最长执行长度，再结合任务对反应速度的要求确定。`time` 度量下放点时间是精确的，执行长度只受模型行为约束。

### 3. 构建

候选清单里的每个配置、以及逐帧基线，都各建一份训练版和一份评估版：

```bash
python -m vla_tools.build_labels --dataset data/<src>-timed --spec robot_spec.json --keyframes runs/<task>/keyframes.json \
    --config <config.json> --exclude <留出集> --out data/<src>-<labels>-train
python -m vla_tools.build_labels --dataset data/<src>-timed --spec robot_spec.json --keyframes runs/<task>/keyframes.json \
    --config <config.json> --out data/<src>-<labels>-eval
```

- 训练版去掉留出集，其余 episode 重新编号；评估版保留全部 episode，编号与定时版一致，留出的 episode 也带标签，可用于比较。
- 关键帧文件有开放问题，或者来自其他数据集版本时，工具拒绝构建。
- 构建结果会打印平均 loss 权重。权重大于 1 时有效学习率随之变大，与基线对比时要说明，或者相应调整学习率。
- `meta/arc_resample.json` 记录机器人布局、配置、关键帧文件和每集统计，训练和服务端都从这里读取参数。

## UR3 上的结果（参考量级）

22 集，H = 10，执行 6 帧：
- 候选清单全部是 `time` 度量；
- span 20 时前 6 帧误差 0.28、整个 chunk 0.43 帧运动量，指令偏移 0；
- 同 span 的 `curvature` 整个 chunk 误差约 2.4，`arc` 没有可接受的格子；
- `time`、span 20 的标签回放在 4–20 帧的执行长度下误差约 0.02 帧运动量。

## 停止条件

- 没有可接受的格子：降低 span、换度量，或放宽某个上限并写明理由，不要直接改阈值；
- 运控跳变上限未知，却需要在超过 H 的 span 之间做选择：只交付候选清单。

## 完成标准

- `sweep.json`（一张表加候选清单）、执行长度表、每个候选与基线的训练版和评估版数据集，都附带 `arc_resample.json`；
- 留出集与 `diversity.json` 一致。
