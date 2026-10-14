---
name: vla-action-labels
description: 设计并构建 VLA 动作块标签：让 H 个点覆盖 span 个录制帧（按时间等步长或按轨迹进度取点），用网格搜索按服务端实际执行的目标与录制路径的几何差异选择参数，用闭环标签回放选执行长度，再构建带留出集的训练版和评估版数据集。属于 vla-pipeline 的 S7。
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
    --horizon <H> --metrics time curvature --spans <H> ... --event-density 0 1 2 \
    --head <计划的执行长度> [--max-speed-ratio <比例>] [--max-jump <运控上限，路径步长>] [--max-pad <比例>] \
    --out runs/<task>/sweep.json
```

- 网格必须包含 `time` 度量（不需要猜时间的对照）和 `span = H` 的格子（逐帧基线）。`second_difference` 只用于复现用它构建的数据集。
- 各列都是在服务端会执行的结果上测得的：标签点按服务端假定的时间放置，插值成逐帧目标，再与录制路径比较几何。静态任务里沿路径的快慢不影响结果，所以不按逐帧时间对齐打分；距离以路径步长为单位。列的定义见 [方法](references/method.md)的“打分”一节。
- **可接受**：前 head 个目标的路径偏差中位 ≤ 1 路径步长，关键帧位置误差中位 ≤ 1 路径步长，所有指令边沿都保留；给定时还要满足速度比、首点跳变和补帧点比例的上限。
- **选择**：先取有可接受格子的最大 span，再在同一 span 内取整段路径偏差中位最小的格子。脚本另外输出每个 span 的最佳格子作为候选清单；进入训练对比的是候选清单，而不是单个最优。
- 同一 span 内各格的差异远小于 1 路径步长时，选择结果只反映数值噪声：如实报告，必要时把不同度量都放进候选清单。
- 几何度量下补帧行和静止帧不增加进度，同一 span 覆盖的录制帧比 `time` 少；跨度量比较时看 `reach` 列。
- `sweep.json` 中按时间对齐的误差和边沿时间偏移只作参考，不参与可接受和选择。

### 2. 执行长度

```bash
python -m vla_tools.label_replay --dataset data/<src>-timed --spec robot_spec.json --keyframes runs/<task>/keyframes.json \
    --config runs/<task>/label_config.json --execute 2 4 6 8 10 20 --out runs/<task>/replay_<labels>.json
```

把标签当成完美模型的输出闭环执行：取当前观测帧的标签，按服务端规则插值，执行前 e 帧，再在录制中重新观测离执行位姿最近、指令状态相同的帧（只向前找），如此拼接到 episode 结束。表中给出路径偏差、接缝、关键帧位置误差、漏掉的指令、指令切换处的停留帧数、速度比、推理次数和执行帧 / 录制帧，是该配置下任何模型都去不掉的误差下限。选择路径偏差、关键帧位置误差都不随 e 明显增大、且不漏指令的最长执行长度，再结合任务对反应速度的要求确定。LIBERO 上执行到整个 chunk（e = 40）误差也不增大，这时 e 由模型需要多久重新观测一次决定。`time`、`span = H` 时路径偏差为 0，执行长度只受模型行为约束。

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

## LIBERO 上的结果（参考量级）

LIBERO `pick up the alphabet soup and place it in the basket`，50 集，20 Hz，H = 10，head 5，span 10–40，密度 0 / 1 / 2，补帧 40 行，2026-10-13：

- `time` 与 `curvature` 的 30 个格子全部可接受；路径偏差中位 ≤ 0.17、p90 ≤ 0.32 路径步长（0.0306 rad），没有漏掉的指令；
- 候选清单：span 10、15 为 `time`、密度 0；span 20、30、40 为 `curvature`、密度 2。选定 `curvature` span 40 密度 2：整段路径偏差 0.06 / 0.13，关键帧位置 0.30，速度比 p95 0.69，覆盖帧数中位 26.5；
- 加入 `second_difference` 的 45 格网格中 span 40 选中 `second_difference`，它与 `curvature` 的整段路径偏差中位只差 0.0003 路径步长；
- 闭环回放（`curvature` span 40 密度 2）：e = 2–40 时路径偏差中位 0.01–0.04、最大 0.43–0.46，关键帧位置中位 0.19–0.66 路径步长，没有漏掉的指令，执行帧数是录制帧数的 1.45–1.58 倍；`time` span 10 基线路径偏差 0；
- 同一选定配置按时间对齐的整段误差中位 12.2 帧运动量（点放在真实时间时 0.08）：点的时间由服务端假定，路径仍然跟上。

## 停止条件

- 没有可接受的格子：降低 span、换度量或密度，或放宽某个上限并写明理由，不要直接改阈值；
- 运控跳变上限未知，却需要在超过 H 的 span 之间做选择：只交付候选清单。

## 完成标准

- `sweep.json`（一张表加候选清单）、每个候选的闭环回放表与选定的执行长度、每个候选与基线的训练版和评估版数据集，都附带 `arc_resample.json`；
- 留出集与 `diversity.json` 一致。
