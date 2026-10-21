---
name: vla-action-labels
description: 设计并构建 VLA 动作块标签：在每帧之后 span 个录制帧的窗口内按时间或按几何曲率从录制路径上取 H 个点，部署时用三次样条从这 H 个点重建轨迹，用网格搜索按服务端实际执行的目标与录制路径的几何差异选择参数，用闭环标签回放和倍数上限（关节速度上限与推理延迟下）选执行长度和执行倍速，再构建带留出集的训练版和评估版数据集。属于 vla-pipeline 的 S7。
---

# 动作标签

## 触发与输入

想让策略一次预测得更远，或想改变监督点在轨迹上的分布时使用。不改标签时同样走这一步，按 `time`、`span = H` 构建：它与逐帧 chunk 完全相同，后续工具和比较口径保持一致。

输入：定时版数据集、定稿的 `keyframes.json`、`diversity.json` 中的留出集、`robot_spec.json`（选倍速和部署时需要 `joint_velocity_limits` 与 `switch_hold_s`，可选 `joint_acceleration_limits`），以及模型的 `action_horizon`（H）、部署的推理延迟和运控对首点跳变的上限。

工具：`vla_tools.sweep`、`vla_tools.label_replay`、`vla_tools.rate_ceiling`、`vla_tools.build_labels`。方法定义见 [方法](references/method.md)。

## 步骤

### 1. 网格搜索

按 `experiment` skill 的做法：先列出全部格子，再全部运行，最后用一张表报告。

```bash
python -m vla_tools.sweep --dataset data/<src>-timed --spec robot_spec.json --keyframes runs/<task>/keyframes.json \
    --horizon <H> --metrics time curvature --spans <H> ... --event-density 0 1 2 [--curvature-length <rad>] \
    --head <计划的执行长度> [--max-speed-ratio <比例>] [--max-jump <运控上限，路径步长>] [--max-pad <比例>] \
    --out runs/<task>/sweep.json
```

- 网格必须包含 `time` 度量（不需要猜时间的对照）和 `span = H` 的格子（逐帧基线）。`second_difference` 只用于复现用它构建的数据集。
- 各列都是在服务端会执行的结果上测得的：标签点按服务端假定的时间放置，插值成逐帧目标，再与录制路径比较几何。静态任务里沿路径的快慢不影响结果，所以不按逐帧时间对齐打分；距离以路径步长为单位。列的定义见 [方法](references/method.md)的“打分”一节。
- **可接受**：前 head 个目标的路径偏差中位 ≤ 1 路径步长，关键帧位置误差中位 ≤ 1 路径步长，所有指令边沿都保留；给定时还要满足速度比、首点跳变和补帧点比例的上限。
- **选择**：先取有可接受格子的最大 span，再在同一 span 内取整段路径偏差中位最小的格子。脚本另外输出每个 span 的最佳格子作为候选清单；进入训练对比的是候选清单，而不是单个最优。
- 同一 span 内各格的差异远小于 1 路径步长时，选择结果只反映数值噪声：如实报告，必要时把不同度量都放进候选清单。
- 每帧的步长只由它自己的窗口（当前状态之后 span 个录制帧）决定：窗口内没有关键帧时，第 H 个点恰好覆盖 span 个录制帧；关键帧加密（密度 > 1）让含关键帧的窗口覆盖不满 span，跨度量比较时看 `reach` 列。
- 曲率的弦长 `--curvature-length` 对整个数据集固定，默认 4 个路径步长并打印，写入每个格子的配置；它让曲率与示教速度无关。
- 打分按倍速 1 进行：执行倍速不改变插值出的关节路径，只改变走完它的帧数，速度比在闭环回放里按部署倍速测。
- `sweep.json` 中按时间对齐的误差和边沿时间偏移只作参考，不参与可接受和选择。

### 2. 执行长度与执行倍速

```bash
python -m vla_tools.label_replay --dataset data/<src>-timed --spec robot_spec.json --keyframes runs/<task>/keyframes.json \
    --config runs/<task>/label_config.json --execute 2 4 6 8 10 20 --rate <部署倍速> --out runs/<task>/replay_<labels>.json
```

把标签当成完美模型的输出闭环执行：取当前观测帧的标签，按服务端规则以倍速 r 插值，执行前 e 帧，再在录制中重新观测离执行位姿最近、指令状态相同的帧（只向前找），如此拼接到最后一个录制帧，并执行到停在最后一个录制位姿。表中给出路径偏差、接缝、关键帧位置误差、漏掉的指令、指令切换处的停留帧数、速度比、推理次数和完成帧数（执行 / 录制），是该配置下任何模型都去不掉的误差下限。

- **执行长度 e**：选择路径偏差、关键帧位置误差都不随 e 明显增大、且不漏指令的最长执行长度，再结合任务对反应速度的要求和推理延迟确定。LIBERO 上执行到整个 chunk 误差也不增大，这时 e 由模型需要多久重新观测一次决定。`time`、`span = H`、倍速 1 时路径偏差为 0，执行长度只受模型行为约束。回放每次重新观测至少前进一个录制帧，e 很小、倍速 1 时这一行高估误差（ISSUE-004）。
- **执行倍速 r**：服务端把第 k 个点放在 `(offset + k·span/H) / r` 帧。r = 1 时不含关键帧的一块标签用 span 帧走完 span 个录制帧，即按录制的节奏执行；r > 1 时沿同一条插值路径走得更快，标签和训练都不变，部署时设置（`ArcInterpolator.from_report(path, rate)`，不写入 `arc_resample.json`）。每块只有 ⌊span / r⌋ 个目标，e 不超过它。r 和 e 用下面的倍数上限在部署延迟下选；指令切换的位置取自路径，不随 r 改变，切换处的停留帧数随 r 减少。

```bash
python -m vla_tools.rate_ceiling --dataset data/<src>-timed --spec robot_spec.json --keyframes runs/<task>/keyframes.json \
    --horizon 10 --per-frame --metrics time curvature --spans <候选 span> --event-density 0 2 \
    --latency <部署延迟，秒> --modes async sync --path-bound 2 --feasible-fraction 0.9 --out runs/<task>/rate_ceiling.json
```

对每个配置、延迟和执行方式（异步：每 e 帧观测一次，chunk 晚 d 帧到达，从当时的指令位姿重建；同步：每次观测停 d 帧），在倍速网格和执行长度上闭环回放。可行：执行目标不超过关节速度上限、每个指令边沿都执行、闭环路径偏差不超过 2 个路径步长；加速比按完成帧数算。结果是至少 90% 的集可行时加速比最大的 (r, e)，另给每集最大可行加速比的中位数。加速度只报告不约束（实机控制器可调）。

- 选 r、e：取部署延迟下候选配置中 90% 结果最高者；它落在阈值上时（恰好 90% 的集可行），参考每集中位数和相邻倍速的可行集数，不要只看一个数。
- 录制本身超速的 episode 被排除并列出；排除很多时先回到清洗。
- 倍数上限把执行的指令当成机器人的运动：跟踪滞后和跟踪误差没有建模，实机上是上限而不是保证。
- **按速度定时**（`--velocity-fraction 1 --switch-hold <闭合 s> <张开 s>`）：服务端按关节速度上限给每块重新分配时间，倍速只作上限，慢段和停顿不再被最快的一段拖住；每个夹爪切换处固定停 h 秒，h 取夹爪实测的动作时间（DROID：0.533 s / 0.667 s，见 `gripper_actuation.py`）。按速度定时是默认部署方式：`ArcInterpolator.for_deployment(path, cap, fraction)`，停留取 `robot_spec.json` 的 `switch_hold_s`（随标签写进 `arc_resample.json`）；传给 `after_latency` 的观测位姿是观测时的指令位姿，起始位姿的指令值必须是已下发的指令。统一定时只作对照；两种定时各跑一份 `rate_ceiling`，按部署延迟选上限。

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

LIBERO `pick up the alphabet soup and place it in the basket`，50 集，20 Hz，H = 10，head 5，span 10–40，密度 0 / 1 / 2，补帧 40 行，曲率弦长 0.1224 rad（4 个路径步长），2026-10-14：

- `time` 与 `curvature` 的 30 个格子全部可接受；路径偏差中位 ≤ 0.21、p90 ≤ 0.45 路径步长（0.0306 rad），没有漏掉的指令；
- 候选清单：span 10、15 为 `time`、密度 0；span 20、30、40 为 `curvature`、密度 2。选定 `curvature` span 40 密度 2：整段路径偏差 0.11 / 0.23，关键帧位置 0.30，速度比 p95 1.03，覆盖帧数中位 33.6（密度 0 时每帧恰为 40）；
- 加入 `second_difference` 的 45 格网格中 span 40 选中 `second_difference`，它与 `curvature` 的整段路径偏差中位差 0.011 路径步长；
- 闭环回放，`curvature` span 40 密度 2，倍速 1：e = 2–40 时路径偏差中位 0.01–0.04、最大 0.42–0.57，关键帧位置中位 0.21–0.70 路径步长，没有漏掉的指令；完成帧数是录制的 1.21–1.25 倍（e = 10：1.23 倍），多出的时间来自关键帧窗口加密（密度 0 时 1.06 倍）；`time` span 10 基线路径偏差 0，完成帧数与录制相同；
- 倍速 2：e = 2–20 时完成帧数是录制的 0.61–0.62 倍（e = 10：比遥操快 1.63 倍），路径偏差最大 0.42–0.56 路径步长，速度比中位 1.89–2.14、最大 2.82–3.40；每块 20 个目标，e = 30、40 与 e = 20 相同；
- 同一选定配置按时间对齐的整段误差中位 5.42 帧运动量（点放在真实时间时 0.16）：点的时间由服务端决定，路径仍然跟上；
- 倍数上限（Panda 关节速度上限，路径上限 2 个路径步长，90% 的集可行）：`curvature` span 64、128 达到倍速 3，异步加速比 2.0–2.9（L = 0–0.4 s），`time` 标签停在倍速 2（密度 0：1.8–2.1）；逐帧标签 L = 0 时 1.99，L = 0.2 s 降到 0.97，0.4 s 没有可行的异步设置；按速度定时 5.83–3.95，加夹爪停留（DROID 的时间）2.94–2.28，接近含停留的上界 2.96。

## DROID 上的倍数上限（参考量级）

DROID 60 条遥操抓放（Franka Panda，15 Hz，只用关节指令和夹爪指令，5 条录制超速被排除），2026-10-17：

- 网格搜索：span 32、64 上 `curvature` 的整段路径偏差小于 `time`（中位 0.31 对 0.34–0.37、0.78 对 1.01–1.07 路径步长）；span 128、256 不可接受；
- 倍数上限（90% 的集可行）：最佳是 `time` span 32 密度 0，异步加速比 1.96、1.96、1.93、1.45、0.97（L = 0、0.05、0.1、0.2、0.4 s），同步 1.96–1.40；逐帧标签 1.59、1.50、1.48、0.99、无。`curvature` span 32 只到倍速 1.5，span 64 以上在超过 10% 的集上超出路径上限；
- 每集最大可行加速比的中位数随延迟平滑变化（`time` span 32：2.07–1.86），90% 结果在阈值附近跳动；
- 录制按一个统一倍速重放时，所有集都不超速只到 1.07 倍（每集中位 1.88 倍）；每步都跑到限速关节上限是 7.65 倍。统一倍速下最快的那段录制决定上限，换标签度量改变不了这一点；
- 按速度定时（90% 的集可行）：异步 7.46、7.20、5.77、5.55、1.39（L = 0、0.05、0.1、0.2、0.4 s），加上夹爪停留 0.533 / 0.667 s 后 4.58、4.53、3.91、3.27、1.54；同步在 0.4 s 时 3.41 和 2.36。L = 0 时达到理论上界的 98%（7.65；含同样停留时 4.68）。最佳配置是 `curvature` span 32–64 和 `time` span 32，上限 6–16；每个切换处至少停 9 帧，录制里中位 2 帧。逐帧标签的块只有 10 帧，有延迟时上限只能到 10 / (d + 1)，0.4 s 时只剩 0.82。

## 停止条件

- 没有可接受的格子：降低 span、换度量或密度，或放宽某个上限并写明理由，不要直接改阈值；
- 运控跳变上限未知，却需要在超过 H 的 span 之间做选择：只交付候选清单。

## 完成标准

- `sweep.json`（一张表加候选清单）、每个候选在部署倍速下的闭环回放表、部署延迟下的 `rate_ceiling.json` 与选定的执行长度和倍速、每个候选与基线的训练版和评估版数据集，都附带 `arc_resample.json`；
- 留出集与 `diversity.json` 一致。
