---
name: vla-keyframes
description: 为一个任务定义、检测、核对和修正关键帧（抓取、抬起、密封、接触、放置、释放等决定任务成败的时刻）。关键帧随任务和末端执行器而不同，必须显式定义、逐类看图核对、记录每一处修正。属于 vla-pipeline 的 S3 与 S5；结果供静止帧判断、动作标签、数据评估和模型评估使用。
---

# 关键帧

## 关键帧是什么，用在哪里

关键帧是任务中决定成败的时刻，每个关键帧带一个窗口，即属于这个时刻的前后若干帧。下游四处使用它：

| 使用者 | 用法 |
|---|---|
| `vla-episode-timing` | 落在关键帧窗口内的静止段属于任务本身（如抓取前稳住、等待密封），一律保留 |
| `vla-action-labels` | 窗口内的点可以加大 loss 权重（`key_weight`），也可以调整点密度（`event_density`） |
| `vla-data-diversity` | 每集只出现一次的关键帧处的关节构型，代表物体位置或放置位置 |
| `vla-eval` | 指令类关键帧在回放轨迹中的时间偏移，例如夹爪提前闭合 |

关键帧定义错了，这四处都会受影响，所以定义、检测、核对和修正都必须显式完成。

## 输入

- 清洗版（S3）或定时版（S5）数据集、`robot_spec.json`；
- 任务描述、末端执行器类型、可用的反馈信号；
- 工具：`vla_tools.keyframe_detect`、`vla_tools.keyframe_review`。

## 步骤

### 1. 定义

把任务按阶段写出来，例如"接近 → 稳住 → 抓取 → 抬起 → 搬运 → 放下 → 释放 → 撤离"。对每个阶段转换，判断它是否决定成败，再为每个关键帧确定：

- **信号**：动作维、状态维、其他数据集列，或关节速度；
- **检测器**：`command_edge`、`threshold_cross`、`motion_onset`、`speed_minimum`、`settle`、`manual`；
- **参照**：是否相对另一个关键帧搜索，以及搜索范围；
- **窗口**：前后各多少帧，或者一直到另一个关键帧为止（实测窗口）；
- **每集应出现几次**，以及各关键帧的先后顺序。

不同末端执行器的常见定义见 [按末端执行器定义关键帧](references/effectors.md)。没有现成信号的时刻（例如需要正运动学算出的末端高度），先由单独的步骤写成数据集的一列，再用 `column` 信号读取。

写成 `keyframes_spec.json`，格式见 `src/vla_tools/keyframe_spec.py` 的文件头。

### 2. 检测与一致性检查

```bash
python -m vla_tools.keyframe_detect --dataset <dataset> --spec robot_spec.json \
    --keyframes-spec runs/<task>/keyframes_spec.json [--overrides runs/<task>/keyframe_overrides.json] \
    --out runs/<task>/keyframes.json
```

输出每类关键帧的次数、在 episode 中的相对位置（中位数与 p10–p90），以及相邻关键帧之间的间隔。问题分以下几类：

| 检查 | 含义 | 常见原因 |
|---|---|---|
| count | 次数不符合预期 | 重抓、失败后重试、检测阈值不合适 |
| order | 顺序不对 | 参照关系写错，或阈值在错误的阶段被触发 |
| phase | 位置离群（稳健 z 分数超过阈值） | 该集节奏异常，或检测错位 |
| interval | 间隔离群 | 检测偏晚或偏早；操作者犹豫 |
| clipped | 窗口超出 episode 边界 | 录制在关键帧附近就开始或结束 |

### 3. 看图核对（每一类都要看，不能只看有问题的）

```bash
python -m vla_tools.keyframe_review --dataset <dataset> --spec robot_spec.json --keyframes runs/<task>/keyframes.json \
    --type <关键帧类型> --camera <最能看清该事件的相机> --out runs/<task>/kf_review/
```

每行是一集：窗口起点、关键帧、窗口终点的画面，以及一条信号曲线（指令维、关节速度、窗口阴影）。按定义逐行判断：

- 关键帧那一帧，画面里发生的是不是定义所说的事件（例如两指刚好围住物体）；
- 窗口起点是否落在事件的准备阶段，窗口终点时事件是否已经完成（例如物体被夹紧、被吸起）；
- 有问题的行带 `[issue]` 标记，但没有标记的行同样要看——检测可能在所有 episode 上一致地错位。

页数较多时，可以按类型分给 subagent 查看。各方结论汇总后，由一个执行者写入修正文件。

### 4. 修正

| 情况 | 做法 |
|---|---|
| 两集以上需要同样的修正 | 改检测器参数（阈值、持续帧数、搜索范围），不要逐集修 |
| 个别 episode 的检测错位 | 在 `keyframe_overrides.json` 的 `set` / `add` / `remove` 中写明帧号与理由 |
| 事件本身失败（抓空、重抓） | 排除该 episode（`exclude_episodes`），或增加一个次数不固定的关键帧类型 |
| 数据本身的性质（如录制在释放后立即停止） | 在 `accept` 中逐条或用 `"*"` 接受，并写明理由；如需改变数据，交给 `vla-episode-timing` |

修正后重新检测，直到开放问题为 0。

## S3 与 S5

- **S3（初检）** 在清洗版上运行，只用于 `vla-episode-timing` 判断静止段是否落在关键帧窗口内。这一步跑通即可，不做人工修正。
- **S5（定稿）** 在定时版上运行，完整执行上面四步。后续阶段只使用 S5 的结果。

两次检测的数据集版本不同、帧号不同，工具会拒绝跨版本使用关键帧文件。

## 停止条件

- 某类关键帧在核对图上与定义不符，且调整检测器和修正文件都无法解决：先回到第 1 步修改定义；
- 任务需要的信号不在数据里，也无法从现有数据推导。

## 完成标准

- `keyframes_spec.json`、`keyframe_overrides.json`（如有）、`keyframes.json` 和每一类的核对图都存在；
- `keyframes.json` 中开放问题为 0；
- 每个检测参数的取值，以及每条修正和接受，都写明了理由。
