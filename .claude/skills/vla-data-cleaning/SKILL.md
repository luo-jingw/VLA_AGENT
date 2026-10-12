---
name: vla-data-cleaning
description: 机器人示教数据的体检与清洗：确认索引约定，找出静止维、时间戳缺口、非有限值、二值指令毛刺、相机冻结、重复 episode 和关节跳变，并区分录制故障与真实快速动作，再按书面决定修复或排除，生成新的数据集版本。属于 vla-pipeline 的 S1–S2。
---

# 数据体检与清洗

## 触发与输入

用于任何派生处理之前，以及源数据更新之后。输入：源数据集（只读）、`robot_spec.json`。

工具：`vla_tools.audit`（只读）、`vla_tools.clean`（写新版本）。运行方式见 `vla-pipeline`。

## S1 体检

```bash
python -m vla_tools.audit --dataset <src> --spec robot_spec.json --out runs/<task>/audit.json \
    --review-dir runs/<task>/audit_review [--review-camera <腕部相机列>]
```

逐项读结果并做决定：

| 检查 | 判断 | 处理 |
|---|---|---|
| `action[t]` 对 `state[t+1]` 与对 `state[t]` 的最大差 | 前者应远小于后者 | 不成立时停止：标签构建与服务端放点都依赖这个约定 |
| 时间戳缺口 | 帧间隔超过 1.5 帧 | 丢帧处动作不连续；缺口多的 episode 排除，少量缺口记录在案 |
| 非有限值 | 任何 NaN/inf | 排除该 episode |
| 静止维 | 某关节在所有 episode 内的变化都小于阈值 | 数据本身不改；在训练阶段用 `norm_floor` 处理（见 `vla-downstream`） |
| 指令取值与边沿数 | 只应有 0/1；每集边沿数应符合任务 | 非二值时停止，该维应归为关节维；边沿数异常的 episode 在关键帧阶段核对 |
| 指令毛刺 | 持续不到 `--pulse-frames` 帧的指令段 | 一般是按键抖动，清洗时去除；若是真实的快速开合，把阈值调低并写明 |
| 相机冻结 | 连续多帧图像字节完全相同 | 相机卡顿；冻结段长的 episode 排除 |
| 重复 episode | 动作数组完全相同 | 只保留一份 |
| 跳变 | 单帧关节变化超过中位步长的 `--jump-factor` 倍 | 见下节 |
| 静止比例、首尾静止 | 低速帧占比与所承载的距离 | 留给 `vla-episode-timing` 决定，这里只记录 |

### 跳变的判断

工具把每处跳变分为三类，分类结果只作为初判，必须对照核对图确认：

- **孤立**：下一帧数值又跳回原处。属于录制故障，用相邻两帧的中点修复。
- **持续**：下一帧沿同一方向继续运动。属于真实的快速动作（例如抓取后的抬起起步），保留，并把它的大小交给运控核对。
- **断点**：跳过去之后立即静止，像瞬移。可能是关节编码器回绕、跟踪丢失后恢复，也可能是真实动作，必须逐个决定。

核对图上，左边是跳变前后 ±10 帧的关节曲线，右边是跳变前后的相机画面。画面与曲线一致地连续运动，才算持续运动。

## S2 清洗

写 `cleaning_plan.json`（格式见 `python -m vla_tools.clean --help`），然后：

```bash
python -m vla_tools.clean --dataset <src> --spec robot_spec.json --plan runs/<task>/cleaning_plan.json --out data/<src>-clean
```

- 每处持续或断点跳变都要在 `jump_decisions` 里写 `keep: <理由>` 或 `repair: <理由>`，缺少时工具拒绝写出并列出缺项；
- 排除的 episode 写在 `exclude_episodes` 中并附理由；
- 清洗只修改数值、排除 episode，不增删行；静止帧与末尾补帧由 `vla-episode-timing` 处理；
- 修复会同步改写对应的状态：`action[t]` 被修复时，`state[t+1]` 也随之更新。

## 停止条件

- 索引约定无法确认；
- 指令维不是二值，且无法归为关节维；
- 某类发现缺少处理决定。

## 完成标准

- `audit.json`、核对图、`cleaning_plan.json`、清洗版数据集及其 `meta/cleaning_report.json` 都存在；
- 对清洗版重新体检，只剩下已决定保留的发现。
