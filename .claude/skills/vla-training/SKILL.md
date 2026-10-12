---
name: vla-training
description: 把构建好的标签数据集接入 openpi 训练：训练配置的改动点、归一化统计及静止维处理、loss 权重、基线与候选的对照设置、启动与监控。属于 vla-pipeline 的 S8。
---

# 训练接入（openpi）

## 要训练哪些模型

`vla-action-labels` 候选清单里的每个配置，加上逐帧基线（`time`、`span = H`），都用同一个留出集构建训练版数据集。各模型之间只有动作标签不同，其余全部一致。

## 训练配置的改动点

以已有的逐帧配置为模板，用 `dataclasses.replace` 只替换 `name` 和 `data`。模型、步数、batch、学习率、EMA、LoRA 冻结都保持不变。

| 位置 | 改动 | 原因 |
|---|---|---|
| `RepackTransform` | `"actions" ← "action_chunk"`；增加 `"action_loss_weight" ← "action_loss_weight"`；观测键与逐帧配置相同 | 标签来自构建好的列，而不是加载器切片 |
| 机器人 `Inputs` 变换 | 把 `(H·D,)` 重排为 `(H, D)`，长度不等于 H·D 时报错；透传 `action_loss_weight` | LeRobot 特征是扁平向量；从数据推断 H 会悄悄接受用别的 H 构建的数据集 |
| `DeltaActions` / `AbsoluteActions` 掩码 | 与逐帧配置相同（关节相对状态取差分，指令维保持绝对值） | 差分是相对当前状态取的，对重采样的点同样成立 |
| `DataConfig.action_sequence_keys` | 写数据集里真实存在的逐帧动作列，例如 `("action",)` | 加载器需要一个有效的键来生成 `delta_timestamps`；这一切片不作为标签使用 |
| 模型配置 | `action_horizon = H` | 不一致时第一个 batch 就会报形状错误 |
| loss 权重 | `Observation.action_loss_weight`（`src/openpi/models/model.py`），以及 `Pi0.compute_loss`（`src/openpi/models/pi0.py`）中逐点乘以权重 | 上游 openpi 没有这一项；PyTorch 训练路径不接收权重，等同于不加权 |
| 数据集位置 | `DataConfig.local_root` 指向数据集的父目录 | 加载器按 `local_root / repo_id` 查找本地数据集 |

上表的改动已整理在 `integrations/openpi/`：`patches/` 下是对上游 openpi 的补丁（loss 权重、本地数据集路径），`vla_openpi` 提供通用的 `ArcInputs`、`RawOutputs`、`ArcOutputs` 和 `LeRobotArcDataConfig`，`vla_openpi.launch` 可直接对配置对象计算 norm stats、训练和起服务。完整例子见 `examples/ur3_dual_arm/`。

## 归一化统计

1. 每个训练数据集各算一份：`uv run scripts/compute_norm_stats.py --config-name <config>`。chunk 覆盖得越远，关节差分的分布越宽，不能沿用基线的统计。
2. 体检标记为静止的维，其差分的 q01/q99 只有编码器噪声的量级（UR3 右臂约 ±9e-5 rad），分位数归一化会把噪声拉伸到 [−1, 1]。用 `norm_floor` 给这些维设一个最小范围：

   ```bash
   python -m vla_tools.norm_floor --stats assets/<config>/<asset_id>/norm_stats.json --key actions \
       --dims <静止维> --min-range 0.05 --out assets/<config>/<asset_id>/norm_stats.json
   ```

   覆盖前先备份原文件；`--min-range` 的单位与差分相同（关节是 rad），应远大于噪声、又远小于这些关节在真正运动时的幅度。
3. 基线和候选对静止维的处理要一致。

## 训练与监控

- 启动方式和资源设置见 openpi 的文档；不使用 W&B 时设置 `WANDB_MODE=offline`，或在配置里关闭 `wandb_enabled`；
- 日志里的 loss 是加权平均，且依赖标签定义，不同数据集之间不能比较；
- 加权后平均权重大于 1 时，有效学习率随之变大。`build_labels` 会打印平均权重；对比报告里要写明，或按平均权重调整学习率；
- 训练结束后，把标签数据集的 `meta/arc_resample.json` 拷到 checkpoint 的 `assets/<asset_id>/` 下，与 `norm_stats.json` 放在一起。openpi 保存 checkpoint 时只写 norm stats，而服务端要用这个文件确定 span、H 和 anchor。

## 停止条件

- 训练配置与数据集的 H 不一致；
- 候选与基线之间除标签外还有其他差别（步数、学习率、留出集、静止维处理）。

## 完成标准

- 每个模型都有 checkpoint，其 `assets/` 中同时有 norm stats 和 `arc_resample.json`；
- 记录了每个模型的配置名、数据集、平均权重、静止维处理和训练时长。
