---
name: vla-downstream
description: 训练就绪数据集的下游使用：接入 openpi 训练、在留出集上开环评估、交付到服务端。只读取数据预处理的产物（标签数据集与 arc_resample.json），接入细节见 integrations/openpi。属于 vla-pipeline 的 S8。
---

# 下游：训练、评估、交付

## 交接

| 输入 | 来自 |
|---|---|
| 训练版、评估版标签数据集 | `vla-action-labels` |
| `meta/arc_resample.json`：机器人布局与标签配置 | 同上，在每个标签数据集里 |
| 留出集 | `diversity.json` |

下游不修改数据。发现数据问题（关键帧不对、需要排除 episode、覆盖不足）时，写进 `runs/<task>/decisions.md` 并回到对应的数据阶段。

## 训练

- openpi 打上 `integrations/openpi/patches/` 的补丁（逐点 loss 权重、本地数据集路径），用 `vla_openpi.LeRobotArcDataConfig` 组装配置；完整例子见 `examples/ur3_dual_arm/`，步骤见 `integrations/openpi/README.md`。
- 每个训练数据集各算一份 norm stats；数据体检标记为静止的维，用 `norm_floor` 设最小归一化范围。
- 候选与逐帧基线（`time`、`span = H` 构建的标签）只在标签上不同；加权后平均权重大于 1 时要写明，它会改变有效学习率。
- 训练后把 `arc_resample.json` 拷到 checkpoint 的 `assets/<asset_id>/`，服务端从这里读取标签配置。不用 W&B 时设置 `WANDB_MODE=offline`。

## 评估

- `label_replay`：标签本身经服务端插值后的误差下限，任何模型都低不过它。
- `replay_eval`：在留出集上开环回放，按 points（模型点与标签点之差）、head、stitched（拼接后的逐帧误差）、edges（指令时机）、static、hold（补帧行上的保持误差）、ms 报告；`coverage_eval` 把误差与"离训练数据多远"对照。
- 闭环试验用同一协议：相同的初始摆放、次数、执行长度，成功率附置信区间，失败按关键帧阶段归类。
- 训练 loss 不能跨标签定义比较；训练集上的回放只说明链路可用。

## 交付

- 服务端把模型输出的点插值成逐帧目标：`vla_tools/arc_interp.py` 可单独拷贝，参数用 `ArcInterpolator.from_report(<ckpt>/assets/<asset_id>/arc_resample.json)` 读取；机器人执行前几帧后重新请求。
- 推理配置与训练一致：模型配置、分位数归一化、差分掩码、观测键映射、checkpoint 中的 norm stats。
- LoRA 训练的模型转 PyTorch 用 `lora_to_pytorch`：先把适配器合并进主权重，注意力输出投影的 `lora_b` 要在 head 维上求和；转换后用 `policy_parity` 与 JAX 在同观测、同噪声下比较。
- 上传模型仓库前确认可见性，凭据按 `PROJECT.md`；上传后比对文件大小，大文件再比对校验和。

## 停止条件

- 训练配置的 `action_horizon` 与数据集不一致；
- 服务端的 span、H、anchor 与 `arc_resample.json` 不一致；
- PyTorch 版本没有通过一致性检查。
