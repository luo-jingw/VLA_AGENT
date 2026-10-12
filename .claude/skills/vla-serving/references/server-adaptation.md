# 服务端适配

## 接口

| 字段 | 内容 |
|---|---|
| 输入 | 与逐帧策略相同：图像、状态、prompt |
| `actions` | `(n, D)` 逐帧目标，`actions[j]` 是观测后第 j+1 帧的目标；n = ⌊offset + span⌋，`anchor="state"` 时 offset 为 0，`"action"` 时为 1 |
| `arc_points` | `(H, D)` 模型输出的点（绝对值），用于区分模型误差和放置误差 |
| `arc_point_frames` | `(H,)` 每个点被放到第几帧 |

`actions` 的含义与逐帧策略一致，已有客户端不用改，只是每次收到的目标数从 H 变成 n。机器人执行前 `execute` 个目标后发送新的观测。

## 插值步骤

`vla_tools/arc_interp.py` 实现以下步骤，只依赖 numpy 和 scipy：

1. **取绝对点**：在 openpi 的输出变换链中，接在 `Unnormalize` 和 `AbsoluteActions` 之后，即机器人 `Outputs` 的位置。此时 `data["actions"]` 是绝对目标，`data["state"]` 是反归一化后的当前状态。截掉模型维度补齐的部分。
2. **定时间**：第 k 个点放在 `offset + k·span/H` 帧。`time` 度量下这就是真实时间，进度度量下是假定的时间。
3. **关节维**：用 PCHIP 经过 `(0, 当前位姿)` 和 `(时间_k, 点_k)`，在第 1…n 帧取值。当前位姿由 `state_dims` 从状态换算到动作布局。
4. **指令维**：取时间上最近的点的值，取整到 0/1。当前状态里的指令值不参与插值：它只是上一帧命令的回显，回显滞后时会让模型已经闭合的夹爪在重新规划后张开一帧。

所有参数都从 checkpoint 自带的 `arc_resample.json` 读取（其中 `robot` 部分给出维度和状态对应关系，`config` 部分给出 span、H 和 anchor）：

```python
interp = ArcInterpolator.from_report(Path(ckpt) / "assets" / asset_id / "arc_resample.json")
targets = interp(points, state)   # (n, D)
```

openpi 输出变换的写法：

```python
@dataclasses.dataclass(frozen=True)
class ArcOutputs(transforms.DataTransformFn):
    interp: ArcInterpolator

    def __call__(self, data: dict) -> dict:
        points = np.asarray(data["actions"][:, : self.interp.action_dim])
        return {
            "actions": self.interp(points, data["state"]).astype(np.float32),
            "arc_points": points.astype(np.float32),
            "arc_point_frames": self.interp.frames().astype(np.float32),
        }
```

## 与训练一致的推理配置

以下各项必须与训练配置逐项一致：
- 模型配置：`pi05`、`action_horizon = H`、`discrete_state_input`、VLM 和动作专家的变体；
- 分位数归一化、差分掩码、观测键映射；
- norm stats 从 checkpoint 的 `assets/` 读取（含 `norm_floor` 处理后的版本）。

推理用不到 repack 和 loss 权重。`vla_openpi` 的配置只依赖上游 openpi（加补丁）和 `arc_interp.py`，它不能注册进 openpi 的 `config.py`（两者会互相 import），所以由 `vla_openpi.launch.serve` 直接对配置对象起服务：它把输出变换换成从 checkpoint 报告构造的 `ArcOutputs`，再用 `policy_config.create_trained_policy` 和 `WebsocketPolicyServer` 启动。
