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
- 每个训练数据集各算一份 norm stats；稀疏标签按点分开统计（`run.py norm-stats --per-point-actions`，得到 (H, D) 的统计）。数据体检标记为静止的维，用 `norm_floor` 设最小归一化范围；它对按点统计的每个点都生效。
- 每集有多条指令时（`vla-data-intake`），配置里设 `LeRobotArcDataConfig(rotate_prompts=True)`、基础配置 `prompt_from_task=False`：每个训练样本从所在集的 `tasks` 里随机抽一条。抽取发生在 repack 阶段，openpi 只在训练时运行这一阶段，服务时用客户端发来的指令。
- `examples/ur3_dual_arm/configs.py` 冻结视觉语言模型（SigLIP 与 Gemma 2B，`vla_openpi.vlm_filter`），只训练动作专家和投影层，不输入状态（π0.5 在 `discrete_state_input=False` 时不加状态 token，也不把状态写进指令），与 LIBERO 的三种标签对比用同一配方。
- 候选与逐帧基线（`time`、`span = H` 构建的标签）只在标签上不同；加权后平均权重大于 1 时要写明，它会改变有效学习率。
- 训练后把 `arc_resample.json` 拷到 checkpoint 的 `assets/<asset_id>/`，服务端从这里读取标签配置。不用 W&B 时设置 `WANDB_MODE=offline`。

## 评估

- `label_replay`：标签本身经服务端插值、在部署倍速下闭环执行后的路径偏差、关键帧位置误差与速度比，是任何模型都低不过的下限。
- `replay_eval`：在留出集上开环回放，按 points（模型点与标签点之差）、head、stitched（拼接后的逐帧误差）、edges（指令时机）、static、hold（补帧行上的保持误差）、ms 报告；`coverage_eval` 把误差与"离训练数据多远"对照。
- 闭环试验用同一协议：相同的初始摆放、次数、执行长度和执行倍速，成功率附置信区间，失败按关键帧阶段归类。
- 训练 loss 不能跨标签定义比较；训练集上的回放只说明链路可用。
- 仿真闭环（LIBERO，`src/vla_libero/`，配置见 `examples/libero_object/`）：
  - 先用 `vla_libero.replay --source joint --kp ...` 回放录制的关节轨迹选控制增益，再用 `--source labels` 在同样的倍速、执行长度、延迟和停留下开环执行标签本身，得到物理里的上限；
  - 夹爪停留用实测的动作时间（从夹爪开合宽度测），不加停留时抓取大多失败（LIBERO 开环 11/50 成功）；
  - `run.py serve-points` 起服务返回 (H, D) 点，每个服务先答一次请求（首次请求要编译模型），再启动 `vla_libero.evaluate`；延迟按仿真帧模拟，与实际推理耗时无关；
  - 每组条件跑任务的 50 个基准初始状态，报告成功率与 Wilson 95% 区间、成功试次的完成帧数与加速比、按物体高度分的失败阶段（没抬起、掉落、没放进），并与同设置的开环标签回放对照。

## 交付

- 服务端把模型输出的点插值成逐帧目标，`vla_tools/arc_interp.py` 可单独拷贝。**默认按速度定时部署**：`ArcInterpolator.for_deployment(<ckpt>/assets/<asset_id>/arc_resample.json)` 从报告的 robot 块读取关节速度上限和夹爪停留 `switch_hold_s`，同一条样条按关节速度上限的 `DEPLOY_FRACTION`（1.0）分配时间，最快 `DEPLOY_CAP`（4）倍录制平均节奏，每个夹爪切换处手臂停住实测的动作时间。机器人执行 e 帧后重新请求（LIBERO：e = 4）。
- 服务：`vla_openpi.launch.serve(config, checkpoint, port)`（默认 `cap=4, velocity_fraction=1.0`），示例 `run.py serve [--cap 4] [--velocity-fraction 1]`；`--uniform`（`velocity_fraction=None`）是按 `cap` 倍速均匀定时，只作对照。服务端每块单独定时，被下一次请求截断的停留不会补到下一块；保存指令历史的机器人端循环用 `for_deployment(report).chunk_targets(points, history, observed, latency_frames, mode, execute)`，未完成的停留会带到下一块（仿真闭环即如此）。
- 夹爪停留写在 `robot_spec.json` 的 `switch_hold_s`（合、张，秒），取夹爪实测的动作时间（开合宽度到 90% 行程），`build_labels` 把它带进 `arc_resample.json`；报告缺关节速度上限或该字段时 `for_deployment` 报错并指出缺的字段，补进规格后重建标签，或显式构造 `VelocityTiming`。
- 部署前在部署延迟下核对标签本身在该上限的表现：仿真用 `vla_libero.replay --source labels`，运动学用 `label_replay` 和 `rate_ceiling`。标签开环成功率随上限下降（LIBERO，`curvature` s32 d1：上限 2 / 4 / 8 为 98% / 88% / 74%），上限按任务对速度和成功率的取舍选。
- 推理配置与训练一致：模型配置、分位数归一化、差分掩码、观测键映射、checkpoint 中的 norm stats。
- LoRA 训练的模型转 PyTorch 用 `lora_to_pytorch`：先把适配器合并进主权重，注意力输出投影的 `lora_b` 要在 head 维上求和；转换后用 `policy_parity` 与 JAX 在同观测、同噪声下比较。
- 上传模型仓库前确认可见性，凭据按 `PROJECT.md`；上传后比对文件大小，大文件再比对校验和。
- 发布 LeRobot 数据集到 Hugging Face Hub：上传数据集目录并附数据集说明（来源与版本、排除的集及理由、处理步骤、各列含义、每集一行的表），然后给提交打 `v2.1` 标签。`LeRobotDataset` 从 Hub 加载时按代码版本标签取版本，没有这个标签会报错。替换已发布的版本时，在同一次提交里删除本地已经不存在的文件（`upload_folder(..., delete_patterns=[...])`），再把标签移到新提交；旧版本仍留在 git 历史里。上传或替换公开仓库由人决定。

## 停止条件

- 训练配置的 `action_horizon` 与数据集不一致；
- 服务端的 span、H、anchor 与 `arc_resample.json` 不一致；
- PyTorch 版本没有通过一致性检查。
