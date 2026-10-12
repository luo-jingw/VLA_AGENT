---
name: vla-serving
description: 把训练好的 VLA 模型交付到服务端：重采样标签的服务端插值适配（逐帧目标接口、放点时间、夹爪规则）、与训练一致的推理配置、标签过服务端检查、LoRA 合并后转 PyTorch 及 JAX/PyTorch 一致性检查、交付目录与说明文档、上传模型仓库。属于 vla-pipeline 的 S10。
---

# 服务端交付

## 输入

checkpoint（其 `assets/<asset_id>/` 下有 norm stats 和 `arc_resample.json`）、服务端机器的情况（已有的 openpi 服务和客户端、JAX 或 PyTorch）、选定的执行长度。

## 1. 服务端插值

模型输出的 H 个点不一定是逐帧的。服务端要把它们插值成逐帧目标再发给机器人。具体接口、插值步骤和推理配置见 [服务端适配](references/server-adaptation.md)，要点：

- 插值实现只有一份：`vla_tools/arc_interp.py`。它只依赖 numpy 和 scipy，单独拷到服务端使用；
- 参数用 `ArcInterpolator.from_report(<ckpt>/assets/<asset_id>/arc_resample.json)` 读取，不在代码里手写；
- 返回的 `actions[j]` 是观测后第 j+1 帧的目标，含义与逐帧策略相同，已有客户端不用改；
- 指令维只由模型的点决定，当前状态里的指令值不参与插值；
- 机器人执行前 `execute` 个目标后重新请求。

服务端已经有成套的服务和客户端时，只需加上插值这一步，以及与训练一致的推理配置。

## 2. 交付前检查

| 检查 | 方法 | 通过标准 |
|---|---|---|
| 标签过服务端 | 把评估版数据集中的标签点和录制状态直接送入输出变换 | 点所在帧的输出等于标签；其余帧的误差等于 L0 下限 |
| 真实服务回放 | 对实际起好的服务跑一次 `replay_eval`（一条留出 episode） | 返回形状、`arc_point_frames` 正确，延迟可接受 |
| PyTorch 一致性 | 见下节 | 关节输出差异在 bf16 舍入量级，指令维一致 |

## 3. PyTorch 版本

LoRA 训练的 checkpoint 必须先把适配器合并进主权重，再转换。openpi 自带的转换脚本只映射主权重，并用 `strict=False` 加载，会不报错地丢掉全部适配器。合并不能简单地对每个 head 做 `lora_a @ lora_b`：openpi 的注意力输出投影把 `lora_b` 在 head 维上求和，`lora_to_pytorch` 按 openpi 的 einsum 方程推导每层的增量。

```bash
PYTHONPATH=<openpi>/examples:.claude/skills/vla-pipeline/tools uv run python -m vla_tools.lora_to_pytorch \
    --checkpoint-dir <run>/<step> --config-name <config> --output-path <run>/<step>_pytorch
```

转换后做一致性检查。两个 dump 分别在独立进程中运行，因为 JAX 预分配的显存不会释放：

```bash
python -m vla_tools.policy_parity dump --config-name <config> --checkpoint <run>/<step> ... --out jax.npz
python -m vla_tools.policy_parity dump --config-name <config> --checkpoint <run>/<step>_pytorch ... --out torch.npz
python -m vla_tools.policy_parity compare --a jax.npz --b torch.npz --spec robot_spec.json
```

PyTorch 推理需要 openpi 打过补丁的 `transformers`。默认配置下第一次推理会先执行 `torch.compile`（max-autotune），耗时可达数分钟。

## 4. 交付与上传

交付目录与说明文档的写法见 [交付](references/release.md)。上传到模型仓库前：

- 确认仓库是公开还是私有，并在交付说明中写明；
- 凭据按 `AGENTS.md` 的"Credentials and Environment Isolation"使用项目范围内的 token，来源记在 `PROJECT.md` 中，不写入任何文件内容；
- 上传后逐个比对文件大小，大文件再比对校验和。

## 停止条件

- 服务端配置与训练配置有任何一项不一致；
- PyTorch 版本没有通过一致性检查；
- 凭据的归属不明确。

## 完成标准

- 交付目录包含权重、norm stats、`arc_resample.json`、`arc_interp.py`、推理配置和说明文档；
- 交付前检查全部通过，或未通过的项目已写入说明文档；
- 上传后的文件与本地一致。
