---
name: vla-pipeline
description: VLA（openpi π0/π0.5）从录制数据到可部署模型的总流程：按阶段调度数据体检与清洗、关键帧、静止帧与末尾补帧、数据评估、动作标签、训练、模型评估、服务端交付等子 skill，管理阶段产物与放行条件。新机器人、新任务或新数据集进入微调时从这里开始。
---

# VLA 微调总流程

## 触发与输入

适用于：一个新的机器人/任务/数据集要微调 openpi 策略并上机部署，或者已有流程中的某一阶段要重做。只改训练超参、不动数据和标签时，直接用 `vla-training`。

开始前确定以下输入；缺任何一项，就先停在 S0：

| 输入 | 用途 |
|---|---|
| 数据集（LeRobot v2.1）路径，只读 | 所有阶段的源头；派生版本另存，不覆盖源数据 |
| `robot_spec.json`：动作维、关节维、指令维、状态维对应、fps、相机列 | 所有工具共用，格式见 [产物](references/artifacts.md#robot_specjson) |
| 任务描述与末端执行器类型 | 关键帧怎么定义 |
| 运控约束：控制频率、首点跳变上限、能否执行逐帧目标 | 标签选择的上限、服务端接口 |
| 部署目标：服务端机器、JAX 或 PyTorch | 交付物形式 |
| 模型配置：`action_horizon`、是否 LoRA | 标签点数、转换方式 |

## 阶段

| 阶段 | 子 skill | 输入 → 输出 | 放行条件 |
|---|---|---|---|
| S0 范围 | 本 skill | 上表 → `runs/<task>/robot_spec.json`，`plan.md` 中的本次任务 | 关节维与指令维恰好覆盖全部动作维；状态维对应已核对 |
| S1 体检 | `vla-data-cleaning` | 源数据 → `audit.json`、跳变核对图 | 已确认 `action[t] ≈ state[t+1]`；每类发现都有处理决定 |
| S2 清洗 | `vla-data-cleaning` | 源数据 + `cleaning_plan.json` → 清洗版 | 所有非孤立跳变都有书面决定；清洗报告存在 |
| S3 关键帧初检 | `vla-keyframes` | 清洗版 + 关键帧定义 → `keyframes_pass1.json` | 检测能跑通；此时只用于给静止段分类，不做人工修正 |
| S4 静止帧与末尾 | `vla-episode-timing` | 清洗版 + 初检关键帧 + `timing_plan.json` → 定时版 | 每条非空决定都写了理由；补帧行带 `is_padding` |
| S5 关键帧定稿 | `vla-keyframes` | 定时版 → `keyframes.json`、核对图、修正文件 | 开放问题为 0（或逐条接受并写明理由）；每类关键帧都看过核对图 |
| S6 数据评估 | `vla-data-diversity` | 定时版 + 关键帧 → `diversity.json`、留出集 | 覆盖、聚类、近重复、离群都已解释；留出集覆盖各簇 |
| S7 动作标签 | `vla-action-labels` | 定时版 + 关键帧 + 留出集 → 网格表、执行长度、训练版与评估版数据集 | 候选清单含逐帧基线；选定执行长度；构建报告存在 |
| S8 训练 | `vla-training` | 训练版数据集 → checkpoint | 候选与基线只在标签上不同；norm stats 处理过静止维 |
| S9 模型评估 | `vla-eval` | checkpoint + 评估版数据集 → 对比表 | 在留出集上评估；结论附置信度；未观测项列明 |
| S10 交付 | `vla-serving` | checkpoint → 可部署目录、插值文件、交付说明 | 标签过服务端检查通过；PyTorch 版本通过一致性检查 |
| S11 收尾 | `close` | 全部产物 → `docs/`、`issues.md`、`opportunities.md` | 见 `close` |

S3 和 S5 是同一个工具跑两次：S4 改变了行，帧号随之改变，所以人工核对和修正只在 S5 做。关键帧工具拒绝使用由其他数据集版本得到的 `keyframes.json`，这一步不会漏掉。

S7 不做重采样时（直接用逐帧动作），仍按 `metric="time"`、`span_frames = H` 构建标签：它与逐帧 chunk 完全相同，后续阶段的工具和比较方式不变。

## 工具

所有工具在 `tools/vla_tools/` 下，以模块方式运行：

```bash
PYTHONPATH=.claude/skills/vla-pipeline/tools <openpi>/.venv/bin/python -m vla_tools.<module> --help
```

只读数据的工具依赖 numpy、scipy、pandas、pyarrow、matplotlib、av；写数据集的依赖 `datasets`；`replay_eval` 依赖 `openpi_client`；`lora_to_pytorch`、`policy_parity` 需要 openpi 环境。openpi 的虚拟环境满足以上全部。`arc_interp.py` 不依赖包内其他模块，交付时单独拷贝。

## 状态与产物

每个阶段的产物只有一个写入者，后续阶段只读。路径约定、文件格式和读写关系见 [产物](references/artifacts.md)。派生数据集按 `<源名>-<阶段>` 命名并与源数据并列存放，每一版都在 `meta/` 下带有生成它的报告（`cleaning_report.json`、`timing_report.json`、`arc_resample.json`），可以据此追溯到源数据。

各阶段可能出现的已知问题、检测方法与处理见 [已知失效模式](references/failure-modes.md)。

## 并行

S6 与 S7 的网格搜索互不依赖，可以同时进行；S8 中候选模型与基线可以同时训练。关键帧核对图可按类型分给多个 subagent 查看，但修正文件只有一个写入者：由本流程汇总各方结论后写入。

## 停止条件

在 `AGENTS.md` 的停止条件之外：

- 数据的索引约定（`action[t]` 对应哪一帧的状态）无法确认；
- 某个关键帧类型在核对图上与它的定义不符，且无法通过调整检测器或修正文件解决；
- 留出集无法覆盖数据的主要簇（数据太少或高度集中）——此时如实报告，不给出泛化结论；
- 运控约束未知，却需要在超过逐帧基线的 span 之间做选择。

## 完成标准

- 每个阶段的放行条件都满足，产物都在约定位置；
- 交付目录、插值文件与训练所用标签配置一致（span、H、anchor 来自同一份 `arc_resample.json`）；
- 已验证的事实写入 `docs/`，未解决的问题写入 `issues.md`，没跑的候选写入 `opportunities.md`。
