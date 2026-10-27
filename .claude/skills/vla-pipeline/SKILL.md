---
name: vla-pipeline
description: VLA 示教数据预处理的总流程：把录制数据处理成训练就绪的数据集——原始录制接入与指令标注、体检与清洗、关键帧、静止帧与末尾补帧、数据评估与留出集、动作标签——并生成数据报告；训练、评估、部署作为下游交给 vla-downstream。新机器人、新任务或新数据集从这里开始。
---

# VLA 数据预处理

## 定位

这个流程的交付物是数据：一份训练就绪的数据集（训练版与评估版，附标签配置 `arc_resample.json`）和一份数据报告。下游的训练、评估、部署只读取这两样东西，细节见 `vla-downstream`。

数据阶段占流程的大部分，原因是 VLA 策略的上限主要由示教数据决定：索引约定错一帧、把真实动作当成噪声删掉、关键帧定义与末端执行器不符、录制在任务完成时就截断、留出集和训练集来自同一批次，这些问题在训练日志里都看不出来，只能在数据阶段发现。

## 触发与输入

每次开始或恢复一个任务时，先读 `AGENTS.md` 与 `PROJECT.md`，再运行 `python -m vla_tools.run_status runs/<task>`，从第一个没有通过的阶段继续。需要的输入：

| 输入 | 用途 |
|---|---|
| 数据集（LeRobot v2.1）路径，只读；原始录制（其他格式）先经 `vla-data-intake` 转换和标注 | 源头；派生版本另存，不覆盖源数据 |
| `robot_spec.json`：动作维、关节维、指令维、状态维对应、fps、相机列 | 所有工具共用，格式见 [产物](references/artifacts.md#robot_specjson) |
| 任务描述与末端执行器类型 | 关键帧怎么定义 |
| 模型 `action_horizon`；运控对首点跳变的上限 | 标签选择；上限未知时只交付候选清单 |

缺数据集或 `robot_spec.json` 时停在 S0。

## 阶段

| 阶段 | skill | 输入 → 输出 | 放行条件 |
|---|---|---|---|
| S0 接入与范围 | `vla-data-intake`、本 skill | 原始录制 → 标注版源数据集、`conversion.json`、`prompts.json`；上表 → `robot_spec.json`、`run.json` | 每个排除都有理由；每集拼图看过；指令覆盖全部保留的集；关节维与指令维恰好覆盖全部动作维；状态维对应已核对 |
| S1 体检 | `vla-data-cleaning` | 源数据 → `audit.json`、跳变核对图 | 已确认 `action[t] ≈ state[t+1]`；每类发现都有处理决定 |
| S2 清洗 | `vla-data-cleaning` | 源数据 + `cleaning_plan.json` → 清洗版 | 所有非孤立跳变都有书面决定 |
| S3 关键帧初检 | `vla-keyframes` | 清洗版 → `keyframes_pass1.json` | 检测跑通；只用于给静止段分类 |
| S4 静止帧与末尾 | `vla-episode-timing` | 清洗版 + 初检关键帧 + `timing_plan.json` → 定时版 | 每条决定都有理由；补帧行带 `is_padding` |
| S5 关键帧定稿 | `vla-keyframes` | 定时版 → `keyframes.json`、核对图 | 开放问题为 0；每一类都有核对图并已看过 |
| S6 数据评估 | `vla-data-diversity` | 定时版 + 关键帧 → `diversity.json`、留出集 | 每个簇都有解释；留出集覆盖各簇 |
| S7 动作标签 | `vla-action-labels` | 定时版 + 关键帧 + 留出集 → 训练版、评估版 | 候选清单含逐帧基线；按标签回放选定执行长度；训练版去掉的正是留出集 |
| S8 下游 | `vla-downstream` | 训练版、评估版 → checkpoint、评估表、交付目录 | 见 `vla-downstream` |
| S9 收尾 | `close` | 全部产物 → `docs/`、`issues.md`、`opportunities.md` | 见 `close` |

S7 通过后运行 `data_report` 写出 `data_report.md`，再把训练版、评估版和数据报告交给下游。

S3 和 S5 是同一个工具跑两次：S4 改变了行和帧号，人工核对与修正只在 S5 做；关键帧工具拒绝使用另一个数据集版本的关键帧文件。不做标签重采样时，S7 仍按 `metric="time"`、`span_frames = H` 构建，它与逐帧 chunk 完全相同。

下游发现的数据问题（错误的关键帧、需要排除的 episode、覆盖不足）回到产生它的数据阶段处理，下游不修改数据。

## 决策权限

| 决定 | 谁定 | 记录在 |
|---|---|---|
| 检测器参数与阈值（跳变倍数、静止阈值、关键帧检测器） | agent，附核对图或统计作为证据 | 对应的规格或计划文件 |
| 修复孤立跳变、去除指令毛刺 | agent | `cleaning_plan.json` |
| 保留持续运动的跳变 | agent，看过核对图 | `jump_decisions` |
| 断点跳变的修复或保留 | 人 | `jump_decisions` |
| 排除 episode（含机位、批次异常） | 人 | `exclude_episodes` |
| 接入时排除 episode（帧率、丢帧、没有运动、画面里有人或任务失败） | 人，agent 附拼图或统计 | `conversion.json` |
| 每集的指令标注 | agent 写，人抽查 | `prompts.json` |
| 接受关键帧检查的问题 | 人，agent 提出理由 | 关键帧修正文件的 `accept` |
| 裁掉开头、中途或结尾的静止帧 | 人 | `timing_plan.json` |
| 末尾补帧的帧数（不少于一个 span） | agent | `timing_plan.json` |
| 留出集 | agent 用 `diversity` 提出，人确认 | `diversity.json` |
| 标签候选清单 | agent，按网格搜索的选择规则 | `sweep.json` |
| 送去训练的候选、启动长时间 GPU 任务 | 人 | 项目 `plan.md` |
| 删除数据或 checkpoint；上传或替换公开仓库中的文件 | 人 | — |

需要人决定的事项写进 `runs/<task>/decisions.md`，每条写明阶段、事项、证据和建议；agent 停在这一阶段，等决定后再写入对应文件继续。

## 运行记录

| 文件 | 写入者 | 内容 |
|---|---|---|
| `runs/<task>/run.json` | `run_index`，每生成一个数据集版本时运行一次 | 本次运行各数据集版本的路径：`source`、`clean`、`timed`、`labels`（每个候选的 `train` / `eval`） |
| `runs/<task>/decisions.md` | 本 skill | 待人决定的事项；决定后标记并注明写入了哪个文件 |
| `runs/<task>/data_report.md` | `data_report`，S0–S7 都通过后 | 数据集版本谱系、体检发现与清洗决定、关键帧、静止帧与补帧、覆盖与留出集、标签候选与配置、人的决定 |

`run_status` 只从这些文件和各阶段产物判断状态：`passed`（产物齐全、能由文件检查的放行条件都满足）、`open`（某项检查不满足，原因列出）、`missing`（缺产物）、`waiting`（`decisions.md` 中有该阶段的未决事项）。核对图是否看过、簇的解释是否合理，仍由 agent 和人判断。

其余产物的路径、格式和读写关系见 [产物](references/artifacts.md)；已知问题与检测方法见 [已知失效模式](references/failure-modes.md)。

## 工具

```bash
pip install -e <VLA_AGENT>          # 一次；或把 <VLA_AGENT>/src 加到 PYTHONPATH
python -m vla_tools.<module> --help

# 在项目根目录运行
python -m vla_tools.run_status runs/<task>                                   # 各阶段状态
python -m vla_tools.run_index runs/<task> --task <task> --spec runs/<task>/robot_spec.json --source data/<src>
python -m vla_tools.run_index runs/<task> --clean data/<src>-clean             # 每生成一个版本登记一次
python -m vla_tools.run_index runs/<task> --labels <labels> data/<src>-<labels>-train data/<src>-<labels>-eval
python -m vla_tools.data_report runs/<task>                                  # S0–S7 都通过后
```

数据工具只依赖 `pyproject.toml` 中的核心依赖；`replay_eval` 需要 `openpi-client`；`vla_openpi`、`lora_to_pytorch`、`policy_parity` 需要 openpi 环境。`arc_interp.py` 不依赖包内其他模块，可单独拷到服务端。

## 并行

S6 与 S7 的网格搜索互不依赖，可以同时进行。关键帧核对图可按类型分给多个 subagent 查看，但修正文件只有一个写入者：由本流程汇总后写入。

## 停止条件

在 `AGENTS.md` 的停止条件之外：

- 数据的索引约定（`action[t]` 对应哪一帧的状态）无法确认；
- 某类关键帧在核对图上与定义不符，调整检测器和修正文件都无法解决；
- `decisions.md` 中有未决事项；
- 留出集无法覆盖数据的主要簇：如实报告，不给泛化结论；
- 运控约束未知，却需要在超过逐帧基线的 span 之间做选择。

## 完成标准

- `run_status` 报告 S0–S7 全部 `passed`，看图类的放行条件已确认，`data_report.md` 已生成，`decisions.md` 没有未决事项；
- 已验证的数据事实写入 `docs/`，未解决的问题写入 `issues.md`。
