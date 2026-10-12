# VLA_AGENT

An agent harness for VLA (vision-language-action) policy work: robot demonstration data → cleaned,
keyframed, evaluated datasets → action labels → openpi π0/π0.5 training → evaluation → serving.

面向 VLA 策略的 agent harness：把机器人示教数据处理成可训练的数据集，构建动作标签，接入 openpi 训练，
按统一口径评估，并交付到服务端。它由三部分组成：

- **工作约定与通用 skill**：由 [explicit-agent](https://github.com/luo-jingw/AgentTemplate) 初始化
  （`AGENTS.md`、`plan` / `work` / `experiment` / `close` 等）；
- **VLA skill 与工具**：`.claude/skills/vla-*` 九个 skill 及其共享工具包 `vla_tools`；
- **openpi 接入**：`integrations/openpi/`，包括对上游 openpi 的补丁、通用 transform 与数据配置、启动函数。

## 流程

| 阶段 | skill | 做什么 |
|---|---|---|
| S0 | `vla-pipeline` | 范围、机器人布局、运控约束；调度以下各阶段 |
| S1–S2 | `vla-data-cleaning` | 体检（索引约定、静止维、时间戳、指令毛刺、相机冻结、重复、跳变）与按书面决定清洗 |
| S3、S5 | `vla-keyframes` | 按任务与末端执行器定义关键帧，检测、一致性检查、逐类看图核对、记录修正 |
| S4 | `vla-episode-timing` | 静止帧保留或裁掉；末尾补帧，让策略学会完成后保持 |
| S6 | `vla-data-diversity` | 覆盖、聚类、近重复、离群；分层留出集；按覆盖解读模型误差 |
| S7 | `vla-action-labels` | 动作块标签：按服务端实际执行的误差做网格搜索，选执行长度，构建训练/评估版 |
| S8 | `vla-training` | openpi 配置、静止维归一化、loss 权重、基线对照 |
| S9 | `vla-eval` | 放置误差下限、留出集开环回放、完成后保持、覆盖分层、闭环协议 |
| S10 | `vla-serving` | 服务端插值、推理配置、LoRA 合并转 PyTorch 与一致性检查、交付 |

每个阶段的产物只有一个写入者，凡是需要判断的处理都要写明决定和理由，缺少时工具拒绝写出。
细节见 `.claude/skills/vla-pipeline/SKILL.md`。

## 循环

agent 在两层循环里工作。外层沿阶段向下推进，每个阶段过了放行条件才进入下一阶段；评估和交付发现的问题沿虚线回到产生它的阶段，收尾之后进入下一个任务。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/pipeline-loop-dark.svg">
  <img alt="VLA 流程的推进与回环：S0 范围、S1–S2 体检与清洗、S3–S5 关键帧与补帧、S6 数据评估、S7 动作标签、S8 训练、S9 模型评估、S10 交付、S11 收尾；评估回到标签或数据，交付回到训练，收尾进入下一个任务" src="docs/figures/pipeline-loop-light.svg" width="680">
</picture>

图 1　外层循环：agent 如何沿流程推进，以及在哪里回到前面的阶段。

内层是每个阶段内部的循环：运行工具得到新版本或报告，观察统计和核对图，把判断写成带理由的决定，再做放行检查；没有通过就调参数、改定义后重跑，或者停下报告阻塞。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/stage-loop-dark.svg">
  <img alt="一个阶段内部的循环：运行工具、观察、决定、放行检查；未通过时回到运行工具，通过后进入下一阶段" src="docs/figures/stage-loop-light.svg" width="680">
</picture>

图 2　内层循环：一个阶段内部怎么推进。决定文件是唯一需要人判断的产物，也是审查 agent 行为的入口。

两张图由 `docs/figures/make_figures.py` 生成（浅色和深色各一份，几何相同），修改后重新运行即可。

## 目录

```text
AGENTS.md, PROJECT.md, plan.md, issues.md, opportunities.md, docs/, templates/   explicit-agent 约定与状态文件
.claude/skills/{plan,work,parallelize,experiment,close,...}                     通用 skill（模板）
.claude/skills/vla-*/                                                           VLA skill
.claude/skills/vla-pipeline/tools/vla_tools/                                    工具包（python -m vla_tools.<module>）
integrations/openpi/patches/                                                    上游 openpi 补丁
integrations/openpi/vla_openpi/                                                 通用 transform、数据配置、启动函数
examples/ur3_dual_arm/                                                          双臂 UR3 的完整配置示例
docs/validation.md                                                              已验证的行为
docs/figures/                                                                   README 中的循环图及其生成脚本
```

## 使用

**环境**：Python 3.11。数据处理工具只依赖 `requirements.txt`；训练、服务、转换、一致性检查需要 openpi
的环境，并打上 `integrations/openpi/patches/` 中的补丁。

**两种方式**：
1. 以本仓库为项目根：数据放在 `data/`，每个任务的计划与报告放在 `runs/<task>/`，从 `vla-pipeline` skill 开始；
2. 在已用 explicit-agent 初始化的项目中，拷贝 `.claude/skills/vla-*` 与 `integrations/openpi/`。

**命令链**（以 UR3 示例为例）：

```bash
export PYTHONPATH=.claude/skills/vla-pipeline/tools
python -m vla_tools.audit           --dataset data/<src> --spec robot_spec.json --out runs/<task>/audit.json --review-dir runs/<task>/audit_review
python -m vla_tools.clean           --dataset data/<src> --spec robot_spec.json --plan cleaning_plan.json --out data/<src>-clean
python -m vla_tools.keyframe_detect --dataset data/<src>-clean --spec robot_spec.json --keyframes-spec keyframes_spec.json --out runs/<task>/keyframes_pass1.json
python -m vla_tools.episode_timing  --dataset data/<src>-clean --spec robot_spec.json --keyframes runs/<task>/keyframes_pass1.json --plan timing_plan.json --out data/<src>-timed
python -m vla_tools.keyframe_detect --dataset data/<src>-timed --spec robot_spec.json --keyframes-spec keyframes_spec.json --out runs/<task>/keyframes.json
python -m vla_tools.keyframe_review --dataset data/<src>-timed --spec robot_spec.json --keyframes runs/<task>/keyframes.json --type grasp --camera <col> --out runs/<task>/kf_review
python -m vla_tools.diversity       --dataset data/<src>-timed --spec robot_spec.json --keyframes runs/<task>/keyframes.json --holdout 4 --out runs/<task>/diversity.json
python -m vla_tools.sweep           --dataset data/<src>-timed --spec robot_spec.json --keyframes runs/<task>/keyframes.json --horizon 10 --spans 10 20 30 --out runs/<task>/sweep.json
python -m vla_tools.build_labels    --dataset data/<src>-timed --spec robot_spec.json --keyframes runs/<task>/keyframes.json --config label_config.json --exclude 3 4 11 19 --out data/<src>-time-s20-train
```

训练与服务见 `integrations/openpi/README.md` 和 `examples/ur3_dual_arm/run.py`。

## 约定

本仓库按 `AGENTS.md`（explicit-agent 0.1.0, `964b9d9`）管理：模板自带的文件不手改，项目内容写入
`PROJECT.md`、`docs/`、`issues.md`、`opportunities.md`；VLA skill 是项目自有 skill。
