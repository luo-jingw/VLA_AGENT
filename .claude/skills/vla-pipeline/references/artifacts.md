# 产物

## 目录约定

```text
runs/<task>/                     一个任务的计划文件、报告与核对图（小文件）
  run.json                       S0 起，每生成一个数据集版本时更新
  decisions.md                   待人决定的事项
  robot_spec.json                S0
  audit.json, audit_review/      S1
  cleaning_plan.json             S2 输入
  keyframes_spec.json            S3/S5 输入
  keyframes_pass1.json           S3
  timing_plan.json               S4 输入
  keyframe_overrides.json        S5 输入
  keyframes.json, kf_review/     S5
  diversity.json (.png)          S6
  sweep.json, label_config.json  S7
  replay_<config>.json           S7
  data_report.md                 S7 通过后
  replay_spec.json               S8 输入
  eval_<model>.json              S8
  parity_*.npz                   S8
data/<source>                    源数据，只读
data/<source>-clean              S2
data/<source>-timed              S4
data/<source>-<labels>-train     S7，去掉留出集
data/<source>-<labels>-eval      S7，全部 episode，编号与定时版一致
```

`<labels>` 写出标签配置的关键量，例如 `time-s20`。

## 写入者与读取者

| 产物 | 写入者 | 读取者 |
|---|---|---|
| `robot_spec.json` | S0（手写） | 所有工具 |
| `run.json` | `vla-pipeline` | `run_status`、`data_report` |
| `decisions.md` | `vla-pipeline` | 人；`run_status` |
| `audit.json` | `audit` | S2 决策 |
| 清洗版数据集 + `meta/cleaning_report.json` | `clean` | S3、S4 |
| `keyframes_pass1.json` | `keyframe_detect` | `episode_timing` |
| 定时版数据集 + `meta/timing_report.json` | `episode_timing` | S5 起所有阶段 |
| `keyframes.json` | `keyframe_detect` | `keyframe_review`、`diversity`、`sweep`、`label_replay`、`build_labels` |
| `diversity.json` | `diversity` | `build_labels --exclude`、`coverage_eval` |
| `sweep.json` | `sweep` | S7 选择 |
| 标签数据集 + `meta/arc_resample.json` | `build_labels` | S8 下游、`data_report` |
| `data_report.md` | `data_report` | 人；S8 下游 |
| `eval_<model>.json` | `replay_eval` | `coverage_eval` |

## 文件格式

### robot_spec.json

```json
{"action_dim": 14,
 "joint_dims": [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
 "command_dims": [6, 13],
 "state_dims": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
 "fps": 15,
 "action_key": "action", "state_key": "observation.state",
 "image_keys": ["observation.images.cam_high", "observation.images.cam_left_wrist"]}
```

- `joint_dims`：连续的位置量，参与度量、插值和清洗。连续夹爪开度、灵巧手手指关节也属于这一类。
- `command_dims`：0/1 指令（二值夹爪、吸盘开关），不插值。
- `state_dims`：每个动作维对应的状态维，没有对应时写 -1。动作与状态布局不同（如状态里还有腰、头、真空度）时，靠它把状态换算到动作布局。

### run.json

```json
{"task": "ur3_simple_pick",
 "spec": "runs/ur3_simple_pick/robot_spec.json",
 "datasets": {"source": "data/ur3-test",
              "clean": "data/ur3-test-clean",
              "timed": "data/ur3-test-timed",
              "labels": {"time-s20": {"train": "data/ur3-test-time-s20-train",
                                      "eval": "data/ur3-test-time-s20-eval"}}}}
```

相对路径以项目根（`runs/` 的上一级）为基准。还没生成的版本不写。

### decisions.md

每条事项一个 `## D-NNN` 标题，下面固定七行：

```markdown
## D-001
- stage: S2
- status: pending
- item: 第 7 集第 120 帧的断点跳变：修复或保留
- evidence: runs/<task>/audit_review/ep007_f120.png
- proposal: 保留；前后 5 帧是连续运动
- decision:
- written_to:
```

`status` 只取 `pending` 或 `decided`。人做出决定后填写 `decision`，agent 把它写进对应文件，填写 `written_to`，再把 `status` 改为 `decided`。

### cleaning_plan.json、timing_plan.json、keyframes_spec.json、keyframe_overrides.json

格式分别写在 `clean`、`episode_timing`、`keyframe_spec`、`keyframe_detect` 模块的说明里（`python -m vla_tools.<module> --help`）。共同规则：凡是需要判断的处理，都要有一条决定和理由；工具对缺少决定的情况拒绝写出。

### label_config.json

```json
{"metric": "time", "span_frames": 20, "action_horizon": 10, "event_density": 0.0,
 "key_weight": 3.0, "pad_weight": 1.0, "deadzone": 0.0, "anchor": "state"}
```

`anchor` 必须写明，缺失时工具报错。

### arc_resample.json

由 `build_labels` 写在标签数据集的 `meta/` 下，内容包括 `robot`（机器人布局）、`config`（标签配置）、`keyframes`（所用关键帧文件）、`excluded_source_episodes`、`episode_map` 和每个 episode 的统计。训练完成后，把它拷到 checkpoint 的 `assets/<asset_id>/` 下；服务端用 `ArcInterpolator.from_report` 读取。

### replay_spec.json

```json
{"images": {"observation/image": "observation.images.cam_high"},
 "state": {"observation/state": "observation.state"},
 "prompt_key": "prompt", "static_dims": [7, 8, 9, 10, 11, 12]}
```

用于把客户端的观测键映射到数据集列。`static_dims` 来自体检结果。
