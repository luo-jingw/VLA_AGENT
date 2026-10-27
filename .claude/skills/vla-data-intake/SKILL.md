---
name: vla-data-intake
description: 把原始录制接入成流程的源数据集：转换成 LeRobot v2.1（转换前用原始时间戳查丢帧），逐条看关键画面排除不能训练的 episode，为每条 episode 标注多条描述其实际内容的指令。属于 vla-pipeline 的 S0，在体检与清洗之前；源数据已经是 LeRobot、且每条都有准确指令时跳过。
---

# 原始录制接入

## 为什么单独一步

- 后面的工具都读 LeRobot v2.1，而录制器各有自己的格式（每集一个 HDF5、ROS bag 等）。
- 转换会把时间戳改写成连续的 `frame / fps`。录制里的丢帧在转换后就看不出来了：S1 体检检查的是转换后的时间戳。所以丢帧只能在转换时、用录制器自己的时间戳查。
- 有些问题只在画面里：有人伸手进场景、任务失败或被撤销、拿错物体、场景被人重置。数值检查发现不了。
- 录制自带的指令常是占位符（任务名，或每集同一句话）。同一个场景里不同 episode 做的事不同时，模型只能从指令知道这一次要做什么。

## 输入

原始录制目录（只读）、录制器的格式说明、`robot_spec.json`（见 `vla-pipeline` 的 S0）。

## 步骤

### 1. 数值概况

转换前逐集列出：步数、实测帧率、相邻两步的最大间隔、每个执行器指令的切换次数、各臂是否运动。帧率不足或有丢帧的集、完全没有运动的集，记下理由，交给人决定是否排除。

### 2. 转换

转换脚本遵守下面的约定，它们是后续所有工具的前提：

| 列 | 约定 |
|---|---|
| `observation.state` | 关节位置，加上每个夹爪的指令值（0 张开、1 闭合）；维度顺序与 `robot_spec.json` 一致 |
| `action` | 下一行的 `observation.state`，最后一行重复自身（`action[t] = state[t+1]`，S1 会核对） |
| 图像 | 录制时的原始编码字节（如 JPEG）原样写入 parquet，不重新编码、不缩放 |
| `timestamp`、`frame_index`、`index` | 连续改写 |
| `task_index` | 一个占位任务，由第 4 步替换 |

- 保留的 episode 只要相邻两步的间隔超过 1.5 帧，转换就停下，要求在 `conversion.json` 里排除并写理由。
- 输出 `meta/conversion_report.json`：来源（仓库与版本）、排除的集及理由、原始编号到新编号的 `episode_map`、每集的最大步间隔。

UR3 录制器（embodyx `episode_writer.py` 写的每集一个 HDF5）的实现是 `examples/ur3_dual_arm/convert_hdf5.py`；其他录制器按同样的约定另写：

```bash
pip install -e <VLA_AGENT>[hdf5]
python examples/ur3_dual_arm/convert_hdf5.py --raw <raw_dir> --conversion runs/<task>/conversion.json \
    --task "<占位指令>" --out data/<src>
```

`conversion.json`：`{"source": "<仓库>@<版本>", "exclude": {"<原始编号>": "<理由>", ...}}`。

### 3. 逐条看画面

```bash
python -m vla_tools.episode_sheets --dataset data/<src> --spec runs/<task>/robot_spec.json \
    --cameras <俯视相机列> <腕部相机列> --every 40 --out runs/<task>/sheets
```

每集一张图：开头、每次指令切换后 5 帧、每隔 40 帧、结尾，每行一个相机。逐张看，记下：

- 这一集实际做了什么：哪个物体、从哪里到哪里、做了几次。这是第 4 步的依据；
- 不能训练的集：画面里有人、任务失败或被撤销、物体不对、场景被重置。拿不准时把那一段放大再看；
- 指令切换是否对应真实的抓和放，例如开头有一次什么都没夹的开合。

只看开头、切换和结尾不够：录制末尾的自动动作段（例如回零）里可能有人在调整场景，而结尾时人已经离开。

排除由人决定，写进 `conversion.json` 后重新转换。新编号会变，第 4 步按原始编号标注，所以不受影响。

### 4. 指令标注

每集写一句场景描述 `scene`（画面里实际发生的事）和 5 条英文指令。5 条的写法各不相同：

1. 简短命令句：颜色、物体、目标，例如 "Put the orange cube on the white box."；
2. 带起点的完整描述；多步任务按顺序写出每一步；
3. 同义改写（cube / block，put / place / move / stack）；
4. 礼貌问句，例如 "Could you ...?"；
5. 短语，例如 "orange cube onto white box"。

- 只写画面里确实发生的事。多步任务写全所有步骤和顺序；失败后重抓不写进指令，只写最终结果，在 `scene` 里记一句。
- 同一个动作在不同集里可以用相同的措辞。
- 核对：多步的集逐条复核，其余抽查；拿不准时放大画面。

写成 `runs/<task>/prompts.json`，以原始编号为键：`{"<原始编号>": {"scene": "...", "prompts": ["...", ...]}}`，再写进数据集：

```bash
python -m vla_tools.prompt_annotation --dataset data/<src> --prompts runs/<task>/prompts.json \
    --source-map data/<src>/meta/conversion_report.json --out data/<src>-annotated
```

- 第一条指令是每一行的 `task`；5 条都写进 `meta/episodes.jsonl` 的 `tasks`。之后的清洗版、定时版、标签版都原样带着它们。
- 训练时用 `LeRobotArcDataConfig(rotate_prompts=True)`，每个样本从所在集的指令里随机抽一条（见 `vla-downstream`）。
- 标注版是 S1 的源数据集。

## 决策权限

| 决定 | 谁定 | 记录在 |
|---|---|---|
| 排除 episode（帧率、丢帧、没有运动、画面问题） | 人，agent 提出证据 | `conversion.json` |
| 指令标注 | agent 写，人抽查 | `prompts.json` |

## 停止条件

- 录制器格式不明，无法确定状态、指令和图像的对应关系；
- 画面里的问题无法判断是否影响训练：交给人决定。

## 完成标准

- `conversion.json` 里每个排除都有理由；有 `conversion_report.json`；每集都有拼图并且看过；
- `prompts.json` 覆盖全部保留的集，标注版数据集已生成，之后进入 S1。

## UR3 上的结果（参考量级）

双臂 UR3（只用右臂）遥操作抓放，30 Hz，2026-10-26：

- 58 集原始录制，排除 6 集：4 集有 4.5–6.3 s 的丢帧（实测 2.7–20.6 Hz）；1 集完全没有运动；1 集在自动回零期间有人伸手挪动方块，结尾帧看不到，每 40 帧一张的画面里能看到；
- 转换后 52 集、19,857 行；图像 320×240（俯视）和 424×240（腕部），JPEG 原样；
- 场景比数据集名字多样：除了方块，还有勺子、杯子、小黑块；有叠塔、拆塔，每集 1–3 步。260 条指令，其中 222 条不同；
- 训练时轮换指令：同一样本抽 200 次，5 条各出现 29–46 次。
