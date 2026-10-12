# 按末端执行器定义关键帧

每种末端执行器的成败时刻、可用信号和检测器不同。下面是常见组合的起点，具体阈值需要按数据和核对图确定。

## 二值命令平行夹爪，无反馈

例如 UR3 双臂。夹爪状态只是延迟一帧的命令回显，没有接触或开度反馈。

| 关键帧 | 检测器 | 信号 | 窗口 | 说明 |
|---|---|---|---|---|
| grasp | `command_edge` rise | 动作的夹爪维 | 前 8 / 后 5 | 闭合命令发出的时刻 |
| pre_grasp_stop | `speed_minimum`，参照 grasp，向前搜 20 帧 | `joint_speed` | 0 / 0 | 最后接近阶段最慢的一帧 |
| lift | `motion_onset`，参照 grasp，向后搜 40 帧 | `joint_speed`，阈值约 1 倍帧运动量，持续 3 帧 | 前 2 / 后 6 | 抓取后开始抬起 |
| release | `command_edge` fall | 动作的夹爪维 | 前 8 / 后 5 | 张开命令发出的时刻 |

在 UR3 数据上：抬起阈值设为 0.02（约 1.5 倍帧运动量）时，有 2 集检测偏晚，降到 0.012、持续 3 帧后全部正常。释放的窗口在每一集都越界，因为录制在张开后 1–2 帧就停止了，需要交给 `vla-episode-timing` 补帧。

## 带开度反馈的连续夹爪

| 关键帧 | 检测器 | 信号 |
|---|---|---|
| close_start | `threshold_cross` fall，开度命令低于"张开"阈值 | 动作或状态中的开度维 |
| contact | `settle`，参照 close_start：开度反馈在若干帧内不再变化 | 开度反馈 |
| release | `threshold_cross` rise | 开度命令 |

开度维属于关节维（`joint_dims`），参与插值。抓到物体时开度停在物体宽度处，`settle` 比命令更能说明抓取已经完成。

## 带真空反馈的吸盘

例如 AgiBot G2。吸盘命令是二值的，真空度是反馈。

| 关键帧 | 检测器 | 信号 | 窗口 |
|---|---|---|---|
| suction_on | `command_edge` rise | 动作的吸盘命令维 | 前 15，`until: seal` |
| seal | `threshold_cross` rise，参照 suction_on，持续 15 帧 | 状态或列中的真空度 | 0 / 0 |
| suction_off | `command_edge` fall | 吸盘命令维 | 前 0，后到真空度泄掉为止 |

`until` 让窗口一直延伸到密封完成，窗口长度是实测值，因此每集不同。一侧通道密封、另一侧没有时，按"至少一路密封"定义 seal，并在任务描述中写明。

## 灵巧手

手指关节是关节维。闭合程度需要先写成一列，例如各手指屈曲角之和，再用 `column` 信号读取：

| 关键帧 | 检测器 | 信号 |
|---|---|---|
| grasp_close | `threshold_cross` rise | 闭合程度列 |
| grasp_settled | `settle`，参照 grasp_close | 闭合程度列 |
| release | `threshold_cross` fall | 闭合程度列 |

## 接触、放置、插入（无专门指令的时刻）

| 时刻 | 检测器 | 信号 |
|---|---|---|
| 放下前稳住 | `speed_minimum`，参照 release，向前搜 | `joint_speed` |
| 接触 | `threshold_cross` rise | 力/力矩模长列（如有） |
| 末端到达最低点 | `speed_minimum` 或 `threshold_cross` | 由正运动学写出的末端高度列 |

没有力传感器时，接触只能由运动学间接判断，核对图是唯一的确认手段。

## 双臂交接

每只手各自定义关键帧，并用顺序约束交接：`grasp_right` → `grasp_left`（参照 grasp_right）→ `release_right`（参照 grasp_left）。次数不固定时把 `expected` 设为 -1，并在核对图上逐集确认。
