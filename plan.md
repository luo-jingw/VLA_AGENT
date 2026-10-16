# Plan

Plan Status: approved

# Problem

## Current

Resampled labels exist to raise the execution rate a deployed policy runs at: a chunk that covers
`span_frames` recorded frames is executed r times faster than the recording, with fewer inferences per task.
The current evidence does not measure that rate:

- Data: LIBERO only, simulated demonstrations. Its operators' speed follows the path shape (log speed against
  log κ: slope −0.31, R² 0.45; window progress p90 / p10 1.3×), so `time` and `curvature` place the same
  point 1.25 recorded frames apart (median) and rebuild alike: on the frames both leave unpadded, s40 d2
  spline rebuild p50 / p90 / max 0.10 / 0.21 / 1.02 path steps for `time`, 0.09 / 0.21 / 0.72 for `curvature`.
- Rate: `label_replay` reports the largest executed step against the demonstration's p95 step. No robot
  velocity or acceleration limit, gripper actuation time or inference latency enters any figure; no maximum
  rate is computed.
- Selection: `sweep` scores each cell on its own unpadded frames. `time` labels count end-padding rows as
  rows, `curvature` labels count points on them as padded, so s40 d2 is scored on 7708 frames for `time` and
  6313 for `curvature`; the 1395 extra frames lie at the recording's end (rebuild p50 0.06). The selected
  cell beat `time` s40 d2 by 0.01 path steps on these different sets.
- Span: all 30 cells (spans 10–40) are admissible; the largest-admissible-span rule returns the grid's top.
- Latency: the replay is synchronous with zero latency: the robot waits for no inference and a chunk is never
  older than the pose it starts from.

## Problem

1. Nothing measures the quantity the labels are for: the highest task completion speed-up a robot reaches
   with asynchronous inference at a given inference latency, on real teleoperation data.
2. `sweep` compares cells on different frames.

## Goal

On a small subset of DROID (`lerobot/droid_1.0.1`: Franka Panda, VR teleoperation, 15 Hz, joint position
actions and gripper; LeRobot v3.0), per label configuration (per-frame baseline `time` span 10; `time` and
`curvature` at spans 32, 64, 128, 256; keyframe density 0 and 2; H = 10):

- Execution model (closed-loop label replay, the labels standing in for a perfect model), inference latency
  d = ⌈L · fps⌉ control frames:
  - asynchronous: an observation is taken every e control frames (e ≥ max(d, 1)); its chunk, built from the pose
    at the observation, arrives d frames later and is executed from its target d (the targets for the frames
    that already passed are skipped) until the next chunk arrives; the first chunk is executed from target 0
    after the robot has held still for d frames; a chunk that runs out holds its last target;
  - synchronous: the robot holds still for d frames at every observation, then executes e targets.
  L = 0 gives the current replay in both modes.
- Feasible: the executed targets (one per control frame) stay within the Panda's joint velocity and
  acceleration limits (finite differences at the dataset rate, chunk switches and holds included), every
  recorded gripper edge is executed, and at each gripper switch the robot holds for at least the shorter of
  the gripper's actuation time and the recording's own hold there.
- Rates 1, 1.5, 2, 3, 4, 6, 8, 12, 16; e either max(d, 1) (inference back to back) or ⌊span / rate⌋ − d (the
  whole chunk).
- Result per configuration, latency L ∈ {0, 0.05, 0.1, 0.2, 0.4} s and mode: the largest rate feasible on every
  episode and the completion speed-up there (recorded duration / wall-clock duration, holds included); the
  maximum asynchronous speed-up at each latency is the headline figure, the synchronous one is reported beside
  it; per-episode largest feasible rate, median / p10.
- Upper bound: the time-optimal traversal of each recorded path under the same limits (TOPP-RA) plus the
  gripper actuation time at each switch.
- Path deviation of the executed targets in joint space and of the end effector in mm (Panda forward
  kinematics).
- The same on LIBERO (also a Panda) for comparison.
- `sweep`: cells of the same span scored on the frames every cell of that span leaves unpadded (with spans up to
  256 a common set over all spans would keep only the first frames of each episode); the LIBERO sweep re-run.

# Structure

## Modules

- Robot limits: `src/vla_tools/robot_spec.py`.
- Execution model: `src/vla_tools/label_replay.py` (`closed_loop` with latency and mode; per-episode edge and
  hold check).
- Rate ceiling: `src/vla_tools/rate_ceiling.py` (new), built on `label_replay`.
- Sweep: `src/vla_tools/sweep.py`.
- DROID subset: `FTpi05/reports/rate-ceiling-src/extract_droid.py`; data in `FTpi05/data/droid-pick-place*`;
  run in `FTpi05/runs/droid_pick_place/`.
- Reference figures: `FTpi05/reports/rate-ceiling-src/panda_kinematics.py` (forward kinematics),
  `topp_reference.py` (time-optimal bound), `ee_deviation.py` (end-effector deviation of executed targets).
- Knowledge: `docs/validation.md`, `.claude/skills/vla-action-labels/` (SKILL.md, references/method.md),
  `README.md`.

## Responsibilities

`robot_spec` holds the robot's joint limits. `label_replay` owns the execution model (observation rule,
latency, synchronous / asynchronous switching) and the edge and hold check. `rate_ceiling` searches rates and
execution lengths, applies the limits and computes the speed-ups. `sweep` keeps its scores and
selection rule. The FTpi05 scripts prepare the DROID subset and compute the reference figures; they are not
part of the package.

## State Ownership

- Joint limits: each run's `robot_spec.json`, read through `RobotSpec`.
- Rate ceilings: each run's `rate_ceiling.json`, written by `rate_ceiling`.

# Interface

## Interfaces

- `RobotSpec.joint_velocity_limits: tuple[float, ...] = ()`, `RobotSpec.joint_acceleration_limits:
  tuple[float, ...] = ()`: one entry per joint dim (rad/s, rad/s²); empty when unknown; `load` reads them
  when present and checks their length.
- `label_replay.closed_loop(e, spec, record, config, execute, rate, latency_frames: int = 0, mode: str = "sync")
  -> EpisodeReplay`: `EpisodeReplay.executed` is the wall-clock sequence of targets (holds included);
  `EpisodeReplay.stall_frames` counts the frames held for latency or an exhausted chunk. Defaults reproduce the
  current replay exactly.
- `label_replay.edge_check(e, spec, executed, tolerance) -> EdgeCheck`: per recorded command edge, the executed
  position error and hold frames, the recording's own hold frames, and the edges never switched; used by
  `replay` and `rate_ceiling`.
- `rate_ceiling.check(e, spec, record, config, rate, execute, latency_frames, mode, gripper_frames, path_step)
  -> RateCheck`: velocity ratio (max |Δq · fps| / limit), acceleration ratio (max |Δ²q · fps²| / limit) over the
  executed targets preceded by the first observed pose, edges missed, worst hold against its requirement, path
  max (rad), inferences, wall-clock and stalled frames, feasible.
- CLI: `python -m vla_tools.rate_ceiling --dataset --spec --keyframes --horizon --metrics --spans
  --event-density --rates --latency --modes --gripper-seconds --out`; prints per configuration, latency and mode
  the largest rate feasible on every episode, its execution length, the binding constraint and the speed-up.
- `sweep`: CLI unchanged; the cells of one span are scored on the intersection of their unpadded frames; the
  JSON records that frame count per cell.

## Inputs

- DROID: 3 data parquet files (~260 MB, no video) from different parts of the dataset and the episode
  metadata; successful episodes whose instruction is a pick-and-place, with one gripper close and one open
  edge; 60 episodes spread over buildings and collectors.
- Panda joint velocity and acceleration limits from Franka's published specifications; the gripper's
  actuation time from its datasheet (DROID's gripper model confirmed from the DROID paper).
- LIBERO run inputs, unchanged.

## Outputs

`data/droid-pick-place` (extracted) and its timed version; `runs/droid_pick_place/`: `robot_spec.json`,
`audit.json`, `keyframes.json`, `timing_plan.json`, `sweep.json`, `rate_ceiling.json`,
`topp_reference.json`, `ee_deviation.json`; `runs/libero_alphabet_soup/sweep.json` (re-run),
`rate_ceiling.json`.

## State Changes

None beyond the outputs.

# Flow

## Main Flow

DROID files → `extract_droid.py` (LeRobot v2.1 layout, low-dimensional columns) → `audit` →
`keyframe_detect` (gripper edges; no camera review, videos are not downloaded) → `episode_timing` (still
ends trimmed, end padding ≥ the largest span) → `keyframe_detect` on the timed version → `sweep` (common
frames) → `rate_ceiling` per configuration → `topp_reference.py`, `ee_deviation.py` → one table per data set.

# Code Mapping

## Modules

Robot limits → `src/vla_tools/robot_spec.py`; rate ceiling → `src/vla_tools/rate_ceiling.py`; sweep →
`src/vla_tools/sweep.py`; DROID subset and reference figures → `FTpi05/reports/rate-ceiling-src/`.

## Interfaces

`RobotSpec` limits → `robot_spec.py`; `closed_loop`, `edge_check`, `EdgeCheck` → `label_replay.py`; `check`,
`RateCheck` → `rate_ceiling.py`; common-frame scoring → `sweep.py`.

## State

Joint limits → `runs/*/robot_spec.json`; rate ceilings → `runs/*/rate_ceiling.json`.

# Implementation

## Phase 1

Phase Status: pending

Round: 1

### Goal

Joint limits in `RobotSpec`; latency and asynchronous switching in `label_replay`; `rate_ceiling` module and
CLI.

### Files

`src/vla_tools/robot_spec.py`, `src/vla_tools/label_replay.py`, `src/vla_tools/rate_ceiling.py`,
`tests/test_rate_ceiling.py`, `tests/test_label_replay.py`.

### Structures

`EdgeCheck`, `RateCheck`.

### Affected Modules

Robot limits, execution model, rate ceiling.

### Dependencies

None.

### Observation

Tests: latency 0 in both modes equals the current replay; asynchronous switching starts each chunk at target
d and holds d frames only at the start; a straight constant-speed recording gives velocity ratio r · v / limit;
`time` span 10 at rate 1 executes the recording. The LIBERO replay JSONs unchanged under the defaults; LIBERO
rate ceiling printed for `time` s10 and `curvature` s40 d2.

## Phase 2

Phase Status: pending

Round: 1

### Goal

`sweep` scores the cells of one span on their common unpadded frames; LIBERO sweep re-run.

### Files

`src/vla_tools/sweep.py`, `tests/test_sweep.py`, `FTpi05/runs/libero_alphabet_soup/sweep.json`.

### Structures

None.

### Affected Modules

Sweep.

### Dependencies

None.

### Observation

Equal frame count across the cells of a span; LIBERO selection and cell table before / after.

## Phase 3

Phase Status: pending

Round: 1

### Goal

DROID subset through audit, keyframes and timing (end padding ≥ 256 rows).

### Files

`FTpi05/reports/rate-ceiling-src/extract_droid.py`, `FTpi05/data/droid-pick-place*`,
`FTpi05/runs/droid_pick_place/` (`robot_spec.json`, `audit.json`, `keyframes_spec.json`, `keyframes.json`,
`keyframe_overrides.json`, `timing_plan.json`).

### Structures

None.

### Affected Modules

DROID subset.

### Dependencies

None.

### Observation

Episodes, buildings and collectors kept; length distribution; audit findings (index convention, gripper
value distribution, jumps, idle); keyframe issue list; timing report; operators' speed against curvature
(slope, R²) and window progress p90 / p10, against LIBERO's −0.31 and 1.3×.

## Phase 4

Phase Status: pending

Round: 2

### Goal

DROID rate ceiling for every configuration, upper bound, end-effector deviation; LIBERO rate ceiling.

### Files

`FTpi05/reports/rate-ceiling-src/panda_kinematics.py`, `topp_reference.py`, `ee_deviation.py`,
`FTpi05/runs/droid_pick_place/{sweep,rate_ceiling,topp_reference,ee_deviation}.json`,
`FTpi05/runs/libero_alphabet_soup/rate_ceiling.json`. `toppra` 0.6.10 installed into the FTpi05 work
environment only.

### Structures

None.

### Affected Modules

None in the package.

### Dependencies

Phases 1, 2 and 3.

### Observation

Per configuration and latency: the maximum asynchronous speed-up (and the synchronous one), its rate and
execution length, the binding constraint (velocity, acceleration, gripper hold, edge), per-episode largest
rate median / p10, path deviation (joint path steps, end effector mm); against the per-frame baseline and the
time-optimal bound.

## Phase 5

Phase Status: pending

Round: 3

### Goal

Documentation of the rate ceiling and the DROID results.

### Files

`docs/validation.md`, `.claude/skills/vla-action-labels/SKILL.md`,
`.claude/skills/vla-action-labels/references/method.md`, `README.md`.

### Structures

None.

### Affected Modules

Knowledge.

### Dependencies

Phase 4.

### Observation

The tables match the phase 2 and 4 artifacts; the rate is chosen by `rate_ceiling` in the S7 procedure.
