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
- Measured on LIBERO (50 episodes) against the Panda's libfranka velocity limits (2.175 rad/s joints 1–4,
  2.61 rad/s joints 5–7), finite differences at 20 Hz:
  - the recordings reach median 0.30, max 0.51 of the velocity limit at their fastest frame;
  - with latency, a chunk planned from a pose d frames old and started at its target d does not meet the
    commanded pose at the switch (prototype, `curvature` s40 d2, rate 1, d = 2 frames: largest per-frame step
    6.0 path steps, 3.4 with the arrival rebuild below; at d = 4, rate 2 feasible on 47 of 50 episodes without
    the rebuild, 50 with it);
  - `rate_ceiling` (phase 1), rate feasible on every episode and speed-up there, asynchronous / synchronous:

    | labels | L = 0 | L = 0.1 s | L = 0.2 s | L = 0.4 s |
    |---|---|---|---|---|
    | per-frame `time` s10 | 1.5, 1.48 | 1.5, 1.45 / 1.5, 1.11 | 1, 0.95 / 1.5, 0.89 | none / 1.5, 0.63 |
    | `curvature` s40 d2 | 2, 1.57 | 2, 1.52 / 2, 1.38 | 2, 1.48 / 2, 1.27 | 1.5, 1.08 / 2, 1.09 |

    the next rate fails on velocity; per-frame rate 2 fails on one episode at 1.02 of the limit. A span-10
    chunk keeps 0.5 s of motion: at L = 0.2 s it runs out above rate 1, the robot holds and catches up with
    a step beyond the limit; at L = 0.4 s no asynchronous rate is feasible;
  - every recorded gripper edge is executed in every run;
  - the executed targets exceed the Panda's nominal acceleration limits already for the per-frame labels at
    rate 1 (1.24×, the recording's own frame-to-frame changes).

- Measured during phase 4 (2026-10-16), DROID subset (55 episodes after the velocity exclusion) and LIBERO:
  - velocity bounds of the recorded paths (`velocity_bound.py`): uniform replay feasible on every episode
    up to 1.07× on DROID (per episode median 1.88×), 1.96× on LIBERO (3.30×); every step at the limit of its
    binding joint 7.65× and 5.95×. The fastest DROID recordings reach 0.84–0.93 of the limit at rate 1;
  - DROID sweep: `curvature` beats `time` at span 32 (path full median 0.31 against 0.34–0.37 path steps)
    and 64 (0.78 against 1.01–1.07); spans 128 and 256 are inadmissible (path head median 1.4–4.5); selected
    `curvature` s64 d2;
  - without a path bound the rate ceiling rewards cutting corners: at spans 64–256 with e = 1, rate 1, the
    replay finishes 1.3–1.9× faster than recorded while the end effector leaves the recorded path by a
    median 3.5–14 mm and up to 113 mm; closed loop at small e drifts farther than the open-loop sweep
    measures;
  - wall-clock time counts the hold rows a long last chunk executes after the recording's end, and at L = 0
    with e = 1 the replay stops one row short of the final recorded pose;
  - "feasible on every episode" is decided by one frame of one episode: on DROID the result jumps between
    neighbouring latencies (`curvature` s64 d2 with a 2-path-step bound and completion timing: L = 0 1.34×,
    L = 0.05 0.89×, L = 0.1 none, L = 0.2 0.87×);
  - with one execution rate, every window of span recorded frames runs in span / r frames, so the fastest
    window of the recording sets the ceiling whatever the label metric; `curvature` placement moves faster
    on straights, where the velocity limit binds.

## Problem

1. Nothing measures the quantity the labels are for: the highest task completion speed-up a robot reaches
   with asynchronous inference at a given inference latency, on real teleoperation data.
2. `sweep` compares cells on different frames.
3. The rate ceiling as defined in phase 1 does not measure that speed-up faithfully: it admits runs that cut
   corners (no path bound), counts the hold rows after the task is done and stops one recorded row short of
   the final pose at small e (completion), and lets one frame of one episode decide the result (every
   episode feasible).

## Goal

On a small subset of DROID (`lerobot/droid_1.0.1`: Franka Panda, VR teleoperation, 15 Hz, joint position
actions and gripper; LeRobot v3.0), per label configuration (per-frame baseline `time` span 10; `time` and
`curvature` at spans 32, 64, 128, 256; keyframe density 0 and 2; H = 10):

- Execution model (closed-loop label replay, the labels standing in for a perfect model), inference latency
  d = ⌈L · fps⌉ control frames:
  - asynchronous: an observation is taken every e control frames (e ≥ max(d, 1)); its chunk arrives d frames
    later, is rebuilt from the commanded pose (arrival rebuild) and executed until the next chunk arrives; the
    first chunk, built from the observed pose, is executed from target 0 after the robot has held still for d
    frames; a chunk that runs out holds its last target; a rate whose chunk keeps no target after the latency
    has no run;
  - synchronous: the robot holds still for d frames at every observation, then executes e targets;
  - end: once an observation reaches the last recorded frame, its chunk is executed and the robot keeps
    observing that frame until the executed pose is within the final tolerance (0.25 path steps, the hold
    tolerance) of the final recorded pose with the final command state; the chunk of the last recorded frame
    is always executed.
  L = 0 gives the same switching in both modes.
- Arrival rebuild (async, d > 0): when a chunk arrives d frames after its observation, the server rebuilds it
  from the currently commanded pose: knots = the commanded pose at 0 and the points whose time minus d exceeds
  half a point spacing, at their times minus d; the same not-a-knot spline. The switch is continuous in
  position. A chunk with no point left holds the commanded pose.
- Completion: frames until a joint sequence reaches the final recorded pose and stays there (within the final
  tolerance, final command state), counted the same way for the executed targets (holds included) and for
  the recorded actions. Speed-up = recorded completion frames / executed completion frames; the rows the
  robot holds after completion are not counted.
- Feasible run: the executed targets (one per control frame, the first observed pose before them) stay within
  the Panda's joint velocity limits (finite differences at the dataset rate, chunk switches and holds
  included); every recorded gripper edge is executed; the closed-loop path deviation (each executed chunk
  target to the recorded segment its chunk covers) stays within 2 path steps on the whole episode (about
  10 mm of end-effector motion on DROID). The gripper's timing near an edge is carried by the labels'
  keyframe density; the rate scales it with the rest of the motion.
- Reported, not constrained: the largest acceleration as a fraction of the Panda's nominal limits and the
  largest per-frame step at a chunk switch. A deployed controller's acceleration is a setting; the measured
  quantity is the speed-up within the velocity limits.
- Episodes whose recording exceeds a velocity limit are left out of every case and listed. Path step and
  curvature chord length come from all episodes, as in `sweep`.
- Rates 1, 1.5, 2, 3, 4, 6, 8, 12, 16; e either max(d, 1) (inference back to back), half or all of the targets
  left after the latency (async), half or all of the chunk (sync). Per episode the search stops after two
  consecutive rates without a run within the velocity limit.
- Result per configuration, latency L ∈ {0, 0.05, 0.1, 0.2, 0.4} s and mode: the (rate, e) with the largest
  speed-up among those feasible on at least 90% of the episodes, the speed-up summed over the episodes
  feasible there, and those left out; per episode the largest feasible rate and the largest feasible
  speed-up, median / p10. The maximum asynchronous speed-up at each latency is the headline figure, the
  synchronous one is reported beside it.
- Upper bound: the time-optimal traversal of each recorded path under the same velocity limits: every
  recorded step at the speed of its limiting joint, duration Σ_k max_j |Δq_kj| / v_j.
- Path deviation of the executed targets up to completion in joint space and of the end effector in mm
  (Panda forward kinematics), on the episodes feasible at the headline setting.
- The same on LIBERO (also a Panda) for comparison.
- `sweep`: cells of the same span scored on the frames every cell of that span leaves unpadded (with spans up to
  256 a common set over all spans would keep only the first frames of each episode); the LIBERO sweep re-run.
- `label_replay` (geometric replay) uses the same end rule; its frame counts are completion frames. The
  LIBERO replays are re-run.

# Structure

## Modules

- Robot limits: `src/vla_tools/robot_spec.py`.
- Server rebuild: `src/vla_tools/arc_interp.py` (arrival rebuild from the commanded pose).
- Execution model: `src/vla_tools/label_replay.py` (`closed_loop` with latency, mode and the end rule;
  completion frames; per-episode edge check).
- Rate ceiling: `src/vla_tools/rate_ceiling.py`, built on `label_replay`.
- Sweep: `src/vla_tools/sweep.py`.
- DROID subset: `FTpi05/reports/rate-ceiling-src/extract_droid.py`; data in `FTpi05/data/droid-pick-place*`;
  run in `FTpi05/runs/droid_pick_place/`.
- Reference figures: `FTpi05/reports/rate-ceiling-src/panda_kinematics.py` (forward kinematics),
  `velocity_bound.py` (time-optimal bound), `ee_deviation.py` (end-effector deviation of executed targets).
- Knowledge: `docs/validation.md`, `.claude/skills/vla-action-labels/` (SKILL.md, references/method.md),
  `README.md`.

## Responsibilities

`robot_spec` holds the robot's joint limits: velocity is a constraint, acceleration is reported. `arc_interp`
owns every rebuild of a chunk into per-frame targets, including the arrival rebuild. `label_replay` owns the
execution model (observation rule, latency, synchronous / asynchronous switching, the end rule), completion
frames and the edge check. `rate_ceiling` searches rates and execution lengths, applies the velocity limit,
the edge check and the path bound, and computes the speed-ups and the 90% summary. `sweep` keeps its scores
and selection rule. The FTpi05 scripts prepare the DROID subset and compute the reference figures; they are
not part of the package.

## State Ownership

- Joint limits: each run's `robot_spec.json`, read through `RobotSpec`.
- Rate ceilings: each run's `rate_ceiling.json`, written by `rate_ceiling`.

# Interface

## Interfaces

- `RobotSpec.joint_velocity_limits: tuple[float, ...] = ()`, `RobotSpec.joint_acceleration_limits:
  tuple[float, ...] = ()`: one entry per joint dim (rad/s, rad/s²); empty when unknown; `load` reads them
  when present and checks their length.
- `arc_interp.after_latency(points, times, commanded, elapsed, command_dims) -> np.ndarray`: per-frame targets
  from the commanded pose, `elapsed` frames after the observation; `ArcInterpolator.after_latency(points,
  commanded, elapsed_frames)` for serving.
- `label_replay.closed_loop(e, spec, record, config, execute, rate, latency_frames: int = 0, mode: str = "sync",
  *, final_tolerance: float) -> EpisodeReplay`: `EpisodeReplay.executed` is the wall-clock sequence of targets
  (holds included) ending within `final_tolerance` (rad) of the final recorded pose;
  `EpisodeReplay.completion_frames` its completion; `EpisodeReplay.stall_frames` counts the frames held for
  latency or an exhausted chunk.
- `label_replay.completion_frames(targets, final, spec, tolerance) -> int`: frames until `targets` reach the
  (D,) `final` row and stay within `tolerance` of its joints with its command state; raises when the last
  target is not there.
- `label_replay.edge_check(e, spec, executed, tolerance) -> EdgeCheck`: per recorded command edge, the executed
  position error and hold frames, the recording's own hold frames, and the edges never switched; used by
  `replay` and `rate_ceiling`.
- `label_replay.ReplayRow.executed_frames`, `recorded_frames`: completion frames summed over episodes.
- `rate_ceiling.check(e, spec, record, config, rate, execute, latency_frames, mode, path_step, path_bound)
  -> RateCheck`: velocity ratio (max |Δq · fps| / limit) over the executed targets preceded by the first
  observed pose, edges missed, path max (path steps) against `path_bound`, executed and recorded completion
  frames; reported: acceleration ratio (max |Δ²q · fps²| / nominal limit), largest switch step (path steps),
  inferences, stalled frames.
- `rate_ceiling.summarize(case, checks, episodes, fps, rates, feasible_fraction) -> CaseResult`: the
  largest-speed-up (rate, e) feasible on at least `feasible_fraction` of the episodes, the episodes left out
  there, per-episode largest rate and speed-up.
- CLI: `python -m vla_tools.rate_ceiling --dataset --spec --keyframes --horizon --per-frame --metrics --spans
  --event-density --rates --latency --modes --path-bound 2 --feasible-fraction 0.9 --out`; prints per
  configuration, latency and mode the selected rate, its execution length, the episodes feasible there, the
  speed-up and the constraints failing at the next rate.
- `sweep`: CLI unchanged; the cells of one span are scored on the intersection of their unpadded frames; the
  JSON records that frame count per cell.

## Inputs

- DROID: 3 data parquet files (~260 MB, no video) from different parts of the dataset and the episode
  metadata; successful episodes whose instruction is a pick-and-place, with one gripper close and one open
  edge; 60 episodes spread over buildings and collectors.
- Panda joint velocity and acceleration limits: libfranka 0.9.2 `include/franka/rate_limiting.h`.
- LIBERO run inputs, unchanged.

## Outputs

`data/droid-pick-place` (extracted) and its timed version; `runs/droid_pick_place/`: `robot_spec.json`,
`audit.json`, `keyframes.json`, `timing_plan.json`, `sweep.json`, `rate_ceiling.json`,
`velocity_bound.json`, `ee_deviation.json`; `runs/libero_alphabet_soup/`: `sweep.json` (re-run),
`rate_ceiling.json`, `ee_deviation.json`, `velocity_bound.json`, the five `replay_*.json` (re-run).

## State Changes

None beyond the outputs.

# Flow

## Main Flow

DROID files → `extract_droid.py` (LeRobot v2.1 layout, low-dimensional columns) → `audit` →
`keyframe_detect` (gripper edges; no camera review, videos are not downloaded) → `episode_timing` (still
ends trimmed, end padding ≥ the largest span) → `keyframe_detect` on the timed version → `sweep` (common
frames) → `rate_ceiling` per configuration (path bound, completion, 90% summary) → `velocity_bound.py`,
`ee_deviation.py` at the selected settings → one table per data set.

# Code Mapping

## Modules

Robot limits → `src/vla_tools/robot_spec.py`; execution model → `src/vla_tools/label_replay.py`; rate
ceiling → `src/vla_tools/rate_ceiling.py`; sweep → `src/vla_tools/sweep.py`; DROID subset and reference
figures → `FTpi05/reports/rate-ceiling-src/`.

## Interfaces

`RobotSpec` limits → `robot_spec.py`; `after_latency` → `arc_interp.py`; `closed_loop`, `completion_frames`,
`edge_check`, `EdgeCheck`, `ReplayRow` → `label_replay.py`; `check`, `summarize`, `RateCheck`, `CaseResult` →
`rate_ceiling.py`; common-frame scoring → `sweep.py`.

## State

Joint limits → `runs/*/robot_spec.json`; rate ceilings → `runs/*/rate_ceiling.json`.

# Implementation

## Phase 1

Phase Status: completed

Round: 1

### Goal

Joint limits in `RobotSpec`; latency and asynchronous switching in `label_replay`; `rate_ceiling` module and
CLI.

### Files

`src/vla_tools/robot_spec.py`, `src/vla_tools/arc_interp.py`, `src/vla_tools/label_replay.py`,
`src/vla_tools/rate_ceiling.py`, `tests/test_arc_interp.py`, `tests/test_rate_ceiling.py`,
`tests/test_label_replay.py`.

### Structures

`EdgeCheck`, `RateCheck`.

### Affected Modules

Robot limits, execution model, rate ceiling.

### Dependencies

None.

### Observation

Tests: latency 0 in both modes equals the current replay; asynchronous switching rebuilds each arriving chunk
from the commanded pose and holds d frames only at the start; a straight constant-speed recording gives velocity ratio r · v / limit;
`time` span 10 at rate 1 executes the recording; the arrival rebuild starts at the commanded pose. The LIBERO
replay JSONs unchanged under the defaults; LIBERO rate ceiling printed for `time` s10 and `curvature` s40 d2.

## Phase 2

Phase Status: completed

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

Result: LIBERO, every span-40 cell on the same 2917 inference frames (stride 2); `curvature` cells unchanged,
`time` s40 d2 path full median 0.092 → 0.096 path steps, d0 0.123 → 0.133; selection unchanged
(`curvature` s40 d2, 0.082), shortlist unchanged.

## Phase 3

Phase Status: completed

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

Result: lerobot/droid_1.0.1 revision 0eabc778, files 000, 078 and 155: 2335 episodes, 1951 successful, 891
pick-and-place instructions, 580 with one grasp and one release; 60 kept from 44 buildings and 38
collectors, 63–743 frames (median 216). The measured joints follow the commands about 4 frames later
(ISSUE-005), so the extracted state is the command echo and action[t] = state[t + 1] holds exactly; the
gripper command is thresholded at 0.5. Audit: 17 jumps above 8× the median step (10 discontinuities, 6
continuing, 1 isolated) in 10 episodes; the recordings reach median 0.54 of the velocity limit at their
fastest frame, 5 episodes exceed it (1.41 max, all at jumps) and are left out by `rate_ceiling`. Timing:
leading still frames trimmed (37 episodes, up to 41 frames), trailing (8, up to 41), pauses kept, 256
padding rows. Keyframes: grasp and release from the command edges, 60 / 60 each; 5 open phase / interval
outliers, left open (no camera review; edges are exact). Operators' speed against curvature: slope −0.39,
R² 0.26 (LIBERO −0.31, 0.45); window progress p90 / p10 at span 32 3.9× (LIBERO at span 40: 1.3×).

## Phase 4

Phase Status: completed

Round: 2

### Goal

The end rule and completion frames in `label_replay`; the path bound, completion speed-up, velocity-based
early stop, all-episode path step and the 90% summary in `rate_ceiling`.

### Files

`src/vla_tools/label_replay.py`, `src/vla_tools/rate_ceiling.py`, `tests/test_label_replay.py`,
`tests/test_rate_ceiling.py`, `tests/test_pipeline_chain.py`.

### Structures

`EpisodeReplay.completion_frames`, `RateCheck` (path bound, completion frames), `CaseResult` (episodes left
out, per-episode speed-up).

### Affected Modules

Execution model, rate ceiling.

### Dependencies

Phases 1, 2 and 3.

### Observation

Tests: per-frame `time` labels at rate 1, e = 1 and L = 0 execute every recorded action including the last
and complete in the recorded frames (speed-up 1); a sparse chunk at small e keeps observing the last frame
until it settles; a run beyond the path bound fails on "path"; the summary selects a rate feasible on 90%
but not all episodes. Printed: completion and executed frames per test run.

Result: 129 passed, 1 skipped. Per-frame labels at rate 1, e = 1: 80 executed targets equal the 80 recorded
actions, completion 80 = recorded completion. `time` span 40 at e = max(d, 1): the last frame is observed
5 times (d = 0, sync), 2 (d = 2, async: 84 executed, completion 83) and 2 (d = 2, sync). Corner, `time`
span 40, H = 2, e = 10: path max 3.23 path steps, fails on "path"; per-frame 0.00. Ten minimum-jerk
episodes, one 1.5× faster: every episode feasible up to rate 2 (speed-up 2.00), 90% up to rate 3 (2.97, the
fast one left out). Minimum-jerk recording of 200 frames completes in 190 (the last 10 within the final
tolerance); per-frame labels complete in 190, 95, 64 frames at rates 1, 2, 3.

## Phase 5

Phase Status: active

Round: 3

### Goal

DROID and LIBERO rate ceilings for every configuration, end-effector deviation at the selected settings,
velocity bounds; LIBERO replays re-run.

### Files

`FTpi05/reports/rate-ceiling-src/ee_deviation.py`,
`FTpi05/runs/droid_pick_place/{rate_ceiling,velocity_bound,ee_deviation}.json`,
`FTpi05/runs/libero_alphabet_soup/{rate_ceiling,velocity_bound,ee_deviation,replay_*}.json`.

### Structures

None.

### Affected Modules

None in the package.

### Dependencies

Phase 4.

### Observation

Per configuration and latency: the maximum asynchronous speed-up (and the synchronous one), its rate and
execution length, episodes feasible there, the constraints failing at the next rate, per-episode largest
rate and speed-up median / p10, the reported acceleration and switch step, path deviation (joint path
steps, end effector mm); against the per-frame baseline and the time-optimal bound. LIBERO replay rows
before / after the end rule.

## Phase 6

Phase Status: pending

Round: 4

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

Phase 5.

### Observation

The tables match the phase 2 and 5 artifacts; the rate is chosen by `rate_ceiling` in the S7 procedure.
