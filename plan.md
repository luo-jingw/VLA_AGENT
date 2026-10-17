# Plan

Plan Status: completed

# Problem

## Current

`rate_ceiling` executes every chunk at one uniform execution rate r: a chunk that covers `span_frames`
recorded frames takes span / r control frames, whatever its geometry. A run is feasible when the executed
targets stay within the Panda's joint velocity limits, every gripper edge is executed and the closed-loop
path deviation stays within 2 path steps; the result is the largest completion speed-up feasible on 90% of
the episodes.

- Uniform timing, asynchronous, L = 0 / 0.05 / 0.1 / 0.2 / 0.4 s: DROID (55 episodes) best 1.96 / 1.96 /
  1.93 / 1.45 / 0.97 (`time` s32 d0), LIBERO (50) 2.44 / 2.87 / 2.65 / 2.29 / 2.03 (`curvature` s64–s128).
  The next rate fails on velocity. The recorded paths allow 7.65× (DROID) and 5.95× (LIBERO) with every
  recorded step at the speed of its limiting joint; the recording replayed at one rate is feasible on every
  episode up to 1.07× and 1.96×.
- At one rate the fastest stretch of a recording sets the rate for all of it: straights, slow segments and
  pauses run at the pace that keeps the fastest stretch within the limit.
- Prototype (2026-10-17): each chunk's spline (the uniform knots at rate r) traversed at the joint velocity
  limit, never faster than rate r; L = 0, asynchronous, path bound 2, 90%. DROID: per-frame 5.47 (r 8,
  e 1), `time` s32 d0 7.10 (r 16, e 1, 50 of 55), `curvature` s32 d0 7.09 (r 16, e 2, 54), `curvature` s64 d0
  7.24 (r 12, e 5, 50); `time` s64 none (path bound). Integrating the timing with the trapezoid rule
  overshot the limit by 0.3%; the larger endpoint per integration interval keeps the targets within it.
- Gripper: frames the arm stays within 1 path step of the pose at a command switch, median / p10 over 110
  DROID switches: recording 7 / 3, uniform `time` s32 r 2 e 16 4 / 1, velocity timing at r 16 1 / 1. Under
  uniform timing the keyframe density holds the arm near a switch for a share of span / r frames; velocity
  timing traverses those nearly still stretches at the cap. DROID's measured gripper position reaches 90%
  of its travel a median 8 frames (0.53 s) after a close command (p10 / p90 4 / 10) and 10 frames
  (0.67 s) after an open command (7 / 11); file 000, 1374 and 1316 edges.

## Problem

Uniform timing cannot use the velocity margin of slow segments, so the rate ceiling stays near the
recording's fastest stretch. Timing each chunk by its geometry removes that limit but also removes the time
the labels' keyframe density gives the gripper at a switch.

## Goal

- Velocity timing on the server: the chunk's joint spline, unchanged (not-a-knot over the knots
  `(offset + k · span / H) / r` from the pose it starts at), is traversed with chunk time T mapped to control
  frames s(T) = ∫ max(1, fps · max_j |dq_j/dT| / (α · v_j)) dT, the integrand taken as the larger endpoint
  value on intervals of 1/16 frame; one target at each s = 1, 2, …, ⌊s(T_end)⌋; commands from the nearest
  point in T. A chunk is never faster than rate r (the cap) and never moves a joint faster than α of its
  limit along the spline. Without velocity timing the server is unchanged.
- Switch hold: a fixed still segment at each command switch. At the first target whose command differs from
  the previous target (or from the pose the chunk starts at), the arm stays for round(h · fps) further frames at
  that target before the chunk continues, h = h_rise for a command that rises above 0.5 and h_fall for one
  that falls below it (close and open on DROID and LIBERO); seconds, a deployment setting like r. The hold
  replaces the time that uniform timing gives the nearly still stretch around a switch through the labels'
  keyframe density. A chunk that replaces one before its hold is over starts with the rest of the hold, so
  replanning never shortens it; an arriving chunk's T_d counts only the latency frames the robot was not
  holding.
- Arrival rebuild with velocity timing: a chunk observed at commanded pose p_obs and arriving d frames later
  keeps the points whose T exceeds T_d by more than half a point spacing, T_d the chunk time of target d of
  the chunk timed from p_obs (holds included); the knots shift by T_d and the chunk is rebuilt from the pose
  commanded at arrival and timed the same way. With uniform timing T_d = d, as now.
- Rate ceiling with velocity timing: α = 1, holds (h_rise, h_fall) ∈ {(0, 0), (0.53 s, 0.67 s)} (none;
  DROID's median close and open times: 8 and 10 frames at 15 Hz, 11 and 13 at LIBERO's 20 Hz), the same
  configurations (per-frame; `time` and `curvature` at DROID spans 32, 64, 128, 256 and LIBERO spans 32, 40,
  64, 128, 256; density 0 and 2), latencies, modes, rate grid (now the cap), execution lengths, path bound
  and 90% summary. Feasibility unchanged: velocity is still checked over all executed targets, switches and
  holds included. Reported, not constrained: the executed frames held at each command switch against the
  recording's (`EdgeCheck.holds`, 0.25 path steps), median / min.
- Per data set: uniform timing against velocity timing without and with the holds per latency and mode,
  per-episode speed-up, end-effector deviation, against the time-optimal bound.

# Structure

## Modules

- Server rebuild: `src/vla_tools/arc_interp.py` (velocity timing, switch hold, arrival rebuild; standalone,
  numpy + scipy only).
- Execution model: `src/vla_tools/label_replay.py` (`closed_loop` passes the server timing; elapsed chunk
  time at arrival).
- Rate ceiling: `src/vla_tools/rate_ceiling.py` (velocity-timed cases, switch-hold report).
- Reference figures (outside the package): `FTpi05/reports/rate-ceiling-src/gripper_actuation.py` (new:
  DROID gripper close / open time from the measured position), `ee_deviation.py` (server timing from the
  result).
- Knowledge: `docs/validation.md`, `.claude/skills/vla-action-labels/` (SKILL.md, references/method.md),
  `README.md`.

## Responsibilities

`arc_interp` owns every mapping from points to per-frame targets: uniform timing at rate r or velocity
timing capped at r, the switch hold and the arrival rebuild. `label_replay` owns the execution model and
hands the server timing to `arc_interp`; it tracks the commanded pose at each observation for the arrival
rebuild. `rate_ceiling` builds the server timing from the robot's velocity limits, α and h, searches the cap
and execution lengths and reports holds. Deployment wiring (`vla_openpi.launch`) is not changed.

## State Ownership

- Server timing (α, h, velocity limits): a deployment setting held by `ArcInterpolator.timing`; in the
  rate ceiling, built per case from `robot_spec.json` and the CLI; recorded in `rate_ceiling_velocity.json`.
- Rate ceilings: each run's `rate_ceiling.json` (uniform) and `rate_ceiling_velocity.json` (velocity
  timing), written by `rate_ceiling`.

# Interface

## Interfaces

- `arc_interp.VelocityTiming(velocity_limits: tuple[float, ...], fraction: float, fps: float,
  rise_hold_s: float, fall_hold_s: float)`: frozen dataclass; one limit per non-command dim in dim order
  (rad/s).
- `arc_interp.to_frames(points, current, times, command_dims, timing: VelocityTiming | None = None) ->
  np.ndarray`: (n, D) targets; n = ⌊times[-1]⌋ without timing, at least that with it (plus holds).
- `VelocityTiming.owed_hold(history, command_dims) -> int`: hold frames still owed at the last of the
  commanded poses `history` (oldest first).
- `arc_interp.target_times(points, current, times, command_dims, timing: VelocityTiming | None = None) ->
  np.ndarray`: (n,) chunk time T of each target of `to_frames` (holds repeat their target's T).
- `arc_interp.arrival_time(points, observed, times, command_dims, timing, elapsed_frames) -> float`: T_d,
  the chunk time of target `elapsed_frames` of the chunk timed from `observed` (the elapsed frames with
  uniform timing); used by `ArcInterpolator.after_latency` and `closed_loop`.
- `arc_interp.after_latency(points, times, commanded, elapsed: float, command_dims, timing:
  VelocityTiming | None = None) -> np.ndarray`: `elapsed` becomes the chunk time at arrival (T_d; equal to
  the elapsed frames with uniform timing).
- `ArcInterpolator.timing: VelocityTiming | None = None`; `from_report(path, rate=1.0, timing=None)`;
  `after_latency(points, observed, commanded, elapsed_frames)`: `observed` is the pose commanded at the
  observation, used for T_d under velocity timing.
- `label_replay.closed_loop(..., *, final_tolerance: float, timing: VelocityTiming | None = None)`.
- `rate_ceiling`: `Case.timing: VelocityTiming | None`; `check(..., timing)`; `RateCheck.holds`,
  `recorded_holds: tuple[float, float]` (median, min frames); `CaseResult.velocity_fraction`,
  `switch_hold_s: tuple[float, float]` (rise, fall), `holds`, `recorded_holds`; CLI `--velocity-fraction α`
  (velocity timing for every case) and `--switch-hold <rise s> <fall s>`, repeatable, one case set each.

## Inputs

DROID and LIBERO timed datasets, keyframes and robot specs as in the uniform runs; DROID raw file 000 for
the gripper actuation time.

## Outputs

`runs/droid_pick_place/` and `runs/libero_alphabet_soup/`: `rate_ceiling_velocity.json`,
`ee_deviation_velocity.json`; `runs/droid_pick_place/gripper_actuation.json`.

## State Changes

None beyond the outputs.

# Flow

## Main Flow

`gripper_actuation.py` → (h_rise, h_fall). `rate_ceiling --velocity-fraction 1 --switch-hold 0 0
--switch-hold 0.53 0.67` per data set: per case, `closed_loop(timing=VelocityTiming(limits, 1, fps, h_rise,
h_fall))` → per chunk `to_frames` (first chunk, sync
chunks) or `target_times` on the observation pose then `after_latency` (async arrivals) → checks → summary.
`ee_deviation.py` on the velocity results → comparison table with the uniform results.

# Code Mapping

## Modules

Server rebuild → `src/vla_tools/arc_interp.py`; execution model → `src/vla_tools/label_replay.py`; rate
ceiling → `src/vla_tools/rate_ceiling.py`; reference figures → `FTpi05/reports/rate-ceiling-src/`.

## Interfaces

`VelocityTiming`, `to_frames`, `target_times`, `arrival_time`, `after_latency`, `ArcInterpolator` →
`arc_interp.py`;
`closed_loop` → `label_replay.py`; `Case`, `check`, `RateCheck`, `CaseResult`, CLI → `rate_ceiling.py`.

## State

Server timing → `ArcInterpolator.timing` (`arc_interp.py`), `Case.timing` (`rate_ceiling.py`); results →
`runs/*/rate_ceiling_velocity.json`.

# Implementation

## Phase 1

Phase Status: completed

Round: 1

### Goal

Velocity timing, switch hold and the velocity-timed arrival rebuild in `arc_interp`.

### Files

`src/vla_tools/arc_interp.py`, `tests/test_arc_interp.py`.

### Structures

`VelocityTiming`; `target_times`.

### Affected Modules

Server rebuild.

### Dependencies

None.

### Observation

Tests, measured values printed: without timing the targets are unchanged; a straight constant-speed chunk
whose rate-r speed exceeds α of a limit runs at α of it (frame count against the closed form); a still
chunk takes ⌊times[-1]⌋ frames (the cap); the largest per-frame joint speed over random splines as a
fraction of α · v (≤ 1); a rising and a falling command switch hold round(h_rise · fps) and round(h_fall · fps)
further frames; `target_times` matches
`to_frames`; the velocity-timed arrival rebuild starts at the commanded pose and keeps the points past T_d.

Result: 135 passed, 1 skipped. Without a binding limit the velocity-timed targets equal the uniform ones
(difference 0, 10 targets at rate 2); a 2 rad straight in 10 frames at 80% of 0.5 rad/s takes 50 frames of
0.040000 rad; a still chunk takes ⌊times[-1]⌋ = 8 frames at rate 4; 60 random chunks at rates 1, 4, 16:
largest joint speed 0.9999 of 90% of the limit, 12–36 targets per span-32 chunk (sampling the integrand per
1/16 chunk-time unit instead reached 1.026 at rate 16, where one unit spans many control frames); a rise and
a fall hold 5 and 2 further frames (0.5 s, 0.2 s at 10 fps), 20 → 27 targets, a switch against the start
pose holds too; arriving 3 frames late at rate 4 the chunk time passed is 0.302 of 8, 27 of 30 targets
remain and the first step from the commanded pose is 0.975 of the limit.

## Phase 2

Phase Status: completed

Round: 2

### Goal

`closed_loop` and `rate_ceiling` with the server timing; hold report.

### Files

`src/vla_tools/label_replay.py`, `src/vla_tools/rate_ceiling.py`, `tests/test_label_replay.py`,
`tests/test_rate_ceiling.py`.

### Structures

`Case.timing`; `RateCheck` and `CaseResult` hold fields.

### Affected Modules

Execution model, rate ceiling.

### Dependencies

Phase 1.

### Observation

Tests: timing None reproduces the uniform results; on the minimum-jerk profile velocity timing at α = 1
keeps the velocity ratio ≤ 1 at every cap and completes faster than the uniform ceiling (printed); with
latency the arrival rebuild under velocity timing switches without a jump; the switch hold appears in
`holds`.

Result: 138 passed, 1 skipped. Minimum-jerk profile under a limit of 3.5× its largest step: uniform timing
reaches rate 3 (speed-up 3.00); velocity timing keeps every cap within the limit (largest ratio 1.0000)
and completes 5.28× faster at cap 8, e 1. Planar arc, span 40, cap 4, limit 0.5 rad/s, 2 frames of latency,
holds 0.5 / 0.2 s at 10 fps: velocity ratio 0.9999 at e = 2, 4, 8; the arm stays 6 and 3 frames at the close
and the open. Before the hold was carried into the replacing chunk, e = 2 held 2 and 1 frames: the next
chunk, built from a pose that already had the new command, saw no switch.

## Phase 3

Phase Status: completed

Round: 3

### Goal

Gripper actuation time from DROID; DROID and LIBERO velocity-timed rate ceilings; end-effector deviation;
comparison with uniform timing.

### Files

`FTpi05/reports/rate-ceiling-src/gripper_actuation.py`, `FTpi05/reports/rate-ceiling-src/ee_deviation.py`,
`src/vla_tools/rate_ceiling.py` (all cases' episode jobs submitted to the pool at once; results unchanged),
`FTpi05/runs/droid_pick_place/{gripper_actuation,rate_ceiling_velocity,ee_deviation_velocity}.json`,
`FTpi05/runs/libero_alphabet_soup/{rate_ceiling_velocity,ee_deviation_velocity}.json`.

### Structures

None.

### Affected Modules

None in the package.

### Dependencies

Phase 2.

### Observation

Per data set, latency and mode: speed-up under uniform timing and under velocity timing without and with
the holds, cap and e, episodes feasible, constraints failing at the next cap, per-episode speed-up median /
p10, holds at switches against the recording's, end-effector deviation; against the time-optimal bound.

Result (2026-10-17; path bound 2 path steps, 90% of the episodes, α = 1; holds 0.533 / 0.667 s = 8 / 10
frames on DROID, 11 / 13 on LIBERO). DROID's measured gripper (`gripper_actuation.py`, file 000): close
median 8 frames (p10 / p90 4 / 10, 1374 edges), open 10 (7 / 11, 1316). Largest speed-up feasible on ≥ 90%,
asynchronous / synchronous:

| L (s) | DROID uniform | DROID velocity | DROID velocity + holds | LIBERO uniform | LIBERO velocity | LIBERO velocity + holds |
|---|---|---|---|---|---|---|
| 0 | 1.96 | 7.46 | 4.58 | 2.44 | 5.83 | 2.94 |
| 0.05 | 1.96 / 1.84 | 7.20 / 5.89 | 4.53 / 3.52 | 2.87 / 2.32 | 5.60 / 5.11 | 2.87 / 2.71 |
| 0.1 | 1.93 / 1.73 | 5.77 / 4.89 | 3.91 / 3.49 | 2.65 / 2.21 | 5.38 / 4.71 | 2.83 / 2.58 |
| 0.2 | 1.45 / 1.64 | 5.55 / 4.41 | 3.27 / 2.84 | 2.29 / 2.09 | 4.97 / 4.18 | 2.69 / 2.39 |
| 0.4 | 0.97 / 1.40 | 1.39 / 3.41 | 1.54 / 2.36 | 2.03 / 1.88 | 3.95 / 3.41 | 2.28 / 2.19 |

- Bounds: every recorded step at the speed of its limiting joint 7.65× (DROID), 5.95× (LIBERO); with the
  same holds at each recorded edge 4.68× and 2.96×. At L = 0 velocity timing reaches 98% of them with and
  without holds.
- Best configurations under velocity timing: `curvature` s32–s64 and `time` s32 on DROID, `curvature`
  s64–s128 on LIBERO, at caps 6–16; the next cap fails on the path bound or keeps no target after the
  latency (cap ≤ span / (d + 1)). Velocity is never the binding constraint (largest ratio 1.00).
- Per-frame labels under velocity timing: DROID 5.73 at L = 0, 0.82 at 0.4 s (asynchronous); their 10-frame
  chunks cap the rate at 10 / (d + 1) once there is latency. At 0.4 s the best sparse labels (spans
  32–128) keep 3.41× without holds and 2.19–2.36× with them (synchronous; LIBERO asynchronous 3.95 and 2.28).
- Holds at the switches: the arm stays at least 9 frames (DROID) and 12 (LIBERO) at every switch of every
  feasible run (1 + 8 at a close, 1 + 11 on LIBERO); the recordings stay a median 2 and 1–1.5 frames within
  0.25 path steps of the switch pose. Without holds the arm stays 1 frame.
- Asynchronous execution at 0.4 s collapses on DROID (1.39 / 1.54, path bound on 9–21 episodes at the next
  cap); synchronous keeps 3.41 / 2.36. Uniform timing shows the same failure (0.97).
- End-effector deviation at the selected settings: median 0.2–2.2 mm, p90 0.9–6.3 mm, max 5.4–29.1 mm.
- Replay fixes found on the data, committed in `label_replay` and re-run: the re-observation advanced
  during switch holds and counted the held pose as path deviation (holds: 3.74 → 4.58 at L = 0 on DROID);
  a run whose last switch fell on the final recorded rows stopped before its hold ended (episode 33: 3 of
  11 frames), and completion now lasts until the last hold is over, for the recording too.

## Phase 4

Phase Status: completed

Round: 4

### Goal

Documentation of velocity timing, the switch hold and the results.

### Files

`docs/validation.md`, `.claude/skills/vla-action-labels/SKILL.md`,
`.claude/skills/vla-action-labels/references/method.md`, `README.md`.

### Structures

None.

### Affected Modules

Knowledge.

### Dependencies

Phase 3.

### Observation

Tables match the phase 3 artifacts; the S7 procedure states when velocity timing and the switch hold apply
and how they are set at deployment.

Result: `docs/validation.md` gains test rows for velocity timing and a velocity-timing section (gripper
actuation, results table, bounds, holds, latency); `method.md` defines velocity timing, the switch hold and
its carry-over; the vla-action-labels skill states how to run and deploy it and records the DROID and LIBERO
figures; README lists the command; OPT-008 records the serving wiring. 138 passed, 1 skipped.
