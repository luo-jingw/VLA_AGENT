# Plan

Plan Status: completed

# Problem

## Current

- Velocity timing with switch holds (`arc_interp.VelocityTiming`: the chunk's spline re-timed so every joint
  stays within a fraction of its velocity limit, the rate as a cap, the arm held still for the gripper's
  actuation time at each command switch) is the execution that gave resampled labels their measured advantage
  (docs/validation.md → LIBERO simulator: S3b at cap 4, 77–81% in 65–103 frames against the recordings' 152).
- Deployment defaults to uniform timing: `launch.serving_config(config, checkpoint, rate=1.0)`,
  `launch.serve(..., rate=1.0)`, the examples' `serve --rate 1`, the vla-downstream skill ("执行倍速 r 是部署参数
  （默认 1）"). Velocity timing is reachable only through `vla_libero.evaluate` / `label_replay` flags.
- The joint velocity limits are in `RobotSpec` and travel in `arc_resample.json`'s robot block; the gripper's
  actuation times (the hold durations) are measured by a script outside the repository
  (`reports/rate-ceiling-src/gripper_actuation.py`) and passed by hand.

## Problem

A deployment built from the harness executes resampled points with uniform timing unless the person knows to
switch, and the hold durations have no home in the robot description.

## Goal

Velocity timing with switch holds is the default execution for deployment: the hold durations are part of the
robot spec and of every label report; one constructor builds the deployment interpolator from a report (cap,
velocity fraction); the serving paths and examples default to it; uniform timing remains available only when
requested explicitly; the skills describe it as the default.

# Structure

## Modules

- Robot description: `src/vla_tools/robot_spec.py`.
- Interpolation: `src/vla_tools/arc_interp.py` (single file delivered to serving machines).
- Serving: `src/vla_openpi/launch.py`; examples' `run.py` (`ur3_dual_arm`, `ur3_optical_bench`, `libero_object`).
- Evaluation client: `src/vla_libero/evaluate.py`.
- Skills: `.claude/skills/vla-downstream/SKILL.md`, `.claude/skills/vla-action-labels/SKILL.md`.

## Responsibilities

The robot spec owns the hold durations; `arc_interp` builds the deployment timing from a report; `launch` and
the examples choose velocity timing unless told otherwise.

## State Ownership

Hold durations: `RobotSpec.switch_hold_s` (rise, fall, seconds), copied into `arc_resample.json`'s robot block by
`build_labels` as the other robot fields are.

# Interface

## Interfaces

- `RobotSpec.switch_hold_s: tuple[float, float] = ()`: seconds the arm stays still after a command rises /
  falls; empty when not measured.
- `VelocityTiming.from_robot(robot: dict, fraction: float) -> VelocityTiming`: limits, fps and holds from a
  report's robot block; raises with the missing field named when limits or holds are absent.
- `ArcInterpolator.for_deployment(path: Path, cap: float = 4.0, fraction: float = 1.0) -> ArcInterpolator`:
  `from_report(path, cap, VelocityTiming.from_robot(report["robot"], fraction))`.
- `launch.serving_config(config, checkpoint, cap=4.0, velocity_fraction=1.0)` and `launch.serve(...)` with the
  same parameters; `velocity_fraction=None` gives uniform timing at rate `cap`.
- Examples' `run.py serve`: `--cap 4 --velocity-fraction 1` by default, `--uniform` for uniform timing.
- `vla_libero.evaluate`: velocity timing with the report's holds when `--velocity-fraction` is given and
  `--switch-hold` is not.

## Inputs

Robot specs of the runs (`runs/*/robot_spec.json`, outside the repository) gain `switch_hold_s` where measured
(LIBERO 0.2 / 0.55 s; UR3 optical bench 0.533 / 0.667 s).

## Outputs

Code, tests, skills, docs; the commits pushed to `luo-jingw/VLA_AGENT`.

Alongside, outside the repository: a simulator grid of the labels open loop over the sampling and execution
parameters (span 16, 24, 32, 48 x event density 0, 1, 2 x velocity fraction 0.6, 0.8, 1.0 x cap 4, 8; e 4, holds
0.2 / 0.55 s, sync, L = 0), `runs/libero_alphabet_soup/sim/label_grid/`.

## State Changes

Label reports written after the change carry `switch_hold_s`; earlier reports lack it and need the holds passed
explicitly (`from_robot` names the missing field).

# Flow

## Main Flow

Serving: `launch.serve(config, checkpoint, port)` → `serving_config` → `ArcInterpolator.for_deployment(report)` →
`ArcOutputs` per-frame targets re-timed within each chunk with holds. Robot-side clients that keep the commanded
history use `for_deployment(report).chunk_targets(...)`, which also carries a hold across chunk boundaries.

# Code Mapping

## Modules

As in Structure.

## Interfaces

`switch_hold_s` → `robot_spec.py`; `from_robot`, `for_deployment` → `arc_interp.py`; serving defaults →
`launch.py`, `examples/*/run.py`; evaluation default holds → `evaluate.py`.

## State

`RobotSpec.switch_hold_s`; report robot block.

# Implementation

## Phase 1

Phase Status: completed

Round: 1

### Goal

Hold durations in the robot spec and report; deployment constructor.

### Files

`src/vla_tools/robot_spec.py`, `src/vla_tools/arc_interp.py`, `tests/test_arc_interp.py`, tests of the robot spec.

### Structures

`RobotSpec.switch_hold_s`, `VelocityTiming.from_robot`, `ArcInterpolator.for_deployment`.

### Affected Modules

Robot description, interpolation.

### Dependencies

None.

### Observation

Tests: a report with limits and holds gives the same targets as the explicit `VelocityTiming`; a report without
holds raises naming `switch_hold_s`; the LIBERO S3b report with holds added reproduces a recorded trace's first
chunk.

Result: `RobotSpec.switch_hold_s`, `VelocityTiming.from_robot`, `ArcInterpolator.for_deployment` (`DEPLOY_CAP` 4,
`DEPLOY_FRACTION` 1.0); three tests pass (peak joint speed 1.000 of the limit, 5 still rows after a 0.4 s rise
at 10 Hz). The LIBERO S3b report with the holds added gives the interpolator of the evaluated velocity-timing
runs. Run robot specs: LIBERO 0.2 / 0.55 s, UR3 optical bench 0.533 / 0.667 s.

## Phase 2

Phase Status: completed

Round: 2

### Goal

Serving and evaluation defaults; skills and docs.

### Files

`src/vla_openpi/launch.py`, `src/vla_libero/evaluate.py`, `examples/*/run.py`, the two skills,
`docs/validation.md`, `README.md` where it describes serving.

### Structures

None new.

### Affected Modules

Serving, evaluation client, skills.

### Dependencies

Phase 1.

### Observation

`serving_config` on the S3b checkpoint returns targets equal to `for_deployment(report)(points, state)`; full
test suite; ruff.

Result: `launch.serving_config` / `serve` default to `cap=4, velocity_fraction=1.0` (`None`: uniform timing);
UR3 examples' `serve --cap --velocity-fraction --uniform`; `vla_libero.evaluate` takes the spec's holds when
`--switch-hold` is absent; skills, README and docs updated. On the S3b checkpoint (report with holds, the dataset
and checkpoint copies identical) `serving_config` gives `for_deployment`'s interpolator and equal targets. Tests
150 passed, 2 skipped; openpi integration 8 passed; ruff clean.

## Phase 3

Phase Status: completed

Round: 3

### Goal

Push.

### Files

None.

### Structures

None.

### Affected Modules

None.

### Dependencies

Phases 1 and 2.

### Observation

`git push` to `luo-jingw/VLA_AGENT` with the project token through a temporary GIT_ASKPASS; remote main equals
local main.

Result: pushed to `luo-jingw/VLA_AGENT`, remote main = local main = 3216a75 (26 commits).
