# Plan

Plan Status: completed

# Problem

## Current

The server places label point k at offset + k·span/H frames after the observation
(`arc_interp.point_frames`), so a chunk that covers about span recorded frames is executed over span
control frames: on average the robot moves at the recording's pace and never faster. For static tasks
timing along the path does not matter except stopping at keyframes, so the deployment pace is a free
choice, but no tool exposes it.

The progress step (`arc_core.progress_curve`) divides the total progress by every row of the episode,
end-padding rows included. Under `curvature` and `second_difference` those rows add no progress, so a
span of S reaches fewer recorded frames than S (ISSUE-002). LIBERO run, span 40, e = 10, 50
demonstrations: executed / recorded frames are 1.02× for `time` density 0, 1.29× for `curvature`
density 0 (1.03× with the padding rows removed), 1.49× for `curvature` density 2 (1.23× without padding
rows); density 2 slows `time` labels to 1.21× as well.

The deck's robot page (page 6) offers a time / curvature switch that duplicates the left view's
baseline, and cannot show a faster execution.

## Problem

1. Deployment pace is tied to the recording; there is no parameter to execute faster.
2. Geometric labels are stretched by end padding (ISSUE-002).
3. The robot page compares the wrong things.

## Goal

- An execution rate r at deployment: point k at (offset + k·span/H) / r frames. Labels and training are
  unchanged; the joint path executed is the same PCHIP curve, traversed r times faster.
- The progress step normalized over recorded rows only: with no padding effect, `curvature` density 0
  executes within a few percent of the recording at r = 1.
- Page 6: left, teleop playback at 1×; right, closed-loop execution of geometric (`curvature`) labels
  with span, execution rate, execution length and keyframe density adjustable; completion time of both,
  speed-up, largest joint step relative to the recorded p95 step, path deviation.

# Structure

## Modules

- Labels: `src/vla_tools/arc_core.py`.
- Deployment: `src/vla_tools/arc_interp.py` (server, standalone), `src/vla_openpi/launch.py`.
- Scoring: `src/vla_tools/label_replay.py`, `src/vla_tools/sweep.py`; `src/vla_tools/build_labels.py`
  calls the label builder.
- LIBERO run: `FTpi05/runs/libero_alphabet_soup/`, `FTpi05/data/libero-alphabet-soup-*`.
- Knowledge: skills, `docs/validation.md`, `issues.md`, `README.md`.
- Deck: `FTpi05/reports/action-sparsification-src/deck/`, `FTpi05/reports/action-sparsification.html`,
  `gh-pages`.

## Responsibilities

The rate is a deployment parameter owned by the server (`ArcInterpolator.rate`) and by the replay that
imitates it; label datasets and `arc_resample.json` do not record it. The step normalization belongs to
`arc_core` alone; callers pass the number of recorded rows.

## State Ownership

`arc_core.progress_curve` owns the step. `ArcInterpolator` owns the deployed rate. The LIBERO run's
artifacts are owned by the run directory and rewritten by S7.

# Interface

## Interfaces

- `arc_core.progress_curve(actions, spec, config, windows, recorded: int)` and
  `arc_core.chunks(actions, spec, config, windows, recorded: int)`: `recorded` is the number of recorded
  rows (padding rows follow them). The step is (sum of the free progress over the segments between
  recorded rows) × span/H ÷ (number of non-respaced segments between recorded rows). Padding segments
  keep their own progress (1 under `time`, 0 under geometric metrics).
- `arc_interp.point_frames(span_frames, action_horizon, anchor, rate: float)`: (offset + k·span/H) / rate.
  `ArcInterpolator.rate: float = 1.0`; `ArcInterpolator.from_report(path, rate: float = 1.0)`.
  `vla_openpi.launch.serving_config` and `serve` take `rate`; the example launcher has `--rate`.
- `label_replay.closed_loop(e, spec, record, config, execute, rate: float)`; `replay(..., rate)`;
  CLI `--rate` (default 1); the payload records the rate. The tie-break for re-observation moves to
  t + round(r · executed targets).
- `sweep` scores at r = 1 (path geometry does not depend on r; speed figures do, and the replay reports
  them at the deployed rate).

## Inputs

The LIBERO timed dataset (end padding 40), keyframes, label configurations.

## Outputs

Rebuilt `sweep.json`, `sweep_three_metrics.json`, label datasets, `replay_<labels>.json` (r = 1) and
`replay_<labels>_r2.json` (r = 2), data report; updated docs; rebuilt deck.

## State Changes

Label datasets built with a geometric metric before this change are not reproducible by the new code;
the only ones are the LIBERO `curvature-s40-d2` datasets, which S7 rebuilds. `time` labels are unchanged.

# Flow

## Main Flow

Build labels: progress curve over the whole episode, step from the recorded rows → chunks. Deploy:
`ArcInterpolator(rate)` places the H points at (offset + k·span/H)/r frames and interpolates. Replay:
the closed loop executes e targets at rate r and re-observes. Deck: the JavaScript port does the same
for one LIBERO demonstration and animates teleop playback beside execution.

# Code Mapping

## Modules

`src/vla_tools/arc_core.py`, `arc_interp.py`, `label_replay.py`, `sweep.py`, `build_labels.py`,
`src/vla_openpi/launch.py`; `tests/test_arc_core.py`, the interpolator tests, `tests/test_pipeline_chain.py`;
`.claude/skills/vla-action-labels/`, `.claude/skills/vla-downstream/SKILL.md`, `docs/validation.md`,
`issues.md`, `README.md`; deck files under `FTpi05/reports/`.

## Interfaces

`progress_curve`, `chunks` → `arc_core.py`; `point_frames`, `ArcInterpolator` → `arc_interp.py`;
`closed_loop`, `replay` → `label_replay.py`; `--rate` → `launch.py`, `label_replay.py`; JavaScript port →
`FTpi05/reports/action-sparsification-src/deck/core.js`.

## State

Step: `arc_core.py`. Deployed rate: `arc_interp.py`. LIBERO artifacts: `FTpi05/runs/libero_alphabet_soup/`.

# Implementation

## Phase 1

Phase Status: completed

Round: 1

### Goal

Normalize the progress step over recorded rows (ISSUE-002).

### Files

`src/vla_tools/arc_core.py`, `src/vla_tools/sweep.py`, `src/vla_tools/label_replay.py`,
`src/vla_tools/build_labels.py`, `tests/test_arc_core.py`.

### Structures

`recorded` argument.

### Affected Modules

Labels, scoring.

### Dependencies

None.

### Observation

The same episode with 10 and 40 padding rows gives identical progress over recorded rows and identical
labels on frames whose label has no padded point; `time` labels unchanged (existing tests); median reach
of `curvature` labels on a synthetic episode printed against span.

## Phase 2

Phase Status: completed

Round: 2

### Goal

Execution rate r in the server, the launcher and the closed-loop replay.

### Files

`src/vla_tools/arc_interp.py`, `src/vla_openpi/launch.py`, `src/vla_tools/label_replay.py`,
`src/vla_tools/sweep.py`, tests of the interpolator and the pipeline chain.

### Structures

`rate` field and arguments.

### Affected Modules

Deployment, scoring.

### Dependencies

Phase 1 (`label_replay.py`, `sweep.py`).

### Observation

Targets at rate 2 equal the rate-1 targets at every second frame (≤ 1e-12); `time`, span = H replay at
rate 2 observes every 2e frames with path deviation 0; executed frames printed for rates 1 and 2.

## Phase 3

Phase Status: completed

Round: 3

### Goal

Re-run S7 on LIBERO: sweeps, label datasets, replays at r = 1 and r = 2, data report.

### Files

`FTpi05/runs/libero_alphabet_soup/**`, `FTpi05/data/libero-alphabet-soup-*-train|eval`.

### Structures

None.

### Affected Modules

LIBERO run.

### Dependencies

Phases 1, 2.

### Observation

`run_status` all passed; executed / recorded frames of the selected labels at r = 1 and r = 2 printed;
`curvature` reach close to span.

## Phase 4

Phase Status: completed

Round: 4

### Goal

Docs: rate and step normalization in the action-labels and downstream skills, method reference,
validation numbers, README; ISSUE-002 resolved.

### Files

`.claude/skills/vla-action-labels/SKILL.md`, `.claude/skills/vla-action-labels/references/method.md`,
`.claude/skills/vla-downstream/SKILL.md`, `.claude/skills/vla-pipeline/references/failure-modes.md`,
`docs/validation.md`, `issues.md`, `README.md`.

### Structures

None.

### Affected Modules

Knowledge.

### Dependencies

Phase 3.

### Observation

LIBERO numbers match the rebuilt artifacts; no text describes padding-stretched spans as current.

## Phase 5

Phase Status: completed

Round: 4

### Goal

Deck: port the step normalization and the rate; page 6 rebuilt as approved (no metric switch; span,
execution rate, execution length, keyframe slow-down; completion times and speed-up); other pages' LIBERO
numbers from the rebuilt run; publish to `gh-pages`.

### Files

`FTpi05/reports/action-sparsification-src/deck/**`, `FTpi05/reports/action-sparsification.html`,
`gh-pages:index.html`.

### Structures

None.

### Affected Modules

Deck.

### Dependencies

Phases 1, 2, 3.

### Observation

`check_core.py` ≤ 1e-9 including rates 1 and 2; smoke tests pass; 3-D capture in view; served page
identical to the local build.
