# Plan

Plan Status: approved

# Problem

## Current

A teleoperated recording samples the joint path uniformly in time, so the recorded frames sit unevenly
along the path. `arc_core` builds every observation frame's label from three things that are mixed
together:

- the continuous curve: PCHIP over the row index, i.e. parameterized by recorded time;
- the curvature: chord angles between the raw recorded samples, with chords of `curvature_window` × the
  episode's median per-frame step, a length that scales with the teleop speed (ISSUE-001: re-timing a
  LIBERO demonstration to 2× moves the normalized progress by 1.6% median, 0.9% with an absolute window);
- the point spacing: progress s accumulated as κ^(1/3) · chord length, and one step for the whole episode,
  `step = (free progress over the recorded rows) × (span/H) ÷ (free segments)`; each frame starts at its own
  s(t−1) and walks H steps of that shared size.

Because the step is episode-wide, how far a frame's label reaches depends on the curvature of the entire
episode, not on the stretch ahead of the frame. LIBERO, `curvature`, span 40, H 10, density 0, 5708 frames
whose window fits in the recording: the 10th point reaches p05 30.1 / p50 37.9 / p95 45.3 recorded frames;
a step computed from each frame's own 40 frames would be p05 0.86 / p50 1.06 / p95 1.28 times the episode
step and would move points by p50 1.5 / p95 6.6 / max 28 frames. The same local motion gets different
spacing in different episodes, and the end-padding dilution of ISSUE-002 came from the same average.

The per-window rule exists in the G2 Orion 0.5 resampling package only as an experiment
(`sliding_window.py`: 1.67 mm error / 1.78 mm replan jitter against 8.39 / 9.52 for a fixed global grid at
span 100, H 50); its dataset builder uses the episode-wide step, as `arc_core` does.

## Problem

1. The curve representation is not chosen or validated on its own: its parameterization is recorded time,
   and the curvature is estimated on raw samples with a speed-dependent chord length.
2. Each frame's label must be the sparsification of that frame's own window of the curve, not a walk with
   an episode-wide step.

## Goal

Three separated stages: reconstruct the whole trajectory as a curve → cut each frame's window → sparsify
the window by curvature.

- Reconstruction: a parameterization and interpolant chosen by measurement on LIBERO and a constructed
  path with known geometry, independent of curvature; the curvature estimator chosen by its dependence on
  the teleop speed.
- Window: frame t's window starts at its state and covers the next span recorded frames.
- Sparsify: a frame whose window holds no keyframe segment and lies within the recorded rows has its H-th
  point at exactly the window end; inside keyframe segments the density is `event_density` points per
  span/H recorded frames whatever the rest of the window holds, and free segments of the window keep one
  point per span/H frames on average, so a window holding keyframe frames covers fewer than span frames
  when `event_density` > 1; points past the last recorded row stay on it and are flagged as padding.
- `time` labels with span = H unchanged (they sample recorded rows only).
- LIBERO comparison before / after: reach, replan jitter (strides 1 and 10), path deviation, keyframe
  deviation, executed / recorded frames at rates 1 and 2.

# Structure

## Modules

- Curve: `src/vla_tools/trajectory_curve.py` (new): the reconstructed joint curve of one episode.
- Labels: `src/vla_tools/arc_core.py` (window and sparsification), `src/vla_tools/resample_config.py`.
- Measurement: `FTpi05/reports/action-sparsification-src/libero/curve_check.py` (new, the reconstruction
  experiment) and `window_check.py` (new, label statistics of the current `arc_core`).
- LIBERO run: `FTpi05/runs/libero_alphabet_soup/`, `FTpi05/data/libero-alphabet-soup-*-train|eval`.
- Knowledge: `.claude/skills/vla-action-labels/`, `.claude/skills/vla-pipeline/references/failure-modes.md`,
  `docs/validation.md`, `issues.md`, `opportunities.md`, `README.md`.
- Deck: `FTpi05/reports/action-sparsification-src/deck/`, `FTpi05/reports/action-sparsification.html`,
  `gh-pages`.

Deployment (`arc_interp`, `vla_openpi.launch`), `label_replay` and `build_labels` call `arc_core.chunks`
and are unchanged.

## Responsibilities

`trajectory_curve` owns the reconstruction: the joint curve through the recorded rows and its values
between them. `arc_core` owns the metric (curvature of the recorded rows), the per-frame window, the step
and the points, whose joints it reads from the curve.
`curve_check.py` compares candidate reconstructions; only the selected one is implemented in
`trajectory_curve`.

## State Ownership

The curve is built per episode inside `arc_core.chunks` and not stored. The per-frame step is returned in
`Chunks.step`. LIBERO run artifacts are owned by the run directory and rewritten by S7.

# Interface

## Interfaces

Segment i joins action rows i and i+1; `recorded` rows R are followed by end-padding rows.

- `trajectory_curve.TrajectoryCurve` (frozen dataclass): the not-a-knot cubic spline over the row index
  through every recorded row, constant at the last recorded row beyond it, so end-padding rows do not
  shape it. `joints(frames: np.ndarray) -> np.ndarray` evaluates it at fractional row indices.
  `trajectory_curve.build(joints: np.ndarray, recorded: int) -> TrajectoryCurve`.
- `arc_core.progress_curve(actions, spec, config, windows, recorded) -> ProgressCurve`: `free` (T−1,) the
  metric progress of each segment (`curvature`: max(c, floor · c_mean) · segment length with
  c = ((κ[i] + κ[i+1]) / 2)^(1/3), κ the chord-angle curvature of the recorded rows with chords of
  `curvature_length`; padding segments 0), 0 on respaced keyframe segments; `respaced` (T−1,); `window_mask` (T−1,); `recorded: int`.
- `arc_core.chunks(actions, spec, config, windows, recorded) -> tuple[ProgressCurve, Chunks]`. For frame t
  with anchor index a (t−1 under `anchor="state"`, with a virtual segment copying segment 0 before row 0;
  t under `"action"`):
  - window W = segments a … a+span−1 that lie before row R−1;
  - A = free progress over the non-respaced segments of W, N = their count;
  - A > 0: step = A · (span/H) / N; segment progress is `free` outside keyframes and
    step · event_density / (span/H) on respaced segments;
  - A = 0 (a still window, or one with no free segment): the frame is spaced in time: progress 1 on free
    recorded segments, `free` on padding segments, event_density on respaced segments, step = span/H;
  - point k (k = 1 … H) at cumulative progress s(a) + k·step, mapped to a fractional row index (strictly
    increasing drift as now), joints from `TrajectoryCurve.joints`, commands from the nearest row; a point
    past the episode's cumulative progress is padded.
  `Chunks` gains `step` (T,).
- `arc_core.invert(s, targets)` takes the cumulative progress array explicitly.
- `ResampleConfig.curvature_window` (a multiple of the episode's median step) is replaced by
  `curvature_length` (rad, required for `curvature`); `sweep --curvature-length` defaults to 4 × the
  dataset's path step and is printed; a recorded config with `curvature_window` is refused.
- `curve_check.py --dataset --spec --out`: the experiment grid below.
- `window_check.py --dataset --spec --keyframes --label-config --out`: reach of the H-th point
  (p05 / p50 / p95), |reach − span| over windows without keyframes, per-frame step ÷ span/H, replan jitter:
  for inference frames t and t+d (d = 1 and 10, both unpadded), the largest joint distance between their
  deployed targets (`arc_interp.to_frames` at `point_frames`, rate 1) over the frames both cover,
  median / p90 / max in path steps.

## Experiment

Variables: parameterization ∈ {row index (time), chord length, centripetal (chord length^½)}; interpolant ∈
{PCHIP, cubic spline}; curvature estimator ∈ {chord angles on raw rows with the relative window (current),
chord angles on raw rows with an absolute window, chord angles on the dense reconstruction (spacing ≤
window / 16) with an absolute window, for each reconstruction}. Absolute window: 4 × the dataset path step.

Measurements:
- LIBERO hold-out: keep every k-th recorded row (k = 2, 4, plus the last), reconstruct, distance of every
  dropped row to the reconstructed curve; median / p95 / max in path steps.
- Constructed path (straight, arc, straight; known curve and curvature) sampled at three teleop speed
  profiles (steady, 2× steady, accelerating / slowing / paused): largest distance of each reconstruction to
  the true curve; normalized curvature progress along arc length against the analytic one and its spread
  across the profiles.
- LIBERO overshoot: the same rebuilds measured the other way round, distance of the dense rebuilt curve to
  the full recorded polyline, which shows a curve leaving the path (the hold-out alone does not).
- LIBERO re-timing (uniform, 2×, sine, as `retime_check.py`): spread of the normalized progress per
  estimator. The copies are generated by chord-length PCHIP, which favours that reconstruction; the
  constructed path is the unbiased reference.

Selection rule: the reconstruction with the lowest LIBERO hold-out p95 at k = 4 (then k = 2); within 5% the
geometric parameterization wins. The curvature estimator with the smallest spread across speed profiles on
the constructed path whose largest error against the analytic progress is not larger than the current
estimator's.

Result (`curve_check.json`, 50 LIBERO episodes, path step 0.0306 rad, window 0.1224 rad):

| reconstruction | hold-out p95 k=2 / k=4 | overshoot p95 k=2 / k=4 | constructed geometry, mrad (steady / 2× / varying) |
|---|---|---|---|
| time / PCHIP | 0.046 / 0.186 | 0.030 / 0.134 | 0.115 / 0.776 / 0.164 |
| time / spline | **0.024 / 0.105** | **0.021 / 0.081** | 0.047 / 0.200 / 0.057 |
| chord / PCHIP | 0.050 / 0.201 | 0.031 / 0.153 | 0.115 / 0.776 / 0.166 |
| chord / spline | 0.046 / 0.219 | 0.028 / 0.183 | 0.047 / 0.201 / 0.059 |
| centripetal / PCHIP | 0.047 / 0.191 | 0.031 / 0.141 | 0.115 / 0.776 / 0.165 |
| centripetal / spline | 0.029 / 0.134 | 0.022 / 0.099 | 0.047 / 0.200 / 0.058 |

| curvature estimator | constructed error, max over profiles | constructed spread | LIBERO re-timing 2×, median / max |
|---|---|---|---|
| raw rows, relative window (current) | 0.0995 | 0.0577 | 0.0162 / 0.0450 |
| raw rows, absolute window | 0.0580 | **0.0067** | 0.0082 / 0.0310 |
| dense time / PCHIP | 0.0556 | 0.0099 | 0.0057 / 0.0239 |
| dense time / spline | 0.0626 | 0.0174 | 0.0052 / 0.0196 |
| dense chord / PCHIP | 0.0556 | 0.0096 | 0.0059 / 0.0225 |
| dense centripetal / PCHIP | 0.0556 | 0.0101 | 0.0058 / 0.0234 |

Selected: the time-parameterized not-a-knot cubic spline (hold-out p95 at k = 4 27% below the next
candidate, centripetal / spline; overshoot also lowest at p95), and chord angles over the recorded rows with
an absolute window. The window length decides the speed dependence (spread 0.0577 → 0.0067); computing the
curvature on the dense reconstruction instead of the rows changes the spread by ≤ 0.011 either way.

## Inputs

The LIBERO timed dataset (end padding 40), keyframes, label configurations.

## Outputs

`curve_check.json`, `window_check_episode_step.json` (before), `window_check_window_step.json` (after) in
the run directory; rebuilt sweeps, label datasets, replays at rates 1 and 2, data report; updated docs;
rebuilt deck.

## State Changes

Label datasets built with a geometric metric before this change are not reproducible by the new code; the
only ones in use are the LIBERO `curvature` datasets, which S7 rebuilds. `time` labels with span = H are
identical, checked against the existing LIBERO `time-s10` datasets.

# Flow

## Main Flow

Build labels: reconstruct the episode's curve (time-parameterized cubic spline through the recorded rows)
→ curvature of the recorded rows at a fixed chord length → metric progress per segment → for each frame,
the step from its own window → H points read from the curve. Deploy: unchanged; the server places point k at (offset + k·span/H)/r frames, so a chunk without
keyframes covers exactly span recorded frames in span/r control frames. Replay and sweep: unchanged code,
new labels.

# Code Mapping

## Modules

`src/vla_tools/trajectory_curve.py`, `src/vla_tools/arc_core.py`, `src/vla_tools/resample_config.py`,
`src/vla_tools/sweep.py` (only for `--curvature-length`), `tests/test_trajectory_curve.py`,
`tests/test_arc_core.py`, `tests/test_pipeline_chain.py`; `FTpi05/reports/action-sparsification-src/libero/`
`curve_check.py`, `window_check.py`; knowledge files above; deck files under `FTpi05/reports/`.

## Interfaces

`TrajectoryCurve`, `build` → `trajectory_curve.py`; `progress_curve`, `chunks`, `invert`, `ProgressCurve`,
`Chunks` → `arc_core.py`; JavaScript port → `FTpi05/reports/action-sparsification-src/deck/core.js`.

## State

Curve and per-frame step: `arc_core.py` (built per call). LIBERO artifacts: `FTpi05/runs/libero_alphabet_soup/`.

# Implementation

## Phase 1

Phase Status: completed

Round: 1

### Goal

Run the reconstruction experiment and record the selection; measure the current labels with
`window_check.py` before anything changes.

### Files

`FTpi05/reports/action-sparsification-src/libero/curve_check.py`, `window_check.py`,
`FTpi05/runs/libero_alphabet_soup/curve_check.json`, `window_check_episode_step.json`, `plan.md`
(selection recorded under Phase 2).

### Structures

None in the package.

### Affected Modules

Measurement.

### Dependencies

None.

### Observation

The full grid as one table; the selected parameterization, interpolant and curvature estimator with the
measured margins.

## Phase 2

Phase Status: completed

Round: 2

### Goal

Implement the selected reconstruction in `trajectory_curve` and the per-window sparsification in `arc_core`.

### Files

`src/vla_tools/trajectory_curve.py`, `src/vla_tools/arc_core.py`, `src/vla_tools/resample_config.py`,
`src/vla_tools/sweep.py`, `tests/test_trajectory_curve.py`, `tests/test_arc_core.py`,
`tests/test_pipeline_chain.py`.

### Structures

`TrajectoryCurve`, `ProgressCurve` fields, `Chunks.step`, window rule.

### Affected Modules

Curve, labels.

### Dependencies

Phase 1.

### Observation

On the constructed path: the curve passes through every row (≤ 1e-12) and the H-th point of every window
without keyframes is the recorded pose span frames after the state (≤ 1e-9); inside keyframe segments the
points per frame equal event_density ÷ (span/H); labels of recorded frames identical with 10 and 40
padding rows; LIBERO `time-s10` rebuilt against the datasets on disk (max difference printed); test
suites pass.

## Phase 3

Phase Status: completed

Round: 3

### Goal

Re-run S7 on LIBERO and measure the new labels with `window_check.py`.

### Files

`FTpi05/runs/libero_alphabet_soup/**`, `FTpi05/data/libero-alphabet-soup-*-train|eval`.

### Structures

None.

### Affected Modules

LIBERO run.

### Dependencies

Phase 2.

### Observation

`run_status` all passed; sweep selection printed; before / after table: reach, jitter at strides 1 and
10, path head / full, keyframe position, executed / recorded frames at rates 1 and 2.

## Phase 4

Phase Status: active

Round: 4

### Goal

Docs: reconstruction, window and sparsification in the action-labels skill and method reference; the
experiment and LIBERO numbers in validation; README; issues (ISSUE-001 by the selected estimator) and
opportunities.

### Files

`.claude/skills/vla-action-labels/SKILL.md`, `.claude/skills/vla-action-labels/references/method.md`,
`.claude/skills/vla-pipeline/references/failure-modes.md`, `docs/validation.md`, `issues.md`,
`opportunities.md`, `README.md`.

### Structures

None.

### Affected Modules

Knowledge.

### Dependencies

Phase 3.

### Observation

Numbers match the artifacts; no text describes the time-parameterized curve or an episode-wide step as
current.

## Phase 5

Phase Status: pending

Round: 4

### Goal

Deck: port the curve and the window rule to `core.js`; pages 3–4 show one frame's window and its points
instead of episode-wide progress ticks; titles and cover text describe reconstruct → window → sparsify;
LIBERO numbers from the re-run; publish to `gh-pages`.

### Files

`FTpi05/reports/action-sparsification-src/deck/**`, `FTpi05/reports/action-sparsification.html`,
`gh-pages:index.html`.

### Structures

None.

### Affected Modules

Deck.

### Dependencies

Phases 2, 3.

### Observation

`check_core.py` ≤ 1e-9 against `arc_core` including windows with keyframes and the episode end; smoke
tests pass; served page identical to the local build.
