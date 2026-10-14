# Plan

Plan Status: approved

# Problem

## Current

A label is H points per observation frame; the deployed system rebuilds the trajectory from those few
points. The repository interpolates in two places:

- `trajectory_curve` (label side): a not-a-knot cubic spline over the row index through every recorded
  row, used only to read a point's joints between two recorded rows. It was selected by a hold-out
  experiment (keep every 4th row, rebuild, measure the dropped rows), which is a rebuild from few points,
  but its result was applied here, where every row is a knot: the label points lie p50 0.017 / p90 0.03 /
  max 0.09 path steps off the recorded polyline whatever the interpolant.
- `arc_interp.to_frames` (server side): PCHIP over the knots `(offset + k·span/H) / rate` through the
  current pose and the H points. This is the rebuild from few points; `sweep` and `label_replay` score it.

Rebuild error from the label points themselves, LIBERO 50 episodes, unpadded frames
(`FTpi05/reports/action-sparsification-src/libero/sparse_rebuild_check.py`), larger of recorded rows to
the rebuilt curve and the rebuilt curve to the recorded path, path steps, p50 / p90 / max:

| rebuild from the H points | `curvature` s40 d2 | `curvature` s40 d0 |
|---|---|---|
| polyline | 0.21 / 0.36 / 0.98 | 0.33 / 0.73 / 1.33 |
| PCHIP, uniform knots (server today) | 0.11 / 0.24 / 0.77 | 0.23 / 0.60 / 1.20 |
| cubic spline, uniform knots | 0.09 / 0.21 / 0.74 | 0.17 / 0.44 / 0.74 |
| cubic spline, centripetal | 0.09 / 0.21 / 0.73 | 0.19 / 0.54 / 1.10 |
| PCHIP, chord | 0.12 / 0.25 / 0.81 | 0.24 / 0.63 / 1.24 |
| cubic spline, chord | 0.11 / 0.30 / 1.40 | 0.22 / 0.72 / 2.41 |
| PCHIP, the points' true recorded rows (not available to the server) | 0.11 / 0.24 / 0.75 | 0.23 / 0.59 / 1.19 |

## Problem

1. The interpolant selected for rebuilding from few points is not the one that rebuilds from the label
   points; the server still uses PCHIP.
2. The label side carries a second interpolant that has no measurable role.

## Goal

One reconstruction in the system: the rebuild from the H label points.

- Server: not-a-knot cubic spline over the same knots (timing and execution rate unchanged).
- Labels: a point's joints are the linear interpolation of its two neighbouring recorded rows, so every
  label point lies on the recorded polyline (distance ≤ 1e-12 rad).
- The spline's per-joint overshoot between neighbouring nodes is measured, in keyframe windows separately
  (PCHIP has none by construction), together with the replay's hold frames and speed ratio.
- LIBERO S7 re-run and before / after comparison; documentation and deck describe the rebuild from few
  points.

# Structure

## Modules

- Server rebuild: `src/vla_tools/arc_interp.py`.
- Labels: `src/vla_tools/arc_core.py`; `src/vla_tools/trajectory_curve.py` is removed.
- Measurement: `FTpi05/reports/action-sparsification-src/libero/sparse_rebuild_check.py`.
- LIBERO run: `FTpi05/runs/libero_alphabet_soup/`, `FTpi05/data/libero-alphabet-soup-curvature-s40-d2-*`.
- Knowledge: `.claude/skills/vla-action-labels/` (SKILL.md, references/method.md), `docs/validation.md`,
  `README.md`.
- Deck: `FTpi05/reports/action-sparsification-src/deck/`, `FTpi05/reports/action-sparsification.html`,
  `gh-pages`.

`sweep`, `label_replay`, `build_labels` and `vla_openpi.launch` call `arc_interp.to_frames` /
`ArcInterpolator` and `arc_core.chunks` through unchanged interfaces.

## Responsibilities

`arc_interp` owns the reconstruction from the H points. `arc_core` owns the label points, read from the
recorded rows.

## State Ownership

- Rebuild interpolant: `arc_interp.to_frames`.
- Label point joints: `arc_core.chunks`.

# Interface

## Interfaces

- `arc_interp.to_frames(points, current, times, command_dims) -> np.ndarray`: unchanged signature; joints
  from `scipy.interpolate.CubicSpline(knots, values, axis=0, bc_type="not-a-knot")` (two knots: the chord,
  three: the parabola).
- `arc_core.chunks(...)`: unchanged signature; `points[..., joints]` linear between rows ⌊f⌋ and ⌈f⌉ of the
  recorded joints, f clamped to [0, recorded − 1].

## Inputs

LIBERO timed dataset, keyframes, robot spec, `label_config.json` (unchanged).

## Outputs

`sparse_rebuild_check.json` with an overshoot column; rebuilt label datasets; S7 artifacts.

## State Changes

None beyond the outputs.

# Flow

## Main Flow

Labels: progress → per-window targets → fractional rows → joints linear between recorded rows.
Server: current pose + H points at `(offset + k·span/H) / rate` → cubic spline → per-frame targets.

# Code Mapping

## Modules

Server rebuild → `src/vla_tools/arc_interp.py`; labels → `src/vla_tools/arc_core.py`.

## Interfaces

`to_frames` → `arc_interp.py`; `chunks` → `arc_core.py`.

## State

Rebuild interpolant → `arc_interp.py`; label point joints → `arc_core.py`.

# Implementation

## Phase 1

Phase Status: completed

Round: 1

### Goal

Server rebuild with the not-a-knot cubic spline.

### Files

`src/vla_tools/arc_interp.py`, `tests/test_arc_interp.py`.

### Structures

None.

### Affected Modules

Server rebuild; `sweep` and `label_replay` through `to_frames`.

### Dependencies

None.

### Observation

Tests; `to_frames` equals `sparse_rebuild_check`'s spline/uniform samples at the frame times; the rate-2
target j equals the rate-1 target 2j+1.

## Phase 2

Phase Status: active

Round: 1

### Goal

Label points linear between recorded rows; remove `trajectory_curve`.

### Files

`src/vla_tools/arc_core.py`, `src/vla_tools/trajectory_curve.py` (removed),
`tests/test_trajectory_curve.py` (removed), `tests/test_arc_core.py`.

### Structures

None.

### Affected Modules

Labels.

### Dependencies

None.

### Observation

Tests; label points' distance to the recorded polyline on LIBERO; largest change of the label points
against the spline lookup; `time` s10 labels unchanged.

## Phase 3

Phase Status: pending

Round: 2

### Goal

LIBERO S7 re-run with both changes; overshoot measurement.

### Files

`sparse_rebuild_check.py` (overshoot column), `FTpi05/runs/libero_alphabet_soup/*`,
`FTpi05/data/libero-alphabet-soup-curvature-s40-d2-train|eval`.

### Structures

None.

### Affected Modules

LIBERO run.

### Dependencies

Phases 1 and 2.

### Observation

Sweep selection and cells, replays (density 0 / 2, rates 1 / 2, time s10), window check, retime, sparse
rebuild check, run_status S0–S7, data report; before / after table.

## Phase 4

Phase Status: pending

Round: 3

### Goal

Documentation of the rebuild from few points.

### Files

`docs/validation.md`, `.claude/skills/vla-action-labels/SKILL.md`,
`.claude/skills/vla-action-labels/references/method.md`, `README.md`, `src/vla_tools/arc_core.py`
(docstring), `src/vla_tools/arc_interp.py` (docstring).

### Structures

None.

### Affected Modules

Knowledge.

### Dependencies

Phase 3.

### Observation

No text describes the every-row curve or PCHIP as the rebuild; the tables match the phase 3 artifacts.

## Phase 5

Phase Status: pending

Round: 3

### Goal

Deck: server rebuild as cubic spline, label lookup linear; page text; publish to gh-pages.

### Files

`FTpi05/reports/action-sparsification-src/deck/core.js`, `check_core.py`, `deck.html`, `page.js`,
`README.md`, `FTpi05/reports/action-sparsification.html`.

### Structures

None.

### Affected Modules

Deck.

### Dependencies

Phase 3.

### Observation

`check_core.py` against Python; smoke test; screenshots of pages 3 and 4; published file equal to local.
