# Plan

Plan Status: approved

# Problem

## Current

Action labels are resampled per observation frame (`arc_core.chunks`): one progress curve per episode,
then for frame t the H points at progress s(t−1) + k·step, sampled from the recording. The design intent
for the `curvature` metric is a purely geometric density proxy, curvature^(1/3) of the trajectory in
joint or end-effector space, for static tasks where trajectory timing does not matter except stopping at
keyframes such as a grasp.

Observed:

1. `curvature` uses `‖a[i+1] − 2a[i] + a[i−1]‖^(1/3)` per frame, a second difference over the frame index,
   i.e. acceleration × Δt². On constructed paths (measured with `arc_core._raw_metric`): a straight line
   traversed while accelerating gets 3.93 progress per unit length (geometric curvature: 0); the same
   circle of radius 0.2 gets 10.0 at speed v and 7.94 at 2v (geometric: 1.70 at both). On the UR3 data
   the cube root compresses |Δ²q| (p10–p90 span ≈ 7×) to ≈ 2×, and still frames progress at 80% of the
   rate of moving frames.
2. `sweep` admits a configuration by the deployed targets' error against the recording frame by frame
   in time (`head`, `full`), and by the command-edge shift in frames. For static tasks this penalises
   timing differences that do not matter; on the UR3 data it is why every shortlisted cell uses `time`.
3. `label_replay` re-observes at recorded frame t + e after executing e frames, assuming the robot is
   where the recording was at that time; with progress-spaced labels the executed timing differs.
4. The `arc` metric is not part of the design.
5. The presentation deck (FTpi05/reports/action-sparsification.html, published from the `gh-pages`
   branch) shows the method only on a synthetic path, uses the time-indexed criterion, and quotes UR3
   results; it has no robot rendering of real data. Published material uses public LIBERO data only.

## Problem

The curvature metric is not geometric, the parameter selection and the label replay judge resampled
labels by recorded timing instead of by path and keyframe position, and the deck cannot show the method
on a real arm with public data.

## Goal

- `curvature` progress per frame = κ^(1/3) × segment length in joint space, κ the path's geometric
  curvature: identical progress per unit length for the same path at different speeds (≤ 1% difference
  on the constructed paths), 0 on straight segments with the default floor, R^(−1/3) on a circle.
- `sweep` scores deployed targets by distance to the recorded path, keyframe position error and speed
  relative to the recording; admission and selection use these.
- `label_replay` runs closed loop: the next observation is the recorded state nearest to the executed
  pose, searched forward.
- One LIBERO task (Franka Panda, 50 demonstrations) goes through the agent's stages S1–S7 with the new
  tools; its grid and replay results replace the UR3 figures in the skill text and the deck.
- The deck gains a page with two Franka Panda animations of one LIBERO demonstration, the teleop
  playback and the execution of resampled labels, with the resampling parameters adjustable beside
  them; synthetic demos use a varying teleop speed.

# Structure

## Modules

- Labels: `src/vla_tools/arc_core.py`, `src/vla_tools/resample_config.py`, new
  `src/vla_tools/path_curvature.py` (geometric curvature of a sampled joint path).
- Scoring: new `src/vla_tools/path_metrics.py` (distance to a recorded path, keyframe position error,
  speed ratio), used by `src/vla_tools/sweep.py` and `src/vla_tools/label_replay.py`.
- Reporting: `src/vla_tools/data_report.py` (label and replay tables).
- Deployment: `src/vla_tools/arc_interp.py`, unchanged: point k at k·span/H frames, PCHIP from the
  current pose, so execution speed follows point spacing.
- Knowledge: `.claude/skills/vla-action-labels/**`, `docs/validation.md`, `README.md`, `opportunities.md`.
- LIBERO run (FTpi05 project, not this repository): download, conversion script and the run's
  artifacts under `FTpi05/reports/action-sparsification-src/libero/`, datasets under
  `FTpi05/data/libero-<task>*`, environment `FTpi05/.work/libero-venv` (h5py, mujoco).
- Deck (FTpi05 project): source and assets in `FTpi05/reports/action-sparsification-src/`, built page
  `FTpi05/reports/action-sparsification.html`, published as `index.html` on this repository's
  `gh-pages` branch.

## Responsibilities

`path_curvature` owns the geometric estimate; `arc_core` turns any per-frame progress into labels;
`path_metrics` owns every time-free comparison with the recording; `sweep` and `label_replay` only
orchestrate and report. The LIBERO conversion only reshapes the source file into a LeRobot v2.1
dataset; all processing after it is the agent's tools. The deck consumes the algorithms through a
JavaScript port checked against the Python implementation.

## State Ownership

- Label configuration: `ResampleConfig` (recorded in each label dataset's `meta/arc_resample.json`).
- Grid results: `sweep.json`, written by `sweep`; replay results: `replay_<labels>.json`, written by
  `label_replay`.
- LIBERO source file: read-only after download; converted dataset written once by the conversion
  script; later versions by the agent's tools.
- Franka Panda model used by the deck: `FTpi05/reports/action-sparsification-src/robot/`.

# Interface

## Interfaces

- `path_curvature.curvature(q: np.ndarray, window: float) -> np.ndarray`: κ at every sample of an
  (N, D) joint path; the turning angle between chords spanning `window` arc length on each side,
  divided by that arc length, so still samples do not produce spurious curvature.
- `ResampleConfig.metric: Literal["time", "curvature", "second_difference"]`; new fields
  `curvature_window: float` (multiples of the episode's median per-frame joint step, default 4) and
  `curvature_floor: float` (minimum κ^(1/3) as a fraction of the episode's length-weighted mean, default 0).
  `second_difference` is the previous `curvature`, kept to reproduce existing datasets; `arc` is removed.
  `from_dict` refuses a recorded `curvature` config without `curvature_window` (recorded before this
  change), naming `second_difference`.
- `path_metrics.path_deviation(targets, recorded, start, end) -> float`, `keyframe_position_error(...)`,
  `speed_ratio(targets, pose, recorded_step) -> float`; joint-space Euclidean distances in radians.
- `sweep` cell fields gain `path_head`, `path_full` (median, p90), `kf_position` (median, p90),
  `speed_ratio`; existing time-indexed fields stay as information. Admission: `path_head` median ≤ 1
  path step (median Euclidean per-frame joint step), `kf_position` median ≤ 1 path step, all command
  edges kept, and optional `--max-speed-ratio`, `--max-jump`, `--max-pad`.
- `label_replay` rows: `execute`, path deviation (median, p90, max), seams, keyframe position error,
  inferences, executed frames vs recorded frames, frames held at each command edge.
- LIBERO conversion: `convert_libero.py <task.hdf5> <out_dir>` writes a LeRobot v2.1 dataset with
  `observation.state` = 7 joint angles + gripper command, `action` = next frame's joint angles + gripper
  command (−1/+1 mapped to 0/1), the two camera images, 20 fps.

## Inputs

Timed datasets, `keyframes.json`, `robot_spec.json`, label configurations; one LIBERO task file from
`yifengzhu-hf/LIBERO-datasets` (HDF5 with per-step joint states, gripper states, end-effector position
and images); the robosuite Franka Panda model (MJCF and meshes, MIT license), the model LIBERO uses.

## Outputs

New metric and scoring in `vla_tools`; updated `sweep.json` and `replay_<labels>.json` schemas; a LIBERO
run with data report; updated skill, validation and README text; a rebuilt deck with a robot animation
page.

## State Changes

Recorded configs with `metric: "curvature"` from before the change can no longer be loaded silently;
they must be rebuilt with `second_difference` or the new `curvature`.

# Flow

## Main Flow

Build labels: per episode, `path_curvature.curvature` → per-frame progress κ^(1/3)·ℓ (windows respaced
as before) → `arc_core.chunks` per observation frame. Select: `sweep` deploys each frame's label through
`arc_interp.to_frames`, scores it with `path_metrics`, admits and selects by path criteria. Choose the
execution length: `label_replay` stitches closed loop and reports path, keyframe and speed figures.
LIBERO: download → convert → S1–S7 with the agent's tools → data report. Present: the deck's JavaScript
port reproduces labels and deployment for one LIBERO demonstration and animates the recorded and the
executed joint trajectories on the Franka Panda model.

# Code Mapping

## Modules

`src/vla_tools/path_curvature.py`, `arc_core.py`, `resample_config.py`, `path_metrics.py`, `sweep.py`,
`label_replay.py`, `data_report.py`; `tests/test_path_curvature.py`, `tests/test_arc_core.py`,
`tests/test_path_metrics.py`, `tests/test_pipeline_chain.py`, `tests/pipeline_chain.py`;
`.claude/skills/vla-action-labels/SKILL.md`, `.claude/skills/vla-action-labels/references/method.md`,
`docs/validation.md`, `README.md`, `opportunities.md`; LIBERO and deck files under `FTpi05/reports/`,
`FTpi05/data/`, `FTpi05/.work/`.

## Interfaces

`path_curvature.curvature` → `src/vla_tools/path_curvature.py`; `ResampleConfig` →
`src/vla_tools/resample_config.py`; path metrics → `src/vla_tools/path_metrics.py`; LIBERO conversion →
`FTpi05/reports/action-sparsification-src/libero/convert_libero.py`; deck port →
`FTpi05/reports/action-sparsification-src/`.

## State

`sweep.json`, `replay_<labels>.json` and `arc_resample.json` in projects using the harness; the LIBERO
run's artifacts under `FTpi05/reports/action-sparsification-src/libero/run/`.

# Implementation

## Phase 1

Phase Status: active

Round: 1

### Goal

Geometric `curvature` metric; previous metric renamed `second_difference`; `arc` removed.

### Files

`src/vla_tools/path_curvature.py`, `src/vla_tools/arc_core.py`, `src/vla_tools/resample_config.py`,
`tests/test_path_curvature.py`, `tests/test_arc_core.py`.

### Structures

`path_curvature.curvature`; `ResampleConfig.curvature_window`, `curvature_floor`.

### Affected Modules

Labels.

### Dependencies

None.

### Observation

Tests print: progress per unit length for one path at three teleop speed profiles (spread), on a
straight line, and on circles of three radii against R^(−1/3); a recorded old-style config is refused.

## Phase 2

Phase Status: active

Round: 1

### Goal

Time-free scoring in `sweep`; default metrics `time curvature`.

### Files

`src/vla_tools/path_metrics.py`, `src/vla_tools/sweep.py`, `src/vla_tools/data_report.py`,
`tests/test_path_metrics.py`, `tests/test_pipeline_chain.py`.

### Structures

Path deviation, keyframe position error, speed ratio; new cell fields and admission rule.

### Affected Modules

Scoring, reporting.

### Dependencies

None.

### Observation

Unit tests on hand-built paths (known distances, a shifted command edge, a doubled step); on the
synthetic chain the `time`, span = H cell scores 0 and selection follows the stated rule.

## Phase 3

Phase Status: pending

Round: 2

### Goal

Closed-loop `label_replay` with path, keyframe, speed and hold figures.

### Files

`src/vla_tools/label_replay.py`, `src/vla_tools/data_report.py`, `tests/test_pipeline_chain.py`,
`tests/pipeline_chain.py`.

### Structures

Forward nearest-state re-observation.

### Affected Modules

Scoring, reporting.

### Dependencies

Phase 2 (`path_metrics.py`, `data_report.py`, `tests/test_pipeline_chain.py`).

### Observation

Synthetic chain: `time`, span = H replay has 0 path deviation and ⌈frames / e⌉ inferences; observation
indices increase monotonically for every configuration; figures printed per execution length.

## Phase 4

Phase Status: completed

Round: 1

### Goal

LIBERO assets: environment in `FTpi05/.work/libero-venv`; one LIBERO task file downloaded; the robosuite
Franka Panda model copied under the deck source with its license; a JavaScript kinematic chain built
from the MJCF, driven by 7 joint angles, with the gripper fingers driven by the command.

### Files

`FTpi05/reports/action-sparsification-src/robot/` (MJCF, meshes, license, chain module, check script),
`FTpi05/reports/action-sparsification-src/libero/download.md` (source, file, license, size, checksum).

### Structures

Joint-angle → link-pose chain from the MJCF.

### Affected Modules

Deck, LIBERO run.

### Dependencies

None.

### Observation

The downloaded file has per-step joint states (stop and report if not). The chain's end-effector site
matches MuJoCo forward kinematics of the same MJCF on 100 recorded frames to ≤ 1e-6 m, and its offset
from the recorded end-effector position is reported; one rendered frame inspected.

## Phase 5

Phase Status: pending

Round: 3

### Goal

Run the LIBERO task through the agent: convert to LeRobot v2.1, then S0–S7 with the new metric and
scoring (audit, clean, keyframes, timing, diversity, sweep, closed-loop replay, labels), and the data
report.

### Files

`FTpi05/reports/action-sparsification-src/libero/convert_libero.py`,
`FTpi05/reports/action-sparsification-src/libero/run/**` (specs, plans, reports), `FTpi05/data/libero-<task>*`.

### Structures

LIBERO robot spec: 7 joint dims, 1 command dim, 20 fps.

### Affected Modules

LIBERO run.

### Dependencies

Phases 1–4.

### Observation

`run_status` reports every stage passed; `data_report.md` written; grid and replay tables printed.

## Phase 6

Phase Status: pending

Round: 4

### Goal

Update the skill, method reference, validation table, README wording and OPT-001 with the new metric,
the path criteria and the LIBERO results.

### Files

`.claude/skills/vla-action-labels/SKILL.md`, `.claude/skills/vla-action-labels/references/method.md`,
`docs/validation.md`, `README.md`, `opportunities.md`.

### Structures

None.

### Affected Modules

Knowledge.

### Dependencies

Phase 5.

### Observation

LIBERO tables recorded with date and command; no skill text describes the removed `arc` metric or the
time-indexed admission.

## Phase 7

Phase Status: pending

Round: 4

### Goal

Rebuild the deck: sparsification and deployment pages on a synthetic path with varying teleop speed,
using the new metric and path scoring; a new page with two Franka Panda animations of one LIBERO
demonstration, teleop playback beside the execution of resampled labels, with metric, span, keyframe
density and execution length adjustable; real-data figures from the LIBERO run only. Publish to
`gh-pages`.

### Files

`FTpi05/reports/action-sparsification-src/**`, `FTpi05/reports/action-sparsification.html`,
`gh-pages:index.html` (plus the Panda meshes and their license if served as separate files).

### Structures

JavaScript port of `path_curvature`, `arc_core.chunks`, `arc_interp.to_frames` and the closed-loop
stitching.

### Affected Modules

Deck.

### Dependencies

Phases 1, 3, 4, 5.

### Observation

The port matches Python labels, deployed targets and stitched trajectories on reference cases
(≤ 1e-9); the page script runs without errors under a stub DOM; the served page is identical to the
local build and contains no UR3 data.
