# Plan

Plan Status: completed

# Problem

## Current

- Labels in the LIBERO alphabet-soup simulator come in two families (docs/validation.md → LIBERO simulator):
  per-frame time labels (Fpp: time metric, span 10, H 10, per-point action statistics) succeed most at the
  recorded pace (83–89% in 158–189 frames); curvature-resampled labels executed with velocity timing (S3b:
  curvature, span 32, event density 1, per-point statistics) finish 2.1–2.6× faster at 77–81%.
- `arc_core.chunks` already expresses a two-phase sampling: under metric `time` outside keyframe windows the
  points are spaced `span/H` recorded frames apart, evenly in time; a segment inside a keyframe window carries
  `event_density` points per `span/H` frames. `event_density = span/H` puts one point on every recorded frame
  inside the windows.
- The keyframe windows of the run (grasp −6 / +4, lift −2 / +6, release −4 / +6 frames) cover the frames
  right at the events. The end effector is within 20 / 40 / 60 mm of the grasp pose 3 / 5 / 7 frames before
  the grasp, and of the release pose 5 / 8 / 10 frames before the release.
- Deployment executes at 1.0 of the joint velocity limits (`arc_interp.DEPLOY_FRACTION`); S3b was evaluated
  closed loop only at that fraction, while the labels open loop succeed more at 0.6–0.8 than at 1.0 under cap 8.

## Problem

Whether labels that are sparse and uniform in time during the gross motion and dense (one point per recorded
frame) during the fine manipulation train a policy that keeps the per-frame labels' success at the resampled
labels' speed is unmeasured; and whether S3b's deployment fraction should be below 1.0 is unmeasured closed
loop.

## Goal

- Two-phase labels (TP): metric `time`, span 32, H 10, `event_density` 3.2, anchor `state`, fine-phase
  windows grasp −8 / +4, lift −2 / +6, release −10 / +6 frames; same held-out episodes (3 5 12 21 34) and
  training recipe as S3b (per-point action statistics, 2 GPUs, 10k steps).
- Closed-loop success and frames for TP under the matched conditions (task, initial states, kp 400, measured
  state, budget 280, three runs per cell), beside Fpp and S3b; TP labels open loop under the same execution.
- S3b closed loop at velocity fraction 0.8 (cap 4 and cap 8, three runs); `DEPLOY_FRACTION` becomes 0.8 when
  its success exceeds fraction 1.0's by more than the run-to-run spread at both caps.

# Structure

## Modules

- Label sampling: `src/vla_tools/arc_core.py`, `src/vla_tools/build_labels.py` (used unchanged).
- Interpolation: `src/vla_tools/arc_interp.py` (`DEPLOY_FRACTION`).
- Evaluation scripts outside the repository: `reports/sim-eval-src/gripper_trace.py`,
  `reports/sim-eval-src/replay_with_command_rule.py`.
- Run inputs and outputs outside the repository: `runs/libero_alphabet_soup/`.
- Validation record: `docs/validation.md`.

## Responsibilities

The run directory owns the TP keyframe windows and label config; `build_labels` turns them into datasets;
the evaluation scripts measure; `docs/validation.md` records the measured results.

## State Ownership

- Fine-phase windows: `runs/libero_alphabet_soup/keyframes_two_phase.json` (the run's keyframes with the
  windows widened).
- TP label config: `runs/libero_alphabet_soup/label_config_two_phase.json`.
- Deployment fraction: `arc_interp.DEPLOY_FRACTION`.

# Interface

## Interfaces

No new interface. `build_labels --keyframes keyframes_two_phase.json --config label_config_two_phase.json`;
`run.py norm-stats --per-point-actions`, `run.py train`, `run.py serve-points` with config name
`pi05_libero_alphabet_soup_two_phase_pp`; `gripper_trace.py` cells as in
`runs/libero_alphabet_soup/sim/matched/run.sh`.

## Inputs

`data/libero-alphabet-soup-timed`, `runs/libero_alphabet_soup/robot_spec.json`, the run's keyframes.

## Outputs

Datasets `data/libero-alphabet-soup-two-phase-{train,eval}`; checkpoint
`openpi/checkpoints/pi05_libero_alphabet_soup_two_phase_pp/run/9999`; traces and logs under
`runs/libero_alphabet_soup/sim/two_phase/`; a section of `docs/validation.md`.

## State Changes

None in the repository unless phase 4's condition holds (`DEPLOY_FRACTION` 1.0 → 0.8).

# Flow

## Main Flow

Keyframes with fine-phase windows + TP config → `build_labels` → parquet features rewrite → per-point norm
stats → training → checkpoint with `arc_resample.json` copied into its assets → `serve-points` →
`gripper_trace.py` closed loop per cell; `replay_with_command_rule.py --source labels` for the labels open
loop.

# Code Mapping

## Modules

As in Structure.

## Interfaces

As in Interface.

## State

As in State Ownership.

# Implementation

## Phase 1

Phase Status: completed

Round: 1

### Goal

TP labels.

### Files

`runs/libero_alphabet_soup/keyframes_two_phase.json`, `runs/libero_alphabet_soup/label_config_two_phase.json`,
`data/libero-alphabet-soup-two-phase-{train,eval}`, `openpi/assets/pi05_libero_alphabet_soup_two_phase_pp/`.

### Structures

None.

### Affected Modules

None in the repository.

### Dependencies

None.

### Observation

Sample positions of a chunk outside and inside the windows; dataset summary; norm-stats shapes.

Result: 45 training and 50 evaluation episodes, window fraction 32.5% of frames, padding 8.8%. Episode 0:
frame 10's points at recorded frames 12.2, 15.4, …, 31.4 (3.2 apart) then 34.2, 35.2, 36.2 once inside the
grasp window; frames 38 and 42 one frame apart over 10 frames; frame 128 leaves the release window after
7 points (then 3.2 apart). Action statistics (10, 8).

## Phase 2

Phase Status: completed

Round: 2

### Goal

Train TP.

### Files

`openpi/checkpoints/pi05_libero_alphabet_soup_two_phase_pp/`, `runs/libero_alphabet_soup/train/`.

### Structures

None.

### Affected Modules

None in the repository.

### Dependencies

Phase 1.

### Observation

Training log loss at steps 0, 100, 9900 against Fpp and S3b; checkpoint 9999 present with `arc_resample.json`.

Result: loss 0.1223 / 0.0875 / 0.0061 at steps 0 / 100 / 9900 (Fpp 0.1171 / 0.0804 / 0.0058, S3b 0.0070 at 9900);
17 GB per GPU with memory on demand, 3.1 s per step on GPUs shared with other jobs; checkpoint 9999 with the report.

## Phase 3

Phase Status: completed

Round: 3

### Goal

Closed-loop and open-loop measurements: TP cells (cap 4 at fraction 1.0 and 0.8, cap 8 at 0.8, cap 4 with
0.1 s async latency, uniform rate 1 with execution 4); TP labels open loop at the same cells; S3b at cap 4
and cap 8 with fraction 0.8.

### Files

`runs/libero_alphabet_soup/sim/two_phase/`.

### Structures

None.

### Affected Modules

None in the repository.

### Dependencies

Phase 2 for TP; the S3b cells and TP labels open loop depend on phase 1 only.

### Observation

Success per run (of 50), mean frames of successful episodes, per cell.

Result (docs/validation.md → LIBERO simulator, Two-phase labels): TP at cap 4 41.7 (96 frames), cap 8 44.3 (58),
cap 4 with 0.1 s async 43.3 (80), fraction 0.8 39.3 / 42.0, uniform rate 1 10.3 (budget-bound); labels open loop
42–48. S3b at fraction 0.8: cap 4 41.3 (66), cap 8 39.7 (56) against 40.7 and 38.7 at 1.0.

## Phase 4

Phase Status: completed

Round: 4

### Goal

Record the results; set `DEPLOY_FRACTION` to 0.8 when phase 3's S3b condition holds, with its tests, the
skills and README mentions, and push.

### Files

`docs/validation.md`, `issues.md`; conditionally `src/vla_tools/arc_interp.py`, `tests/test_arc_interp.py`,
`.claude/skills/vla-downstream/SKILL.md`, `.claude/skills/vla-action-labels/SKILL.md`, `README.md`.

### Structures

None.

### Affected Modules

Validation record; conditionally interpolation.

### Dependencies

Phase 3.

### Observation

Docs tables match the trace files; full test suite and ruff when code changes; remote main equals local main
after a push.

Result: docs/validation.md and ISSUE-011 updated. S3b's fraction-0.8 gain (0.6 and 1.0 successes) lies inside the
run-to-run spread, so `DEPLOY_FRACTION` stays 1.0; no code change, no push.
