# Plan

Plan Status: completed

# Problem

## Current

- LIBERO alphabet soup, policies S (`curvature` s64 d0), S2 (s64 d2) and F (`time` s10), trained identically.
  Closed loop, sync, L = 0 (docs/validation.md → LIBERO simulator): S 28–60%, S2 25–63%, F 61–86% over the
  settings measured; executing 5 targets per inference at rate 1 (e 5), 4 at velocity cap 4, or whole chunks.
- The simulator pipeline and the training data are checked; S's and S2's failures are their grippers resting
  on the object above the commanded pose (ISSUE-011).

## Problem

How S, S2 and F behave when they replan more often (fewer targets executed per inference) is not measured;
whether a resampling configuration with nearer points (span 32), keyframe windows at the recorded pace (event
density 1) and, separately, per-point normalization statistics narrows the gap to F is not measured.

## Goal

Closed loop of S, S2 and F, sync, L = 0, kp 400, 50 trials, nearest-point command rule: rate 1 with e 1, 2, 3;
velocity cap 4 with holds (0.2 / 0.55 s) with e 1, 2; the labels open loop in the same cells. Per cell
successes, failure phases, close-attempt classes.

- The training data define state[t + 1] = action[t]; closed loop the arm reaches about half of a commanded
  step per frame. The same cells with the commanded joints and command as the observation state
  (`gripper_trace.py --state commanded`).
- S3a: labels `curvature` span 32, event density 1, otherwise S's (H 10, anchor state, chord 0.1224 rad,
  holdout 3 5 12 21 34), trained as S on two L40S. S3b: the same labels with action normalization statistics
  per point index (10 x 8 q01 / q99, mean, std; state per dimension). Both evaluated as S2 was, with both
  state sources.
- Velocity timing with switch holds (the execution intended for resampled labels): S3a, S3b and F, measured
  state, e 4, caps 2, 4, 8, 16 at L = 0 and caps 4, 8 at L = 0.1 s async; cap 4 three times; labels open loop
  of S3 and F in the same cells (`runs/libero_alphabet_soup/sim/velocity/`).
- Matched comparison of the two strategies, everything else equal (task, 50 initial states, kp 400, measured
  state, budget 280, three runs per cell counting earlier runs under the same conditions): per-frame F at the
  recorded pace (uniform timing, no holds; e 10 and e 5 at L = 0, e 8 at L = 0.1 s async, e 10 at L = 0.1 s
  sync) against S3b with velocity timing and holds (caps 4 and 8, e 4, at L = 0 and L = 0.1 s async; cap 4 at
  L = 0.1 s sync). Then Fpp, F's labels with per-point action statistics trained on two GPUs as S3b was, in
  the same cells (`runs/libero_alphabet_soup/sim/matched/`).

# Structure

## Modules

Existing scripts outside the repository: `reports/sim-eval-src/gripper_trace.py` (closed loop with traces),
`replay_with_command_rule.py` (labels open loop; no rule applied when COMMAND_RULE is unset),
`gripper_attempts.py`, `close_check.py`.

## Responsibilities

As in the scripts' docstrings.

## State Ownership

Results: `runs/libero_alphabet_soup/sim/replan/`.

# Interface

## Interfaces

None new.

## Inputs

The S, S2 and F checkpoints at step 9999 and their label configs.

## Outputs

Trace and label replay files per policy and cell.

## State Changes

None.

# Flow

## Main Flow

1. Servers: S on GPU 3, S2 on GPU 4, F on GPU 7; rendering on EGL devices 6, 1, 2 (devices that rendered with
   that server placement); one warm-up request each.
2. Closed loop per policy and cell, then the labels open loop.
3. Table with the earlier e 4 / e 5 / whole-chunk cells; results in `docs/validation.md`, ISSUE-011.

# Code Mapping

## Modules

No repository code changes.

## Interfaces

None.

## State

`runs/libero_alphabet_soup/sim/replan/`.

# Implementation

## Phase 1

Phase Status: completed

Round: 1

### Goal

Replanning grid results; results recorded.

### Files

Outside the repository `runs/libero_alphabet_soup/sim/replan/`; `docs/validation.md`, `issues.md`.

### Structures

None.

### Affected Modules

Knowledge.

### Dependencies

Free GPUs.

### Observation

Per policy and cell: successes, phases, close-attempt classes; labels open-loop successes.

Result: replanning grid, commanded-state check, S3a / S3b, velocity grid and the matched comparison recorded in
docs/validation.md → LIBERO simulator; ISSUE-011 and ISSUE-017 updated. F at caps 2, 8, 16 under velocity timing
was not run (GPUs taken by other jobs); Fpp replaced it as the matched per-frame policy.
