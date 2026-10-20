# Plan

Plan Status: completed

# Problem

## Current

- `ArcInterpolator` gives every per-frame target the command of the nearest point in time; the observed pose
  is not a knot for commands. With S's spacing at rate 1 (points 6.4 frames apart) targets 1–9 carry point
  1's command, so a close predicted for point 1 is executed at the next frame (docs/validation.md → LIBERO
  simulator, Grasp accuracy open loop and close timing).
- Rebuilt from the labels, S's (density 0) first target closes 4 frames before the recorded close, 22 mm
  before the grasp pose; S2's (density 2) 1 frame early, 4 mm. Closed loop at e = 5 every first close of S
  and S2 is a chunk's first target, 45–49 mm from the object when it misses (ISSUE-011).
- Open loop S places the grasp as accurately as F at the same lookahead.

## Problem

Whether a command rule that does not move a predicted switch forward raises S's and S2's success is not
measured.

## Goal

Closed loop of S and S2 (existing checkpoints) and their labels open loop, sync, L = 0, kp 400, 50 trials,
in the cells rate 1 e 5, rate 1 whole chunks, velocity cap 4 with holds e 4, under three command rules:

- `nearest`: the nearest point in time (current rule);
- `midpoint`: the nearest knot with the commanded gripper as a knot at time 0 (a switch between the observed
  command and point 1 happens half way to point 1);
- `hold`: the last knot at or before the target's time, the commanded gripper at time 0 (a switch happens at
  the first point that carries it).

Per cell successes, failure phases and close-attempt classes.

Whether the arm lags the plan at the close: commanded joints recorded per frame; at each first close the
commanded and the actual tool-centre-point distance to the object. Two remedies that need no training:
half speed (rate 0.5, 10 targets per chunk, budget 560 frames), and `settle`, the nearest point's command
with the switching target held 3 frames with the old command before every switch.

# Structure

## Modules

- The rule under test replaces `vla_tools.arc_interp._timed` at run time from a script outside the
  repository (`FTpi05/reports/sim-eval-src/command_rule.py`, selected by the environment variable
  `COMMAND_RULE`); the repository is not changed.
- Closed loop: `reports/sim-eval-src/gripper_trace.py` (successes, phases and per-frame traces); labels open
  loop: `vla_libero.replay --source labels` through a launcher that applies the rule.

## Responsibilities

`command_rule.py` owns the three rules; the launcher applies the rule in the main process and in every
spawned worker; `gripper_attempts.py` classes the closes.

## State Ownership

Results: `runs/libero_alphabet_soup/sim/command_rule/`.

# Interface

## Interfaces

`command_rule.timed(points, current, times, command_dims, timing, rule) -> (targets, chunk_times)`, the
signature of `arc_interp._timed` plus the rule; `COMMAND_RULE` in {nearest, midpoint, hold}.

## Inputs

Checkpoints `pi05_libero_alphabet_soup_curvature_s64_{d0,d2}/run/9999`; label configs and datasets of S and S2.

## Outputs

Trace and label replay files per rule, policy and cell.

## State Changes

None.

# Flow

## Main Flow

1. Unit check of the rules on constructed points: `nearest` reproduces `arc_interp._timed` exactly; the
   first closed target under each rule for a close at point 1, 2 and between.
2. Two servers per policy (S on one GPU, S2 on another), one warm-up request each.
3. Per rule and cell: `gripper_trace.py` for S and S2; labels open loop for both label sets.
4. Table of successes and close-attempt classes; results in `docs/validation.md`, ISSUE-011.

# Code Mapping

## Modules

`FTpi05/reports/sim-eval-src/command_rule.py`, `replay_with_command_rule.py` (launcher for the labels open loop); `gripper_trace.py`
reads `COMMAND_RULE`.

## Interfaces

`command_rule.timed`, `command_rule.apply`.

## State

`runs/libero_alphabet_soup/sim/command_rule/`.

# Implementation

## Phase 1

Phase Status: completed

Round: 1

### Goal

The rules, unit-checked.

### Files

Outside the repository: `reports/sim-eval-src/command_rule.py`, `replay_with_command_rule.py`, `gripper_trace.py`.

### Structures

`timed`, `apply`.

### Affected Modules

None in the repository.

### Dependencies

None.

### Observation

`nearest` against `arc_interp._timed` on random chunks (maximum difference); first closed target per rule
on constructed chunks.

Result: `nearest` equals `arc_interp._timed` on 200 random chunks with and without velocity timing (maximum
difference 0). With S's spacing (6.4 frames) a close at point 1 gives the first closed target at frame 1
(`nearest`), 4 (`midpoint`), 7 (`hold`); at point 2: 10, 10, 13. The rule is in effect in spawned workers
and through `ArcInterpolator`.

## Phase 2

Phase Status: completed

Round: 2

### Goal

Grid results; results recorded.

### Files

Outside the repository `runs/libero_alphabet_soup/sim/command_rule/`; `docs/validation.md`, `issues.md`.

### Structures

None.

### Affected Modules

Knowledge.

### Dependencies

Phase 1; two free GPUs.

### Observation

Per rule, policy and cell: successes, phases, close-attempt classes and distances; labels open-loop
successes.

Result: no rule, half speed or settling raises S's or S2's successes beyond run-to-run spread (S 13–32,
S2 7–33, F 30–35 of 50 across the cells; labels 38–50). At the first close the commanded height equals F's,
while S's and S2's grip site stays 13–26 mm above it in failed trials, 21–24 mm off the object's centre
horizontally. The training data check out (stored labels equal recomputed ones; openpi's targets are label
minus state, normalized within −1.25…+1.19). Recorded in docs/validation.md → LIBERO simulator (Command rule,
execution speed, settling; Data checks); ISSUE-011 updated.
