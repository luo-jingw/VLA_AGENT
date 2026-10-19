# Plan

Plan Status: approved

# Problem

## Current

- LIBERO alphabet soup (libero_object task 0), two pi0.5 policies trained identically with the VLM frozen:
  S on `curvature` span 64, event density 0 labels, F on per-frame `time` span 10 labels.
- Sync whole-chunk grid and close-attempt traces, 2026-10-19 (docs/validation.md → LIBERO simulator): S
  succeeds 17–33 of 50 against 32–43 for F; executing whole chunks does not close the gap; 44–63% of S's
  commanded closes shut the fingers on nothing, a median 6–7 cm from the object, against 10–13% for F; the
  labels open loop succeed 44–48 of 50 in every cell.
- S's labels have event density 0: the keyframe windows (close, lift, release) get no extra points; with the
  stop at the object adding no progress to the curvature coordinate, the grasp is covered by one or two of
  the 10 points (ISSUE-011).

## Problem

Whether denser points in the keyframe windows (event density 2) remove S's missed grasps is not measured.

## Goal

- Labels `curvature` span 64, event density 2, everything else as S (H 10, anchor state, chord 0.1224 rad,
  key and pad weight 1; train excludes episodes 3, 5, 12, 21, 34; eval all 50).
- Policy S2 trained as S (`pi05_base`, SigLIP and Gemma 2B frozen, batch 32, 10k steps, cosine learning
  rate 500 warm-up, peak 2.5e-5, end 2.5e-6), on two L40S data parallel.
- S2 closed loop in the S cells of the whole-chunk grid (rate 1, e 5 and full; velocity cap 4 with holds,
  e 4 and full), the labels open loop at the same settings, close-attempt classes at rate 1 (e 5, full);
  compared with S and F of 2026-10-19.

# Structure

## Modules

- Data stages, existing tools: `vla_tools.build_labels`; `FTpi05/reports/sim-eval-src/parquet_features_compat.py`
  (ISSUE-009).
- Training and serving: `examples/libero_object/run.py`, unchanged.
- Simulator evaluation: `vla_libero.evaluate`, `vla_libero.replay`, unchanged; drivers and the close-attempt
  scripts in `FTpi05/reports/sim-eval-src/`.

## Responsibilities

`build_labels` writes the label datasets; `run.py` computes norm stats, trains and serves; the drivers run the
grid cells and classify close attempts.

## State Ownership

- Label config: `runs/libero_alphabet_soup/label_config_curvature_s64_d2.json`.
- Datasets: `data/libero-alphabet-soup-curvature-s64-d2-{train,eval}`.
- Norm stats: `openpi/assets/pi05_libero_alphabet_soup_curvature_s64_d2/<repo_id>/norm_stats.json`.
- Checkpoint: `openpi/checkpoints/pi05_libero_alphabet_soup_curvature_s64_d2/run/`.
- Results: `runs/libero_alphabet_soup/sim/s64_d2/`.

# Interface

## Interfaces

None changed.

## Inputs

`data/libero-alphabet-soup-timed`, `runs/libero_alphabet_soup/{robot_spec,keyframes}.json`, `pi05_base`.

## Outputs

The datasets, norm stats, checkpoint and results above; `runs/libero_alphabet_soup/train/*.log`.

## State Changes

None beyond the outputs.

# Flow

## Main Flow

1. Label config → `build_labels` train and eval → features block rewrite → `run.py norm-stats`.
2. `vla_libero.replay --source labels` for the 4 cells.
3. `run.py train` once two GPUs are free (`NCCL_P2P_DISABLE=1`, `PYTHONUNBUFFERED=1`, `WANDB_MODE=offline`).
4. Two `run.py serve-points` servers, one warm-up request each → `run_closed_loop.py` for the 4 cells →
   `gripper_trace.py` at rate 1, e 5 and full → `gripper_attempts.py`.

# Code Mapping

## Modules

No repository code changes; results in `docs/validation.md` and `issues.md`.

## Interfaces

None.

## State

As in State Ownership.

# Implementation

## Phase 1

Phase Status: completed

Round: 1

### Goal

S2 label datasets, norm stats, labels open loop in the simulator.

### Files

Outside the repository: label config, datasets, norm stats, `runs/libero_alphabet_soup/sim/s64_d2/label_replay/`.

### Structures

None.

### Affected Modules

Data stages.

### Dependencies

None.

### Observation

Episodes and frames, points inside keyframe windows and padded points against S, `action_chunk` shape;
action q01 / q99 against S; labels open-loop successes per cell.

Result: train 45 episodes / 8772 frames, eval 50; reach median 51.5 frames (S 64.0), points inside keyframe
windows 24.9% (S 16.1%), padded points 34.6% (S 38.5%); `action_chunk` (80,). After the features block
rewrite `datasets` 3.6 and 5.1 load the same values. Action q01 / q99 of the 7 joints −0.19…−0.73 /
+0.33…+1.20 (S −0.18…−0.77 / +0.35…+1.27). Labels open loop, kp 400: rate 1 e 5 45, full 45; velocity cap 4
e 4 46, full 48 (S's labels 45, 46, 44, 46); at rate 1 the first success comes a median 163 frames in (S 143).

## Phase 2

Phase Status: active

Round: 2

### Goal

Trained S2 checkpoint.

### Files

Outside the repository: checkpoint, `runs/libero_alphabet_soup/train/pi05_libero_alphabet_soup_curvature_s64_d2.log`.

### Structures

None.

### Affected Modules

None (training run).

### Dependencies

Phase 1; two free GPUs.

### Observation

Seconds per step, loss at steps 0, 1000, 5000, 9900; `arc_resample.json` in the final checkpoint's assets.

Run: GPUs 2 and 3, started 2026-10-19 14:06 PDT (`NCCL_P2P_DISABLE=1`), 1.7 s per step, 41.6 GB held per
GPU; loss 0.1458 at step 0, 0.1054 at 100.

## Phase 3

Phase Status: pending

Round: 3

### Goal

S2 closed loop and close-attempt classes; results recorded.

### Files

Outside the repository `runs/libero_alphabet_soup/sim/s64_d2/`; `docs/validation.md`, `issues.md`.

### Structures

None.

### Affected Modules

Knowledge.

### Dependencies

Phase 2.

### Observation

Per cell successes, failure phases, completion frames; per close-attempt class counts and distances, against
S and F.
