# Plan

Plan Status: completed

# Problem

## Current

- `LGG100/optical-bench-lerobot-cleaned` (Hugging Face, public): dual-arm UR3, 14 dims (per arm 6 joints +
  binary gripper), 15 Hz, 40 episodes, 9588 frames (176–388 per episode), task "pull the metal rod out of the
  post holder and place it in the white box"; three cameras (cam_high, left and right wrist) stored as AV1
  videos, 240 × 320. The left arm and its gripper never move; the right arm does the task with one close and
  one open per episode.
- `dataset_writer` refuses row changes on video datasets (end padding repeats rows). The videos were decoded
  once into a parquet-image copy, `FTpi05/data/optical-bench-images` (every stored PNG equal to its decoded
  frame, state, action and indices equal to the source); the pipeline runs on that copy.
- S0–S7 have run on it (`FTpi05/runs/optical_bench/`) with the agent's proposals for the person's decisions
  D-001–D-005 (jump at 35:80 kept; every still frame kept, 45 padding frames; 35 keyframe issues accepted;
  holdout [8, 15, 24, 27]; candidate `curvature` s30 d2). The sweep admits all 30 cells; the label replay at
  rate 1, e 5 gives path median / p90 / max 0.01 / 0.11 / 3.77 path steps (0.0347 rad) for s30 d2 against
  0.03 / 0.21 / 5.39 for s40 d2.
- `vla_tools.replay_eval` does not import: it calls `label_replay.matched_shifts`, removed when the label
  replay was rewritten (ISSUE-013).
- No example config trains this robot with the VLM frozen; `examples/ur3_dual_arm` trains LoRA on both
  experts.

## Problem

No pi0.5 policy is trained on resampled labels of this dataset, and the held-out evaluation tool is broken.

## Goal

- Label datasets `curvature` span 30, event density 2, H 10, key and pad weight 1, anchor state:
  train (36 episodes, holdout excluded) and eval (40), loadable by openpi; S0–S7 passed with D-001–D-005
  decided; `data_report.md`.
- One pi0.5 policy from `pi05_base`, VLM frozen (SigLIP, Gemma 2B), action expert and projections trained;
  batch 32, 10k steps, cosine learning rate (500 warm-up, peak 2.5e-5, end 2.5e-6), AdamW, quantile
  normalisation with a minimum range of 0.05 rad on the static left-arm action dims; two L40S, data
  parallel (16 samples per GPU).
- Open-loop evaluation of the checkpoint on the 4 held-out episodes (and 4 training episodes for contrast)
  with `replay_eval`: points, head, stitched, edges, static, hold, at e 8, rate 1.

# Structure

## Modules

- Data stages, existing tools, outside the repository: `FTpi05/runs/optical_bench/` (plans, decisions,
  reports), `FTpi05/data/optical-bench-images*` (versions), `FTpi05/reports/optical-bench-src/video_to_images.py`.
- Training example for this robot with the VLM frozen: `examples/ur3_optical_bench/configs.py`, `run.py`.
- Open-loop evaluation: `src/vla_tools/replay_eval.py` (edge shifts computed with `arc_core.command_edges`).

## Responsibilities

`configs.py` builds the TrainConfig; `run.py` runs norm stats, training and serving through
`vla_openpi.launch`. `replay_eval` sends recorded observations to a served policy and scores its targets.

## State Ownership

- Person's decisions: `runs/optical_bench/decisions.md`; carried into `cleaning_plan.json`,
  `timing_plan.json`, `keyframe_overrides.json`, `diversity.json`, the label config.
- Norm stats: `openpi/assets/<config>/<repo_id>/norm_stats.json` (floored copy replaces it; report kept in
  `runs/optical_bench/norm_floor.json`).
- Checkpoints: `openpi/checkpoints/<config>/run/<step>/`, with `arc_resample.json` in the assets.

# Interface

## Interfaces

- `examples/ur3_optical_bench/configs.py`: `train_config(name, data_root, repo_id, exp_name) -> TrainConfig`
  (14 dims; cam_high → base_0_rgb, left wrist → left_wrist_0_rgb, right wrist → right_wrist_0_rgb; joints
  and the idle left gripper (0 in 11 recordings, 1 in 29, constant within each) as deltas, the right gripper
  absolute; freeze filter: SigLIP and every Gemma 2B parameter); `run.py norm-stats |
  train | serve | serve-points`.
- `replay_eval.edge_shifts(stitched: np.ndarray, recorded: np.ndarray, command_dims: Sequence[int]) ->
  tuple[list[int], int]`: signed frame shift of each recorded command edge to the nearest stitched switch of
  the same direction within half the distance to the neighbouring edges, and the count never switched.

## Inputs

`FTpi05/data/optical-bench-images-timed`, `runs/optical_bench/{robot_spec,keyframes,diversity}.json`,
`label_config_curvature_s30_d2.json`; `pi05_base` (cached).

## Outputs

`data/optical-bench-images-curvature-s30-d2-{train,eval}`, `runs/optical_bench/data_report.md`, the
checkpoint, `runs/optical_bench/train/*.log`, `runs/optical_bench/replay_eval_{holdout,train}.json`.

## State Changes

None beyond the outputs.

# Flow

## Main Flow

1. Decisions recorded → `build_labels` (train with `--exclude 8 15 24 27`, eval) → features block rewritten
   to `Sequence` (ISSUE-009) → `run_index` → `run_status` → `data_report`.
2. `run.py norm-stats` → `norm_floor --key actions --dims 0 1 2 3 4 5 6 --min-range 0.05`.
3. `run.py train` on two free L40S (`PYTHONUNBUFFERED=1`, `WANDB_MODE=offline`) → `arc_resample.json` into
   the final checkpoint's assets.
4. `run.py serve` (per-frame targets and points, rate 1), one warm-up request → `replay_eval` on episodes
   8, 15, 24, 27 and 0, 10, 20, 30 at e 8.

# Code Mapping

## Modules

Example → `examples/ur3_optical_bench/{configs,run}.py`; evaluation → `src/vla_tools/replay_eval.py`.

## Interfaces

`train_config`, `vlm_filter` → `configs.py`; `edge_shifts` → `replay_eval.py`.

## State

Decisions → `runs/optical_bench/decisions.md`; norm stats → `openpi/assets/`; checkpoints →
`openpi/checkpoints/`.

# Implementation

## Phase 1

Phase Status: completed

Round: 1

### Goal

Label datasets, S0–S7 passed, data report.

### Files

Outside the repository: `runs/optical_bench/` (decisions, plans), `data/optical-bench-images-curvature-s30-d2-*`.

### Structures

None.

### Affected Modules

Data stages.

### Dependencies

The person's decisions D-001–D-005.

### Observation

`run_status` S0–S7 passed; train / eval episodes and frames, `action_chunk` (140) and weight (10) shapes;
both `datasets` versions load identical values.

Result: D-001–D-005 decided as proposed. Train 36 episodes / 10191 rows (1620 padding), eval 40; reach median
30.0 frames, padded points 17.6%, window points 16.6%, mean weight 1.00; `action_chunk` (140,), weight (10,).
After the `Sequence` rewrite `datasets` 3.6 and 5.1 load 10191 rows with the same `action_chunk` sum
(−237543.72100355). `run_status` S0–S7 passed; `data_report.md` written. UR3 velocity limits added to the
robot spec; rate ceiling at L 0.1 s async (90% of 37 episodes; 3 recordings beyond the limits left out):
uniform timing 0.89× (rate 1, e 14); velocity timing 4.50× without holds (cap 6, e 2), 3.80× with
0.533 / 0.667 s holds (cap 8, e 2).

## Phase 2

Phase Status: completed

Round: 1

### Goal

Training example, norm stats.

### Files

`examples/ur3_optical_bench/configs.py`, `examples/ur3_optical_bench/run.py`; outside the repository the norm
stats and `runs/optical_bench/norm_floor.json`.

### Structures

`train_config`, `vlm_filter`.

### Affected Modules

Training example.

### Dependencies

Phase 1.

### Observation

Trainable and frozen parameter counts; action q01 / q99 per dim before and after the floor.

Result: `examples/ur3_optical_bench/{configs,run}.py`. The left gripper is 0 in 11 recordings and 1 in 29,
constant within each; with no state input an absolute target would have to be guessed from the images, so it
is a delta like the joints (`make_bool_mask(13, -1)`); the right gripper stays absolute. Norm stats over the
10191 training rows: action q99 − q01 1.0e-4 to 1.5e-4 rad on the left joints and 0 on the left gripper,
0.53–2.76 rad on the right joints, 0.9998 on the right gripper; `norm_floor` widens dims 0–6 to ±0.025
(std 0.0125). Raw and floored stats in `runs/optical_bench/norm_stats.{raw,floored}.json`; the floored file
is the one in the assets.

## Phase 3

Phase Status: completed

Round: 2

### Goal

Trained checkpoint.

### Files

Outside the repository: checkpoint, `runs/optical_bench/train/*.log`.

### Structures

None.

### Affected Modules

None (training run).

### Dependencies

Phase 2; two free GPUs.

### Observation

Seconds per step, GPU memory, loss at steps 0, 100, 1000, 5000, 9900; `arc_resample.json` in the assets.

Run: GPUs 2 and 3, started 2026-10-18 14:47 PDT with `NCCL_P2P_DISABLE=1` (a first start without it hung at
the first collective for 45 min and was stopped); 1.7 s per step, 41.6 GB held per GPU at
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`; log `runs/optical_bench/train/`.

Result: 10k steps finished 2026-10-18 19:48 PDT (5 h 01 min); loss 0.2236 / 0.1293 / 0.0159 / 0.0061 /
0.0039 at steps 0 / 100 / 1000 / 5000 / 9900 (grad norm 1.18 → 0.08). Checkpoints 5000 and 9999;
`arc_resample.json` copied into `9999/assets/optical-bench-images-curvature-s30-d2-train/` (byte-identical to
the dataset's).

## Phase 4

Phase Status: completed

Round: 1

### Goal

`replay_eval` imports and scores command edges again.

### Files

`src/vla_tools/replay_eval.py`, `tests/test_replay_eval.py`.

### Structures

`edge_shifts`.

### Affected Modules

Open-loop evaluation.

### Dependencies

None.

### Observation

Tests on constructed command sequences: an edge shifted by +2 and −1 frames, a missed edge, two edges of one
dim; the module imports; full suite.

Result: `edge_shifts` (nearest switch of the same direction within half the distance to the neighbouring
edges); 4 tests in `tests/test_replay_eval.py` pass where `openpi_client` is installed (simulator
environment) and are skipped elsewhere; the module imports; full suite 147 passed, 2 skipped.

## Phase 5

Phase Status: completed

Round: 3

### Goal

Open-loop results on held-out and training episodes; results recorded.

### Files

Outside the repository `runs/optical_bench/replay_eval_*.json`, `runs/optical_bench/replay_spec.json`;
`docs/validation.md`, `issues.md`.

### Structures

None.

### Affected Modules

Knowledge.

### Dependencies

Phases 3 and 4.

### Observation

Per episode group: points, head, stitched and hold errors (path steps), edge shifts (frames) and misses,
static-dim moves, round-trip time.

Result: server on one L40S (first request 15 s, then 98–101 ms round trip); held-out / training stitched
median 0.077 / 0.057 rad, points 0.340 / 0.120 rad, close shifts +8, +5, +5, +5 / +2, +1, +7, +3 frames,
3 of 8 opens not scored inside the recorded rows (ISSUE-014, ISSUE-015); static left arm at most 1.2e-3 rad.
Results in `docs/validation.md` → UR3 optical bench; ISSUE-013 resolved.
