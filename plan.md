# Plan

Plan Status: approved

# Problem

## Current

- `data/pick-and-place-cube-quick-timed` (published as `JingwuLuo/pick_and_place_cube_quick_ur3`): 52 UR3
  episodes at 30 Hz, right arm only, still frames dropped outside 8 frames before to 15 after each gripper
  switch, 60-row end hold, five instructions per episode (first one as `task`). Built from the raw HDF5 by
  `examples/ur3_dual_arm/convert_hdf5.py`, `vla_tools.prompt_annotation` and `vla_tools.episode_timing` with
  the configuration in `runs/pick_place_cube_quick/`.
- On LIBERO (docs/validation.md → LIBERO simulator) three label families were trained with one recipe (pi0.5
  base, VLM frozen, per-point action statistics, 10k steps): per-frame (Fpp), curvature-resampled (S3b) and
  two-phase (TP); TP at cap 8 matched Fpp's success at about twice the demonstration speed.

## Problem

The three label families have not been built or trained on a real-robot recording.

## Goal

Three training datasets of the same 47 episodes (5 held out) that differ only in their labels, their
per-point normalisation statistics, and one training recipe for all three, ready for review before training:

- per-frame: `time`, span 10, H 10;
- resampled sparse: the sweep's selection at the largest admissible span;
- two-phase: `time`, span 48, `event_density` 4.8 (one point per recorded frame inside the windows), windows
  grasp −12 / +6, lift −3 / +9, release −15 / +9 frames (LIBERO's TP windows scaled from 20 to 30 Hz).

# Structure

## Modules

- Label tools (unchanged): `vla_tools.keyframe_detect`, `vla_tools.diversity`, `vla_tools.sweep`,
  `vla_tools.build_labels`.
- openpi integration: `examples/ur3_dual_arm/configs.py` (train config), `examples/ur3_dual_arm/run.py`
  (norm stats with `--per-point-actions`, train, serve).
- Run configuration outside the repository: `runs/pick_place_cube_quick/`.

## Responsibilities

The run directory owns every parameter; the tools turn them into datasets and statistics; the example owns
the training recipe.

## State Ownership

- Keyframe windows: `runs/pick_place_cube_quick/keyframes_spec_labels.json` (per-frame, resampled) and
  `keyframes_spec_two_phase.json`.
- Held-out episodes: `runs/pick_place_cube_quick/diversity.json`.
- Label parameters: `runs/pick_place_cube_quick/label_config_{time_s10,curvature_s48_d0,two_phase_s48}.json`.
- Joint velocity limits and switch holds: `runs/pick_place_cube_quick/robot_spec.json`.
- Training recipe: `examples/ur3_dual_arm/configs.py`.

# Interface

## Interfaces

`run.py norm-stats --per-point-actions` (new option, as in `examples/libero_object/run.py`); otherwise
unchanged.

## Inputs

`data/pick-and-place-cube-quick-timed`.

## Outputs

`data/pick-and-place-cube-quick-{time-s10,curvature-s48-d0,two-phase-s48}-{train,eval}`; norm stats under
`openpi/assets/pi05_ur3_cube_quick_{time_s10,curvature_s48,two_phase_s48}_pp/`; checkpoints after review.

## State Changes

`examples/ur3_dual_arm/configs.py` trains with the VLM frozen and `rotate_prompts`; `LeRobotArcDataConfig`
gains `rotate_prompts`; `examples/ur3_dual_arm/run.py` gains `--per-point-actions`; `vla_tools.norm_floor` floors per-point (H, D)
statistics along the dim axis (`floor_dims`), as well as pooled (D,) ones.

# Flow

## Main Flow

Keyframes on the timed dataset → stratified held-out split → sweep → `build_labels` (train without the
held-out, eval with all) → per-point norm stats → review of the recipe → training.

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

Keyframes, held-out split, sweep and the three label datasets.

### Files

`runs/pick_place_cube_quick/`, `data/pick-and-place-cube-quick-*-{train,eval}`.

### Structures

None.

### Affected Modules

None in the repository.

### Dependencies

None.

### Observation

Keyframe issue list; held-out episodes and their distance to training; sweep table; label positions of
sample frames; dataset summaries.

Audit of `data/pick-and-place-cube-quick` (`runs/pick_place_cube_quick/audit.json`): largest
|action[t] − state[t+1]| 0, largest timestamp gap 1.0 frame, commands binary, static dims 0–5 (left arm), 44
continuing jumps all on dim 9 (right elbow, 8–12 median steps; review images show smooth motion), no isolated
or breakpoint jump: no cleaning step.

Result: grasp, lift and release found in all 52 episodes (lift a median 6 frames after the grasp; three long
carries accepted with reasons). Held-out (stratified) episodes 2, 13, 28, 34, 43 (raw 0003, 0015, 0033, 0039,
0048; 34 and 43 are multi-move combinations absent from training). Sweep (head 5, path step 0.0212 rad):
`time` admissible up to span 32, `curvature` up to span 48 at density 0 (density 1: keyframe position median
1.07 > 1); selected `curvature` span 48 density 0 (path full 0.54, keyframe position 0.91). Two-phase cell
`time` span 48 density 4.8: admissible, path full 0.31, keyframe position 0.34. Train copies 47 episodes /
15,876 rows, eval copies 52 / 17,564; reach median 10 / 48 / 29 frames, window fraction 17.4 / 15.9 / 38.5%,
mean loss weight 1.00. Episode 0: per-frame points at frames +1…+10; curvature at uneven spacings up to +48;
two-phase every 4.8 frames away from the events and every frame next to the grasp and the release.

## Phase 2

Phase Status: completed

Round: 2

### Goal

Per-point norm stats for the three train copies.

### Files

`examples/ur3_dual_arm/run.py`, `openpi/assets/pi05_ur3_cube_quick_*_pp/`.

### Structures

None.

### Affected Modules

openpi integration example.

### Dependencies

Phase 1.

### Observation

Statistics shapes (actions (10, 14), state (14,)); constant left-arm dims.

Result: actions (10, 14), state (14,) for all three. The left-arm joints move by at most 4.1e-4 rad (encoder
noise): delta quantile ranges 1.0e-4–2.2e-4 rad, floored to 0.05 rad on dims 0–5 at every point
(`norm_floor`, raw files kept as `runs/pick_place_cube_quick/norm_stats_*.raw.json`). The left gripper command
is 1 in episodes 2–6 and 0 elsewhere, constant within each episode. `arc_resample.json` copied next to each
norm_stats.json. `tests/test_norm_floor.py` covers pooled and per-point statistics.

## Phase 3

Phase Status: completed

Round: 3

### Goal

Training recipe: VLM frozen, each sample's prompt drawn from its episode's five instructions; check of the
label parameters against the recording and of what the model receives.

### Files

`src/vla_openpi/episode_prompt.py` (EpisodePrompt, load_episode_prompts), `src/vla_openpi/vlm_freeze.py`
(vlm_filter, moved from `examples/libero_object/configs.py`), `src/vla_openpi/arc_data_config.py`
(`rotate_prompts`), `src/vla_openpi/__init__.py`, `examples/ur3_dual_arm/configs.py`,
`examples/libero_object/configs.py`, `tests/test_openpi_integration.py`.

### Structures

`EpisodePrompt(dataset_root)`: in the repack stage (training only), replaces `episode_index` with one of the
episode's `tasks`, drawn uniformly per call; the file is read on the first sample, so serving builds the
config without the dataset.

### Affected Modules

openpi integration.

### Dependencies

Phase 2.

### Observation

Fingertip distance and joint speed around the events (raw end-effector poses); sweep and rate ceiling of
the two-phase windows; samples through openpi's training transform chain; test suite.

Result: before a grasp the operator runs at 0.54–0.67 of the median joint speed from about 18 to 3 frames
before; the fingertip is within 20 / 40 / 60 mm of the grasp pose a median 5 / 15 / 27 frames before (p90 10 /
29 / 30), of the release pose 8 / 17 / 24; joint speed passes 1.3 of the median 9 frames after a grasp or a
release. The two-phase grasp window (−12) starts 33 mm before the grasp, inside the slow approach; windows
covering 40 mm (grasp −15 / +6, release −17 / +12) or 60 mm (−27 / +6, −24 / +14) cover 33.1% and 43.9% of
recorded rows against 28.8%, with path full 0.29 and 0.20 against 0.31 path steps, the same keyframe position
(0.34), and lower rate ceilings (async at 0.1 s: 1.96 and 1.81 against 2.19). Rate ceiling (velocity timing at
1.0 of the UR3 limits, holds 0.5 / 0.517 s, 90% of 50 episodes; episodes 30 and 38 recorded above the limit
are left out), async / sync speed-up at latency 0, 0.1, 0.2 s: per-frame 2.74, 1.30 / 1.17, 0.74 / 0.84;
curvature span 48 3.02, 2.75 / 2.26, 2.13 / 1.83; two-phase 2.59, 2.19 / 1.68, 1.39 / 1.29. Training
transform chain for all three: repack (RepackTransform, EpisodePrompt) → ArcInputs → DeltaActions → Normalize
→ ResizeImages, TokenizePrompt, PadStatesAndActions; joint actions equal chunk − state exactly and grippers
stay absolute; normalised right-joint actions 1.8–2.2% outside [−1, 1], floored left joints at most 0.009;
three 224 × 224 images with masks; 200 draws of one sample give each of its five instructions 29–46 times.
pi0.5 with `discrete_state_input=False` adds no state token and no state text: the model sees images and the
prompt only. 166 tests pass.

## Phase 4

Phase Status: completed

Round: 4

### Goal

Launch the three trainings with the reviewed recipe.

### Files

`openpi/checkpoints/pi05_ur3_cube_quick_*_pp/`, `runs/pick_place_cube_quick/train/`.

### Structures

None.

### Affected Modules

None in the repository.

### Dependencies

Phase 3.

### Observation

Each run past its first steps; step rate; GPU memory.

Started on one 8 × L40S node of the cluster (GPUs 0,1 / 2,3 / 5,6), with the base weights read over NFS from
another node's cache (`OPENPI_DATA_HOME`). The first launch stalled at the first collective
(NCCL peer-to-peer between the PCIe GPUs); relaunched with
`NCCL_P2P_DISABLE=1`: 17 GB per GPU, 1.6 s per step (per-frame, curvature) and 2.0 s per step (two-phase) at
step 200–280.

## Phase 5

Phase Status: completed

Round: 5

### Goal

Per-episode instructions written into a dataset version by a tool rather than by a converter.

### Files

`src/vla_tools/dataset_writer.py` (`EpisodePlan.tasks`), `src/vla_tools/prompt_annotation.py`,
`tests/test_prompt_annotation.py`.

### Structures

`EpisodePlan.tasks: list[str] | None`: the episode's instructions; when any plan sets them, the writer
rewrites `meta/tasks.jsonl`, the episode rows' `tasks` in `meta/episodes.jsonl` and `task_index` (the first
instruction). `prompt_annotation --dataset --prompts prompts.json --out`: `prompts.json` keyed by episode
index of that version, every episode covered.

### Affected Modules

Dataset writing.

### Dependencies

Phase 4.

### Observation

Test on a synthetic dataset; the tool on `data/pick-and-place-cube-quick` reproduces its tasks, episodes and
`task_index` from `runs/pick_place_cube_quick/prompts.json`.

Result: `tests/test_prompt_annotation.py` (task tables, task_index, refusal of an uncovered episode, source-map
renumbering with an excluded source episode) and `tests/test_dataset_writer.py` pass. On the converted
dataset with `--source-map meta/conversion_report.json`: 52 episodes, 260 instructions, 222 distinct;
tasks.jsonl, episodes.jsonl and every episode's task_index identical to the converter-written ones.

## Phase 6

Phase Status: completed

Round: 6

### Goal

Per-episode contact sheets from a LeRobot dataset for annotation and review.

### Files

`src/vla_tools/episode_sheets.py`, `tests/test_episode_sheets.py`.

### Structures

`episode_sheets --dataset --spec --cameras ... --out`: one PNG per episode, a column per moment (first frame,
`--after` frames after every command edge, last frame), a row per camera.

### Affected Modules

None besides the new tool.

### Dependencies

Phase 5.

### Observation

Test on a synthetic dataset (moments and image size); sheets of `data/pick-and-place-cube-quick` against the
raw-HDF5 sheets used for the annotation.

Result: `tests/test_episode_sheets.py` passes. Episode 3 (raw 0004): start, close at action 107 (frame 113),
open 178, close 275, open 433, end 523 — the moments and images of the raw sheet.

## Phase 7

Phase Status: completed

Round: 7

### Goal

The UR3 recorder's HDF5 converter in the repository as the example of a raw-recording intake.

### Files

`examples/ur3_dual_arm/convert_hdf5.py`, `pyproject.toml` (optional dependency `hdf5 = ["h5py"]`).

### Structures

The converter first written outside the repository, without the instructions (a single task
`--task` until `prompt_annotation`), plus the raw step-interval check in its report.

### Affected Modules

UR3 example.

### Dependencies

Phase 5.

### Observation

Converter + `prompt_annotation` against `data/pick-and-place-cube-quick`: equal rows, images and meta.

Result: every parquet column of all 52 episodes equal; info.json, tasks.jsonl, episodes.jsonl and
conversion_report.json identical; episodes_stats.jsonl differs only in task_index statistics (not recomputed
by the writer) and the version adds prompt_annotation.json. Without the exclusion of 0002 the converter stops
there: largest step interval 6.33 s exceeds 1.5 frames. The converter outside the repository is removed.

## Phase 8

Phase Status: completed

Round: 8

### Goal

Skills and references describe the intake, annotation, still-frame and publishing steps.

### Files

`.claude/skills/vla-data-intake/SKILL.md` (new), `.claude/skills/vla-pipeline/SKILL.md`,
`.claude/skills/vla-pipeline/references/{artifacts,failure-modes}.md`, `.claude/skills/vla-episode-timing/SKILL.md`,
`.claude/skills/vla-downstream/SKILL.md`, `README.md`.

### Structures

None.

### Affected Modules

Skills.

### Dependencies

Phases 5–7.

### Observation

Every command in the skills runs as written on this dataset; jargon check of the new skill.

Result: `vla-data-intake` written; S0 in `vla-pipeline` points to it, with its two decisions; references
gain conversion.json, conversion_report.json and prompts.json and six failure modes; `vla-episode-timing`
and `vla-downstream` gain the UR3 results and the training and publishing steps. The converter,
`episode_sheets --every 40` and `prompt_annotation` commands ran as written on this dataset.
`episode_sheets` gained `--every`: the person in raw 0020 was visible between the edges and the end only.

## Phase 9

Phase Status: completed

Round: 9

### Goal

Tests, lint and a local commit.

### Files

All of the above.

### Structures

None.

### Affected Modules

None.

### Dependencies

Phase 8.

### Observation

Full test suite and ruff; `git log` shows the commit under the repository identity.

Result: 170 tests pass; ruff clean outside the template-owned threejs-video script. Commits 97139ce
(training recipe) and 346c004 (intake) under Jingwu Luo; not pushed.

## Phase 10

Phase Status: active

Round: 10

### Goal

Training results.

### Files

`openpi/checkpoints/pi05_ur3_cube_quick_*_pp/`, `runs/pick_place_cube_quick/train/`.

### Structures

None.

### Affected Modules

None in the repository.

### Dependencies

Phase 4.

### Observation

Loss at steps 0, 100 and the end per version; checkpoint with `arc_resample.json`; open-loop error on the
held-out episodes.

All three runs ended together at 23:08 (node time, 2026-10-26) at steps 4,320 / 4,300 / 3,140, with no error in
their logs, no reboot and no out-of-memory record, while another session on the shared account closed the same
second. Resumed at 06:15 on 2026-10-27 from checkpoints 4000 / 4000 / 3000 (`run.py train --resume`).

The source and timed datasets were then rebuilt for cube tasks only (raw 0000, 0004, 0005, 0006 excluded:
spoon, mug and black block): 48 episodes, 13,186 recorded rows, 202 distinct instructions, and the Hub
dataset replaced. The running trainings and their six label datasets keep the 52-episode version (47 training
episodes, 4 of them not cube tasks); the next round rebuilds keyframes, the held-out split and labels from the
48-episode version.

The three runs finished (step 9999), but every checkpoint, from 2000 on, has non-finite values in 19 trained
arrays, and the resumed runs logged NaN loss from their first line. Probes on the same node: two GPUs with only
NCCL peer-to-peer disabled give NaN loss and parameter norm at step 0; one GPU (batch 32) and two GPUs with
shared memory also disabled give loss 0.152–0.153 at step 0 and 0.053–0.054 at step 110. The models are unusable;
the runs have to be repeated.

Repeated on the 48-episode cube-only version (`c48` label datasets: held-out 10, 18, 23, 25, 35; 43 training
episodes): two GPUs per run with peer-to-peer and shared memory disabled still gave NaN or inf loss at step 0;
one GPU per run starts at loss 0.160 / 0.184 / 0.155 (per-frame / curvature / two-phase), batch 32, keep period
5000 so the half-way checkpoint stays. Those runs were stopped for a version with both recording days.

Second recording day (raw 0058-0128, 71 episodes) added: instructions annotated from the sheets (raw HDF5 sheets
with frames every 40 steps), the per-move content kept in `runs/pick_place_cube_quick/moves.json`. Excluded in
`runs/pick_place_cube_quick_all/conversion.json` (22 of 129): the earlier 10, raw 0125 (a person resets the
scene after the task; recording left running) and 0126 (empty), raw 0007 and 0127 (teleoperated resets recorded
as episodes, undone by the next episode), and eight second-day episodes that start with a teleoperated reset of
the target (66, 71, 73, 78, 79, 86, 87, 89). Seven episodes with a missed grasp and a successful re-grasp are
kept (about 1.8% of frames). 107 episodes, 25,490 recorded rows; keyframe issues accepted with reasons (long
carries, slow approaches, multi-move phases, re-grasps); held-out 5, 17, 21, 37, 47, 49, 64, 85, 97, 98 (97
training episodes). Published as `JingwuLuo/pick_and_place_cube_quick_ur3` (commit aad6c38e, tag `v2.1`).
Trainings `pi05_ur3_cube_quick_all_*_pp` started on one GPU each: loss 0.151 / 0.171 / 0.178 at step 0 and
0.126 / 0.149 / 0.128 at step 50, 3.0 s per step.

Real-robot test of the 174-episode curvature policy at 10,000 steps: good overall, grasp height varies from trial
to trial (sometimes the fingertip overshoots into the table); the 5,000-step checkpoint is clearly weaker.
Demonstration grasps from the table are consistent (fingertip z -5 mm median, 8-10 mm standard deviation over
three days; lowest 10% at -15 to -18 mm), so the margin to the table is about the size of that spread; the third
recording day reaches farther (0.195 m against 0.10-0.125 m) and moves faster (0.71 against 0.49-0.52 rad/s).
Normalisation ranges changed by about 6% between the 174- and 228-episode data. Training config changed to
20,000 steps (11 passes over 206 episodes; 10,000 steps would be 5.5 against 7.3 before) and an exponential moving
average of the weights, 0.999 (the saved `params` are the averaged weights); representation (joint deltas, no
state input) unchanged. Started `pi05_ur3_cube_quick_v4_curvature_s48_pp`, exp `ema20k`, on one GPU: loss 0.190 at
step 0, 0.145 at step 50, 3.1 s per step. The three 228-episode runs of the old config were stopped at about 1,000
steps and their checkpoints deleted. Open: absolute joint output, state input and a larger grasp-window loss
weight as one-at-a-time variants; a fingertip-height floor at deployment.
