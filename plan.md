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
- `data/ur3_sim_pick_and_place_cube` (Hub `JingwuLuo/ur3_sim_pick_and_place_cube`, revision 7fa3c586): 500
  successful scripted MuJoCo episodes of the same workcell, 81,087 steps at 30 Hz, in the raw recordings' HDF5
  layout plus a `sim/` ground-truth group; both wrist cameras sit on the table side of the grippers, as on the
  real rig. In each episode one arm moves (left 240, right 260) and the other stays at its start pose; 83
  episodes close the gripper late on air, reopen and re-grasp (the arm lifts before the close). No language
  annotation (`sim/task` names the arm, cube, source and target). The motions are faster and smoother than the
  teleoperated ones (peak flange speed per episode: median 0.34 m/s, against 0.13 m/s median over all steps of
  the real recordings).
- `data/ur3_sim_pick_and_place_cube_5cd8956` (Hub `JingwuLuo/ur3_sim_pick_and_place_cube`, revision 5cd89567): 1000
  successful scripted episodes, 268,700 steps at 30 Hz, same HDF5 layout. 612 episodes are relays (the picking arm
  hands the cube over on a spot in the middle of the table, the other arm places it), 388 are single-arm; 252
  episodes are perturbed in one leg (an arm is pushed 1-5 cm off its path during the approach or the transport,
  6-15 steps, 2,627 steps in all, tagged `command/<arm>/source` = "perturbation"); 236 close late on air and
  re-grasp; pick-to-place distances 0.08-0.82 m spread evenly; five overhead light shapes. No language annotation,
  no held-out split. wrist_2 moves 0.0006 rad per frame (q01-q99), the gripper reaches 90 % of its travel 0.30 s
  after a close command and 0.50 s after an open command.

## Problem

The simulation-only and the adapted checkpoints of revision 7fa3c586 were trained on 500 single-arm episodes; the
dataset now holds relays, perturbed legs and a wider range of distances, so they are replaced. The perturbed steps
show the arm pushed off its path: their actions must not be learned, while the observations after a push, with the
actions that recover from it, must be.

## Goal

For each label family (per-frame, curvature with `event_density` 0, two-phase), on revision 5cd89567:

- three training datasets of 960 episodes and three evaluation datasets of all 1000, differing only in their labels;
  40 episodes are held out, chosen by stratification; the points of a chunk that lie on a pushed step have loss weight
  0 and every other point has weight 1;
- a simulation-only checkpoint (recipe `sim_15k`: 15,000 steps, peak learning rate 5e-5, EMA 0.999, VLM frozen),
  kept at steps 5,000, 10,000 and the latest;
- a checkpoint that adapts it to the real recordings (recipe `lora_4k_ema999`: 4,000 steps, peak learning rate 1e-4,
  EMA 0.999), kept at step 2,000 and the latest, whose saved `params` reproduce the trained weights;
- open-loop errors of the stage-1 and stage-2 checkpoints on the 40 simulated and the 5 real held-out episodes, next
  to the real-only run.

# Structure

## Modules

- Label tools (unchanged): `vla_tools.keyframe_detect`, `vla_tools.diversity`, `vla_tools.sweep`,
  `vla_tools.build_labels`.
- openpi integration: `examples/ur3_dual_arm/configs.py` (train config), `examples/ur3_dual_arm/recipes.py`
  (steps, learning rate, warm-up, EMA decay and kept checkpoints per recipe), `examples/ur3_dual_arm/run.py`
  (norm stats with `--per-point-actions`, train, serve).
- Run configuration outside the repository: `runs/pick_place_cube_quick/`.
- Simulated data: the same tools on `data/ur3_sim_pick_and_place_cube`, configured in
  `runs/ur3_sim_pick_and_place_cube/`; `train_config` gains an initial-weights option and a LoRA option for the
  second stage (`vla_openpi.non_lora_filter`, `vla_openpi.ZeroLoraBLoader`).

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
- Simulated-data parameters (conversion, instructions, timing, keyframes, held-out split, label parameters):
  `runs/ur3_sim_pick_and_place_cube/`. The label parameters are copies of the real run's
  `label_config_*.json`, so both stages share them.

# Interface

## Interfaces

`run.py norm-stats --per-point-actions` (new option, as in `examples/libero_object/run.py`); otherwise
unchanged. `run.py train --init-checkpoint <dir>` starts from the saved `params` of an earlier run instead of
`pi05_base`; with `--lora` it trains only LoRA adapters of the action expert on top of them, starting as a no-op.
`run.py train --recipe <name>` selects a recipe of `recipes.py` (`full_20k`, `sim_10k`, `lora_10k`, `lora_4k`);
without it `full_20k` is used, or `lora_10k` with `--lora`.

## Inputs

`data/pick-and-place-cube-quick-timed`; `data/ur3_sim_pick_and_place_cube`;
`data/pick-and-place-cube-quick-v4-timed` with the real held-out episodes 54, 102, 122, 180, 185.

## Outputs

`data/pick-and-place-cube-quick-{time-s10,curvature-s48-d0,two-phase-s48}-{train,eval}`; norm stats under
`openpi/assets/pi05_ur3_cube_quick_{time_s10,curvature_s48,two_phase_s48}_pp/`; checkpoints after review.

Simulated: `data/ur3-sim-cube-{time-s10,curvature-s48-d0,two-phase-s48}-{train,eval}`; real second-stage
datasets `data/pick-and-place-cube-quick-v4h5-{time-s10,two-phase-s48}-{train,eval}` (the curvature one exists);
checkpoints `openpi/checkpoints/pi05_ur3_sim_<label>_pp/` and `pi05_ur3_sim_real_<label>_pp/`; the second stage's
`norm_stats.json` is a copy of the first stage's, which covers the simulated and the real recordings.

## State Changes

`examples/ur3_dual_arm/configs.py` trains with the VLM frozen and `rotate_prompts`; `LeRobotArcDataConfig`
gains `rotate_prompts`; `examples/ur3_dual_arm/run.py` gains `--per-point-actions`; `vla_tools.norm_floor` floors per-point (H, D)
statistics along the dim axis (`floor_dims`), as well as pooled (D,) ones.

# Flow

## Main Flow

Keyframes on the timed dataset → stratified held-out split → sweep → `build_labels` (train without the
held-out, eval with all) → per-point norm stats → review of the recipe → training.

Simulated pre-training: convert → instructions from `sim/task` → audit → timing and keyframes (checked against
`sim/motion` grasp and release steps) → stratified held-out split → three label datasets → per-point norm stats →
stage 1 (three runs on the simulated data, one GPU each) → stage 2 (three LoRA runs on the real 223 episodes from
the stage-1 checkpoints).

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

Phase Status: completed

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

Held-out fixed at 5 episodes (54, 102, 122, 180, 185 of the 228): training set 223 episodes, 62,410 rows, 10.3 passes
at 20,000 steps. Run `pi05_ur3_cube_quick_v4h5_curvature_s48_pp`, exp `ema20k`, one GPU: loss 0.178 at step 0,
0.144 at step 50, 3.1 s per step. The 206-episode run was stopped at about 10 minutes and its checkpoints deleted.

Two one-variable variants of the 223-episode curvature run (same data, labels, 20,000 steps, EMA; `train_config`
options `absolute_joints` and `state_input`, `run.py --absolute-joints --state-input`): joints as absolute positions
(no delta transform; actions normalised per chunk point, delta range 0.36 rad becomes an absolute range of 2.6 rad),
and the normalised state as discrete prompt tokens (left-arm joint ranges in the state statistics floored to 0.05
rad: their noise range of 1e-4 rad would otherwise be stretched to the full token range). Runs
`pi05_ur3_cube_quick_v4h5_curvature_s48_{abs,state}_pp`, exp `ema20k`, one L40S each on nnmc60 GPUs 2 and 5, 3.1-3.3 s
per step; loss at step 100 0.093 (absolute) and 0.079 (state); losses are not comparable across representations.

All three runs finished at step 19,999 with finite parameters (51 arrays checked per run); checkpoints at 10,000
and 19,999 with `arc_resample.json` in their assets are on the Hub model repo under `223TrialsTrainData/`.

## Phase 11

Phase Status: completed

Round: 11

### Goal

Simulated episodes converted, annotated and audited: `data/ur3-sim-cube`.

### Files

`runs/ur3_sim_pick_and_place_cube/{conversion.json,prompts.json,make_prompts.py,audit.json}`, `data/ur3-sim-cube`,
`data/ur3-sim-cube-prompted`.

### Structures

None.

### Affected Modules

None in the repository; `examples/ur3_dual_arm/convert_hdf5.py` and `vla_tools.prompt_annotation` run unchanged.

### Dependencies

None.

### Observation

500 episodes and row count from `meta/conversion_report.json`; every episode has five instructions; the audit
reports non-finite values, timestamp gaps and joint jumps per episode.

500 episodes, 81,087 rows (`meta/conversion_report.json`); 2,500 instructions, 211 distinct
(`runs/ur3_sim_pick_and_place_cube/make_prompts.py`). Audit: the only findings are 492 frozen wrist-camera runs,
all on the arm that stays still, and 87 episodes with short still ends.

## Phase 12

Phase Status: completed

Round: 12

### Goal

Static frames, end padding, keyframes and held-out split for the simulated data.

### Files

`runs/ur3_sim_pick_and_place_cube/{robot_spec.json,timing_plan.json,keyframes_spec*.json,keyframes*.json,keyframe_overrides*.json,diversity.json}`,
`data/ur3-sim-cube-timed`.

### Structures

Keyframe types per arm (the moving arm differs between episodes).

### Affected Modules

None in the repository.

### Dependencies

Phase 11.

### Observation

Detected grasp and release frames against the simulator's recorded gripper commands; row counts before and after
timing; keyframe counts per type, with 2 grasps in the 83 late-close episodes.

The number of detected grasps and releases equals the number of gripper commands the simulator recorded in all 500
episodes (`make_keyframe_overrides.py`). The real run's `lift` rule (first motion above 0.015 rad per frame after
the grasp) found the lift one frame after the grasp because the arm is still decelerating at the close command; a
`rest` type (below 0.003 rad per frame for 3 frames) before `lift` fixes it (`make_keyframe_specs.py`); `lift` is
missing in 8 episodes whose transport stays below 0.015 rad per frame. In a late-close episode the first grasp
closes on air while the arm hovers, so no lift follows it and the first lift lies after the first release. Outlier
and order issues (49 before timing, 352 after) are accepted with reasons from the generator parameters; 82 of the 83
late-close episodes carry an order issue. Timing: 81,087 rows in, 6,475 still rows dropped, 30,000 padding rows,
104,612 rows out. Held-out split (stratified, 10): 2, 100, 231, 234, 243, 287, 323, 325, 349, 463 (7 right arm, 3
left; 5 short, 4 medium, 1 long; 2 late close).

## Phase 13

Phase Status: completed

Round: 13

### Goal

Label datasets and per-point statistics: three simulated families, and the two missing real families on the
223-episode split.

### Files

`runs/ur3_sim_pick_and_place_cube/{label_config_*.json,merge_norm_stats.py,build_sim_labels.sh}`,
`data/ur3-sim-cube-*-{train,eval}`, `data/pick-and-place-cube-quick-v4h5-{time-s10,two-phase-s48}-{train,eval}`,
`openpi/assets/`.

### Structures

None.

### Affected Modules

None in the repository.

### Dependencies

Phase 12.

### Observation

Mean loss weight and point counts printed by `build_labels`; the per-point statistics' ranges per family.

Label datasets (490 train episodes, 102,500 rows each): reach median 10.0 / 48.0 / 19.5 frames, padding 2.2 / 42.5 /
12.8 %, window 22.7 / 17.1 / 44.3 % (per-frame, curvature, two-phase). Real second-stage datasets
`pick-and-place-cube-quick-v4h5-{time-s10,two-phase-s48}-{train,eval}` are built with the held-out 54, 102, 122,
180, 185. The simulated wrist_2 range of the delta action (about 0.001 rad) is 100 to 200 times below the real
right arm's (0.21 rad), so both stages use the union of the simulated and real q01 and q99 (real: v4h5 curvature, v4
per-frame and two-phase), written by `merge_norm_stats.py`; the simulation-only statistics are kept in
`norm_stats_sim_only/`. Data-loader workers initialise JAX, so the statistics run with `JAX_PLATFORMS=cpu`.

## Phase 14

Phase Status: completed

Round: 11

### Goal

`train_config` and `run.py train` start from the saved parameters of an earlier run, can train LoRA adapters of
the action expert only, and take their optimisation numbers from a named recipe.

### Files

`examples/ur3_dual_arm/{configs.py,recipes.py,run.py}`, `src/vla_openpi/lora_freeze.py`,
`src/vla_openpi/lora_weight_loader.py`, `src/vla_openpi/__init__.py`.

### Structures

Optional `init_checkpoint` and `lora` arguments of `train_config`; `non_lora_filter`; `ZeroLoraBLoader` (the
`lora_b` factors start at zero, the model's own initialisation draws both factors from a normal distribution);
`Recipe` (steps, peak learning rate, warm-up steps, EMA decay, kept-checkpoint period) with `full_20k`, `sim_10k`,
`lora_10k` and `lora_4k`.

### Affected Modules

`examples/ur3_dual_arm`, `vla_openpi`.

### Dependencies

None.

### Observation

Loss at step 0 of a LoRA run started from a saved checkpoint, against that run's final loss and the loss at step 0
from `pi05_base`; the number of trainable parameters; the schedule numbers `train_config` builds for each recipe.

LoRA probe on one GPU: a real-only 223-episode curvature checkpoint (19,999) as the initial weights, adapters only,
10 trainable arrays (22.1 M parameters; the full recipe trains about 430 M): loss 0.0128 at step 0 and 0.0103 at
step 50, against 0.0093 at the end of that checkpoint's own run and 0.178 from `pi05_base`; 3.2 s per step.
Schedules: `full_20k` 20,000 steps, peak 2.5e-5; `sim_10k` 10,000 steps, peak 5e-5, warm-up 500, EMA 0.999;
`lora_10k` 10,000 steps, peak 1e-4; `lora_4k` 4,000 steps, peak 1e-4, warm-up 200, EMA 0.998; every recipe decays to
a tenth of its peak with batch 32.

## Phase 15

Phase Status: completed

Round: 14

### Goal

Stage 1: three simulation-only runs with recipe `sim_10k`, one GPU each, half-way and latest checkpoints kept.

### Files

`openpi/checkpoints/pi05_ur3_sim_{time_s10,curvature_s48,two_phase_s48}_pp/`, `runs/ur3_sim_pick_and_place_cube/train/`.

### Structures

None.

### Affected Modules

None.

### Dependencies

Phases 13 and 14.

### Observation

Loss at steps 0, 100 and the end; parameters finite; `arc_resample.json` in the checkpoint's assets.

### Result

Three runs on one H100 NVL each, 10,000 steps, 1.3–1.5 s/step, no non-finite parameters at steps 5000 and 9999.
Loss at step 0 / 5000 / mean of logged steps 9500–9950: per-frame 0.165 / 0.0195 / 0.0138; two-phase 0.170 / 0.0177 /
0.0126; curvature 0.182 / 0.0198 / 0.0125. Checkpoints 5000 and 9999 carry `arc_resample.json` of the stage-1 label
dataset in their assets and are published in the HF model repository.

## Phase 16

Phase Status: completed

Round: 15

### Goal

Stage 2: three LoRA runs with recipe `lora_4k` on the real 223 episodes, initialised from the stage-1 checkpoints.

### Files

`openpi/checkpoints/pi05_ur3_sim_real_{time_s10,curvature_s48,two_phase_s48}_pp/`, `runs/ur3_sim_pick_and_place_cube/train/`.

### Structures

None.

### Affected Modules

None.

### Dependencies

Phase 15.

### Observation

As Phase 15; the training-loss curves of the three runs; open-loop error on the 5 real held-out and the 10 simulated
held-out episodes for stage-1, stage-2 and real-only checkpoints.

### Result

Training: three runs, 4,000 steps, 1.3 s/step, finite parameters at step 3999. Loss (mean of logged values within 100
steps of step 100 / 1000 / 3900): per-frame 0.064 / 0.0275 / 0.0226; two-phase 0.0665 / 0.0257 / 0.0205; curvature
0.0899 / 0.032 / 0.0262. The real-only curvature run (20,000-step schedule, full action expert) has 0.108 / 0.0377 /
0.0256 at the same steps; its learning rate is still near the peak at step 3900.

Saved weights: the averaged `params` of the stage-2 runs are degraded (ISSUE-020: every frozen bfloat16 array drifted by
about 0.3 in relative norm at EMA decay 0.998; training-objective loss of the saved curvature `params` 0.2385 against
0.0265 logged). `train/repair_stage2_params.py` rebuilds them from the stage-1 frozen arrays and the stage-2 LoRA
arrays (`lora4k_fixed/{2000,3999}`); the loss probe on the rebuilt curvature checkpoint at step 3999 is 0.0293.

Publication: the rebuilt step-3999 checkpoints are in the HF model repository under `stage2_lora_real/<policy>/3999`
(params, norm statistics, `arc_resample.json` with the keyframes path shortened); every file's SHA-256 equals the local
file's, and the curvature checkpoint downloaded from the repository served on CPU with `--lora --init-checkpoint <any
path>` and answered identically to the local copy. Served without `--lora`, the same checkpoint starts without an error
and answers differently (no adapters), with no log message. The repository README documents both stages, the
open-loop table above and the serving flags.

Open loop (`runs/ur3_sim_pick_and_place_cube/eval`, served with `--uniform --cap 1`, execute 8, worst joint, median
over rows in rad; 10 simulated and 5 real held-out episodes): points, head and stitched error

| label | model | sim points / head / stitched | real points / head / stitched |
|---|---|---|---|
| per-frame | stage 1 | 0.138 / 0.112 / 0.045 | 0.157 / 0.130 / 0.066 |
| per-frame | stage 2 | 0.174 / 0.140 / 0.066 | 0.132 / 0.108 / 0.055 |
| two-phase | stage 1 | 0.236 / 0.106 / 0.046 | 0.296 / 0.119 / 0.063 |
| two-phase | stage 2 | 0.276 / 0.122 / 0.054 | 0.229 / 0.107 / 0.055 |
| curvature | stage 1 | 0.472 / 0.110 / 0.053 | 0.418 / 0.116 / 0.061 |
| curvature | stage 2 | 0.499 / 0.135 / 0.067 | 0.382 / 0.107 / 0.058 |
| curvature | real only | 0.522 / 0.154 / 0.078 | 0.288 / 0.087 / 0.052 |

Stage 2 lowers the real held-out error of every label against stage 1 and raises the simulated one; the real-only
curvature model is lower than stage 2 on every real column. No command edge is missed by any stage-1 or stage-2 model;
the real-only model misses 8 of 24 on the simulated held-out. The evaluation reads recorded observations, so it does
not measure task success.

## Phase 17

Phase Status: completed

Round: 16

### Goal

A per-row perturbation flag and a label weight of 0 for the chunk points that lie on a pushed step. A point at row
position f of the action rows lies on a pushed step when the flag of row ceil(f) is set; the weight is
`perturbation_weight` (default 1.0, so existing datasets keep their labels). Samples anchored during or after a push
keep their other points.

### Files

`examples/ur3_dual_arm/convert_hdf5.py` (column `is_perturbation`: 1 where the `source` of either arm is
"perturbation", 0 otherwise or when absent), `src/vla_tools/episodes.py` (the flag in the loaded episode),
`src/vla_tools/resample_config.py` (`perturbation_weight`), `src/vla_tools/arc_core.py` (point weights),
`src/vla_tools/build_labels.py` (passes the flag), `tests/`.

### Structures

`ResampleConfig.perturbation_weight`; `Episode.perturbed` (bool per action row).

### Affected Modules

Intake converter, episode loader, label construction.

### Dependencies

None.

### Observation

On a synthetic episode: weights of the points on pushed rows, of the rest, and the count of both; with the default
config the labels of an existing dataset episode equal those built before. On the new data: the share of points with
weight 0 for each label family.

### Result

Four sample episodes of revision 5cd89567 converted: the flagged rows equal the recorded perturbation start and
length (episode 1 rows 58-71, episode 6 rows 107-114, episode 24 rows 37-50, episode 0 none). With
`perturbation_weight` 0 the points of weight 0 lie on flagged rows only; in the perturbed episodes 3.6-6.3 % of the
points of the per-frame labels and 5.1-6.9 % of the curvature labels (span 48). Tests: a point on row r or between
r - 1 and r takes the weight of row r; no flag, no argument or the default weight leaves weights and points unchanged.

## Phase 18

Phase Status: completed

Round: 17

### Goal

Revision 5cd89567 converted, annotated and audited: `data/ur3-sim2` from the raw episodes, five instructions per
episode drawn from the episode's task attributes (no arm named), `data/ur3-sim2-prompted`, `audit.json`.

### Files

`runs/ur3_sim2_pick_and_place_cube/` (conversion.json, prompts, audit), `data/ur3-sim2*`.

### Structures

None.

### Affected Modules

None.

### Dependencies

Phase 17.

### Observation

Episodes kept, rows, flagged perturbation rows (2,627 expected), instructions per episode, audit findings by kind.

### Result

1000 episodes, 268,700 rows (`data/ur3-sim2`), 2,627 rows flagged `is_perturbation` in 252 episodes, 5,000
instructions (231 distinct; the arm and the relay are not mentioned). Audit: no non-finite values, largest timestamp
gap 1.00 frames, no static joint, idle frames 21.5 % of the rows (0.21 % of the path); findings 1,699 frozen wrist
camera runs (1,690 entirely while the camera's arm is idle, 9 overlapping the start or end of the arm's leg by 3-20 %)
and 145 episodes with idle ends; no episode excluded.

## Phase 19

Phase Status: completed

Round: 18

### Goal

Keyframes, still-frame trimming with end padding, and a stratified held-out set of 40 episodes. Keyframe detection and
review must handle relays (a close and an open for each arm) and late closes; the timing plan keeps the threshold 0.002,
5 frames and 60 padding rows.

### Files

`runs/ur3_sim2_pick_and_place_cube/` (keyframes, timing plan, diversity), `data/ur3-sim2-timed`.

### Structures

None.

### Affected Modules

None.

### Dependencies

Phase 18.

### Observation

Keyframe issues found and accepted, open issues 0; rows dropped and padded; the held-out episodes by relay or single
arm, arm, perturbation phase, late close, distance bin and light shape.

### Result

The keyframe specs group the order and interval checks by arm (`order` accepts a list of lists, one group per arm;
a flat list is one group), because a relay may start with either arm. First pass on the base spec (grasp and release
per arm): 36 issues, 35 accepted with reasons after checking that the detected grasps and releases equal the
simulator's recorded gripper commands in all 1000 episodes. Timing (threshold 0.002, 5 frames, 60 padding rows):
still frames dropped 13,776 (leading 425, trailing 12,325, pauses outside keyframe windows 1,026); 41,750 still frames
inside keyframe windows are kept; 314,924 rows with 60,000 padding rows, 2,564 of the 2,627 flagged rows kept. Keyframes
on the timed data with the label specs: 710 issues in each, all accepted with reasons (0 open), 225 of the 231 episodes
with issues are late-close episodes. Held-out: 40 episodes by `vla_tools.diversity` (stratified over two combined
clusters, seed 0): relay 27 and single arm 13, arm left 15 and right 25, perturbed 11 (approach 4, carry 7), late close 8,
bands short 6, medium 8, long 26, all five light shapes, every 5 cm distance bin.

## Phase 20

Phase Status: completed

Round: 19

### Goal

Label datasets and normalisation statistics for the three families: the sim robot spec with the gripper times measured
on this revision (`switch_hold_s` 0.30 s after a close, 0.50 s after an open), `perturbation_weight` 0, the three
label configs of the previous run, per-point statistics merged with the real recordings' by union.

### Files

`runs/ur3_sim2_pick_and_place_cube/` (robot spec, label configs, build script, statistics),
`openpi/assets/pi05_ur3_sim2_*_pp/`, `data/ur3-sim2-*-{train,eval}`.

### Structures

None.

### Affected Modules

None.

### Dependencies

Phase 19.

### Observation

Reach medians per family, the fraction of points with weight 0, action ranges of the merged statistics.

### Result

Label datasets (`data/ur3-sim2-*-{train,eval}`: 960 training episodes, 1000 evaluation episodes), reach median 10.0 frames
(per-frame), 48.0 (curvature) and 20.3 (two-phase); points with weight 0: 0.82 %, 1.12 % and 0.56 %. Sim robot spec with
`switch_hold_s` 0.30 / 0.50 s measured on this revision. Per-point statistics merged with the real recordings' by union:
the simulated wrist_2 range at the last chunk point is 0.002 rad and the merged one 0.05 (left) and 0.214-0.391 (right,
real recordings), so real wrist_2 stays inside the normalisation range.

## Phase 21

Phase Status: completed

Round: 20

### Goal

Stage 1 (recipe `sim_15k`) for the three families, one GPU each. Stage 2 is Phases 22 and 23.

### Files

`examples/ur3_dual_arm/recipes.py` (`SIM_15K`), `runs/ur3_sim2_pick_and_place_cube/train/`,
`openpi/checkpoints/pi05_ur3_sim2_*_pp/`.

### Structures

`Recipe` entries.

### Affected Modules

Training recipes.

### Dependencies

Phase 20.

### Observation

Loss at steps 0, 100 and the end; finite parameters; `arc_resample.json` in the assets of every checkpoint.

### Result

All three runs trained from step 0 to step 14,999 without interruption; the 51 arrays of each final checkpoint are
finite. Batch loss at step 0 / mean of steps 14,500-14,950: per-frame 0.171 / 0.0173, two-phase 0.170 / 0.0167,
curvature 0.184 / 0.0187. The final checkpoints are on the Hub model repo under `stage1_sim1000/<policy>/14999`; file hashes
equal the local files except `arc_resample.json`, whose `keyframes` path is shortened to the file name.

## Phase 22

Phase Status: completed

Round: 20

### Goal

Stage 2 trains the full action expert on real rows mixed with simulated rows. The simulated fraction of every batch
falls linearly from `mix_start` to `mix_end` over the run; each row of a batch is drawn from the simulated dataset with
the current fraction and from the real dataset otherwise, and each dataset is traversed in shuffled passes. The
real-robot tests show that the LoRA stage 2 (Phase 16) fails at full update scale while stage 1 and a real-only model
work (ISSUE-021); the mixture keeps simulated rows in every batch while the weights move toward the real rows.

### Files

`integrations/openpi/patches/0004-dataset-mixture.patch` (new openpi file `training/mixture.py`; fields `mix_in`,
`mix_fraction`, `mix_steps`, `mix_skip_steps` of `DataConfig`; `create_torch_data_loader`),
`integrations/openpi/README.md`, `src/vla_openpi/arc_data_config.py`, `examples/ur3_dual_arm/{configs,recipes,run}.py`,
`tests/test_openpi_integration.py`.

### Structures

`MixSchedule` (start and end fraction, steps, skipped steps), `MixSampler` (index sequence over the concatenation of
the real and the simulated dataset), `Recipe` entry `mix_8k` (8,000 steps, peak learning rate 2.5e-5, 300 warm-up
steps, EMA 0.999, checkpoint every 1,000 steps, multiples of 2,000 kept).

### Affected Modules

openpi data loading; the UR3 example's configuration.

### Dependencies

None.

### Observation

Realized simulated fraction per 1,000-step block of the sampler against the schedule, and the number of distinct rows
drawn from each dataset; a 30-step training run with finite loss.

### Result

Sampler test (200 + 700 rows, 4,000 steps, schedule 0.5 to 0.1): realized secondary fraction per 1,000-step block
0.450 / 0.350 / 0.249 / 0.152 against 0.450 / 0.350 / 0.250 / 0.150; all 200 and 700 rows are drawn; no row repeats within
the first pass over the primary dataset; a resumed sampler yields the continuation of a fresh one's sequence. 175 tests
pass. A run on real and simulated rows of the previous revision starts with step-0 loss 0.052 (per-frame), 0.063
(two-phase), 0.077 (curvature, schedule 0.5 to 0.0); the dataset sizes in the log are 62,410 real and 102,500
simulated rows.


## Phase 23

Phase Status: completed

Round: 22

### Goal

Mixed stage 2 on the current simulated revision: initialised from the stage-1 checkpoint at step 14,999 of Phase 21
(`pi05_ur3_sim2_<family>_pp/sim15k/14999`), simulated rows from `ur3-sim2-<r>-train` (perturbed steps masked), real rows
from `pick-and-place-cube-quick-v4h5-<r>-train`, `mix_start` 0.5, `mix_end` 0.1, recipe `mix_8k`, one GPU per run
(configuration `pi05_ur3_sim2_mix_<family>_pp`, experiment `mix_s50e10`), for the per-frame, two-phase and curvature
families.

### Files

`runs/ur3_sim2_pick_and_place_cube/train/`, `openpi/checkpoints/pi05_ur3_sim2_mix_*_pp/`,
`openpi/assets/pi05_ur3_sim2_mix_*_pp/`.

### Structures

None.

### Affected Modules

None.

### Dependencies

Phases 21 and 22.

### Observation

Training loss, finite parameters at the last step, `arc_resample.json` of the real label dataset in every checkpoint's
assets; the real-robot test of the checkpoints at steps 2,000, 4,000, 6,000 and 8,000.

### Launch

`train/chain_stage1_to_mix.sh` waits for a family's stage 1, checks its final checkpoint for non-finite values and
starts the mixed run on the same GPU. `train/launch_mix_one.sh` starts a run, `train/prepare_mix_assets.sh` places the
normalization statistics (the simulated and the real statistics files are identical), `train/watch_mix.sh` copies the
real label dataset's `arc_resample.json` into each checkpoint as it is saved and once more after the run has ended
(the final checkpoint is written after the last pass of the copy loop).

### Result

All three runs trained from step 0 to step 7,999 without interruption; the 51 arrays of each final checkpoint are
finite. Batch loss at step 0 / mean of steps 7,500-7,950: per-frame 0.043 / 0.0183, two-phase 0.055 / 0.0156,
curvature 0.074 / 0.0180. Checkpoints at steps 2,000, 4,000, 6,000 and 7,999 are kept. The final checkpoints are on the
Hub model repo under `stage2_mix_sim1000_real/<policy>/7999`; file hashes equal the local files except `arc_resample.json`, whose
`keyframes` path is shortened to the file name.

## Phase 24

Phase Status: completed

Round: 23

### Goal

Mixed stage 2 with a gated state input for the curvature family: Phase 23's configuration (initial weights
`pi05_ur3_sim2_curvature_s48_pp/sim15k/14999`, the same simulated and real datasets, simulated fraction 0.5 to 0.1,
recipe `mix_8k`) with pi0.5's discrete state input (`--state-input`: the normalised 14-dimensional state is written into
the prompt as 256-bin tokens), applied to each training sample with an open probability that rises linearly from 0 at
the first step to 1 at step 6,000 and stays 1 for the last 2,000 steps (`--state-gate 0 1 --state-gate-ramp-steps
6000`). A sample whose gate is closed has the prompt of stage 1 (the instruction without state tokens), so the run
starts on the input format of its initial weights and ends on the format used when serving with `--state-input`. Stage 1
was trained without state tokens; the flag changes the tokenized prompt only, not the parameter structure. The merged
state statistics give left wrist-2 a range of 0.0012 rad, which would stretch encoder noise over the token range; dims 0
to 5 of the state statistics are floored to a range of 0.05 rad (`vla_tools.norm_floor`) and the file is used for both
datasets. Configuration `pi05_ur3_sim2_mix_curvature_s48_state_pp`, experiment `mix_s50e10`, one GPU, started on the GPU
of Phase 23's curvature run once that run has finished.

### Structure

The gate is drawn per sample by a sampler that wraps the mixture sampler (`GatedSampler`, step order known from the
position in the stream) and is applied by a dataset wrapper (`GatedDataset`) that chooses between two tokenizations the
model transform emits for every sample: with state tokens (the format of `TokenizePrompt` with `discrete_state_input`)
and without (stage 1's format). The schedule (`GateSchedule`) is owned by the data configuration; serving builds the
same configuration without a gate and tokenizes with state.

### Interface

openpi patch 0005: `training/state_gate.py` (`GateSchedule(start, end, ramp_steps).open_probability(step)`,
`GatedSampler(inner, batch_size, schedule, skip_steps, seed)` yielding `(index, open)`, `GatedDataset(dataset)` taking
`(index, open)`, the two token key names); `DataConfig.state_gate`, `state_gate_ramp_steps`; `create_torch_data_loader`
wraps the mixture sampler and the dataset when `state_gate` is set (requires a mixture; resumes at `mix_skip_steps`).
`vla_openpi.TokenizePromptWithGate` (both tokenizations), `LeRobotArcDataConfig.state_gate`, `state_gate_ramp_steps`,
`train_config(state_gate, state_gate_ramp_steps)`, `run.py --state-gate FIRST LAST --state-gate-ramp-steps N`.

### Files

`integrations/openpi/patches/0005-state-gate.patch` and its README row, `src/vla_openpi/state_gate_tokenize.py`,
`src/vla_openpi/arc_data_config.py`, `src/vla_openpi/__init__.py`, `examples/ur3_dual_arm/configs.py`,
`examples/ur3_dual_arm/run.py`, `tests/test_openpi_integration.py`; `runs/ur3_sim2_pick_and_place_cube/train/`
(`mix_common.sh` gains the optional `variant` suffix, `prepare_mix_state_assets.sh`, `chain_mix_to_state.sh`),
`openpi/checkpoints/pi05_ur3_sim2_mix_curvature_s48_state_pp/`, `openpi/assets/pi05_ur3_sim2_mix_curvature_s48_state_pp/`.

### Structures

`GateSchedule`, `GatedSampler`, `GatedDataset`, `TokenizePromptWithGate`.

### Affected Modules

openpi data loading (patched), `vla_openpi`, the UR3 example configuration.

### Dependencies

Phase 22 (the mixture), Phase 23 (initial weights are the same; the run starts when Phase 23's curvature run ends).

### Observation

Sampled open fraction per block of steps against the schedule; resume continuing the sequence; for a gate of 1 the batch
tokens equal `TokenizePrompt` with state and for 0 the tokens without state; training loss at steps 0, 100 and the end;
finite parameters at the last step; state statistics equal Phase 23's except for the floored range of dim 4; two serve
requests that differ only in the state return different actions; `arc_resample.json` of the real label dataset in
every checkpoint's assets.

### Result

The run trained from step 0 to step 7,999 without interruption; the 51 arrays of the final checkpoint are finite and its
assets hold `arc_resample.json`. Batch loss at step 0 / mean of steps 7,500-7,950: 0.075 / 0.0176 (the curvature run
without state: 0.074 / 0.0180). Checkpoints at steps 2,000, 4,000, 6,000 and 7,999 are kept. Served on the CPU with
`--state-input` (random images, state at the centre of the statistics' range): two requests with the same state differ by
at most 0.024 (sampling noise) in the spline points; with the 12 joint entries shifted by 5 % of the statistics' range they
differ by at most 0.998, and the action sequence has 58 rows against 12.

## Phase 25

Phase Status: completed

Round: 24

### Goal

Mixed stage-2 checkpoints on the Hub model repo: step 7,999 of the three Phase 23 policies and of the Phase 24 policy
under `stage2_mix_sim1000_real/<policy>/7999`, with the README's serving table extended (no `--lora`; `--state-input` for the state
policy).

### Files

Hub model repo `README.md` and `stage2_mix_sim1000_real/`.

### Structures

None.

### Affected Modules

None.

### Dependencies

Phases 23 and 24.

### Observation

File list of each folder on the Hub; SHA-256 of every weight file against the local file; no internal path or machine
name in any uploaded file.

### Result

The Hub model repo is laid out as `stage1_sim500/`, `stage1_sim1000/`, `stage2_lora_real/` and `stage2_mix_sim1000_real/`, each with the policy
folders `per_frame/`, `two_phase/` and `curvature/` (and `curvature_state/` in the last) and the step folders below them.
Step 7,999 of the four policies is under `stage2_mix_sim1000_real/<policy>/7999`; each folder holds the same files
as its local checkpoint without the optimiser state (17 files, 18 for the state policy). SHA-256 of every file equals the
local file except `arc_resample.json`, whose `keyframes` path is shortened to the file name. The README describes the
`stage2_mix_sim1000_real/` group (data mixture, losses, gated state input) and the serving flags.

## Phase 26

Phase Status: pending

Round: 24

### Goal

Open-loop evaluation of the stage-1 and mixed stage-2 checkpoints (with and without state input) on the 40 simulated and
the 5 real held-out episodes, and of the real-only curvature checkpoint on the same episodes, served with uniform timing
at the recording's pace.

### Files

`runs/ur3_sim2_pick_and_place_cube/eval/`.

### Structures

None.

### Affected Modules

None.

### Dependencies

Phase 24.

### Observation

Points, head and stitched error, hold error and edge shifts per checkpoint on both sets.

## Phase 27

Phase Status: completed

Round: 25

### Goal

Revision `pick_and_place_quick_new` of the real recordings (Hub dataset `JingwuLuo/pick_and_place_quick_new`, raw HDF5,
71 episodes, 20,800 steps at 30 Hz, recorded after the workcell was moved) as a LeRobot v2.1 dataset
`data/pick-and-place-quick-new`: converted by `examples/ur3_dual_arm/convert_hdf5.py` with the converter configuration
of `runs/pick_place_cube_quick_v4/`, frame gaps checked on the raw timestamps before conversion, one to five instructions
per episode that name the cube, the target and the arm after the key frames of the episode have been looked at
(`language_instruction` is `pick and place` in every raw file), episodes that cannot be trained on excluded with the
reason recorded. Known properties of the revision: the left arm is used in two episodes (0051, 0052), the idle left
gripper is closed for the whole of episodes 0003-0009 and open in all others, episode 0044 has no return-to-home segment,
episodes 0028 and 0066 are about half idle. Models of the previous real recordings are not reused: the scene and the robot
placement differ.

### Files

`runs/pick_place_quick_new/` (conversion record, instructions, exclusion list), `data/pick_and_place_quick_new/` (raw,
read-only), `data/pick-and-place-quick-new/`.

### Structures

None.

### Affected Modules

None.

### Dependencies

None.

### Observation

Episode and frame counts against the raw files, step gaps per episode, instructions per episode with the key frames they
were written from, list and reasons of excluded episodes.

### Result

All 71 raw episodes have steps no more than 35.3 ms apart. Four episodes are excluded because they end with a cube on
the table instead of on a target: 44 (orange cube from the white box to the table, the reset between episodes 43 and 45),
54 (blue cube from the plate to the table at the left edge), 56 (blue cube from the white box to the table beside the
plate) and 66 (orange cube to the table at the left edge, half idle). The dataset `data/pick-and-place-quick-new` has 67
episodes and 19,776 rows; `data/pick-and-place-quick-new-prompted` carries five instructions per episode that name the
cube moved (blue 27, pink 20, orange 20 episodes) and the target (white box, plate, or another cube); the arm is not named
(episodes 51 and 52 use the left arm). Audit (`runs/pick_place_quick_new/audit.json`): no static dimension, still frames at
an end of 66 of 67 episodes (24 % of all rows), 11 isolated joint steps of 8.0 to 9.3 times the median step in 10 episodes.

## Phase 28

Phase Status: completed

Round: 26

### Goal

Training-ready real datasets for the three label families (per-frame, two-phase, curvature): audit and cleaning, still
frame and end padding decisions, keyframes checked against the images class by class, stratified held-out episodes
(five, chosen by `vla_tools` diversity measures; the two left-arm episodes are kept in the training set unless the
measures place one of them among the stratified picks), `data/pick-and-place-quick-new-<label>-{train,eval}` with the
label configurations of `runs/pick_place_cube_quick_v4/` and per-point normalisation statistics under
`openpi/assets/pi05_ur3_quick_new_<label>_state_pp/`. Every policy of this plan takes the robot state as input, so the
statistics hold the state as well; a joint entry of the state whose q01-q99 range is below 0.05 rad is floored to 0.05 rad
(`vla_tools.norm_floor`), because its encoder noise would otherwise fill the whole token range.

### Files

`runs/pick_place_quick_new/`, `data/pick-and-place-quick-new-*`, `openpi/assets/pi05_ur3_quick_new_*_state_pp/`.

### Structures

None.

### Affected Modules

None.

### Dependencies

Phase 27.

### Observation

Audit findings with the decision per finding, keyframe counts per class with the corrections, reach medians per family,
share of points with weight 0, state and action ranges per dimension of the statistics, entries floored.

### Result

Audit decisions: the 11 large joint steps are `jump_continuing` findings (the motion continues from the new value, no
spike or discontinuity) and stay; the still frames at the ends and pauses are handled by the timing plan of the earlier real
recordings (no leading, pause or trailing still frames kept, 60 rows of end padding). Keyframes: grasp and release per
arm (79 moves of the right arm in 65 episodes, one move of the left arm in each of 2 episodes), and the lift for the
two-phase family detected in every move. Eight phase findings are accepted: episodes 0 and 1 close the gripper late after a
slow approach, and episodes 6, 26, 31, 45, 47 and 61 begin with a failed grasp (the wrist images show the cube left on
the table) that is repeated; these recoveries stay in the data. Held-out: episodes 10, 11, 19, 31 and 55 (stratified over
the 3 combined clusters, 0.83 to 1.36 times the median nearest-neighbour distance to the training episodes); the two
left-arm episodes (50, 51) are in the training set. Label datasets `data/pick-and-place-quick-new-<label>-train` have 62
episodes and 17,605 rows, the eval datasets all 67; reach median 10.0 frames (per-frame), 48.0 (curvature) and 32.9
(two-phase); padding rows 1.6 %, 31.4 % and 9.4 %; weight 0 on 0.00 % of the points. Statistics: the q01-q99 range of every
state entry is above 0.05 rad (left joints 0.11 to 0.40, right joints 0.57 to 3.0) and the per-point action ranges of
every joint are above 0.02 rad, so no entry is floored.

## Phase 29

Phase Status: completed

Round: 27

### Goal

Real-only models with the state input (`--state-input`): the full action expert from `pi05_base` on `data/pick-and-place-quick-new-<label>-train` for the three
families. The number of steps keeps the number of passes over the training rows of the earlier real-only recipe
(`full_20k`: 20,000 steps of batch 32 over 62,410 rows, 10.3 passes), so the step count is passes x rows / 32 rounded to a
multiple of 500 once Phase 28 has fixed the row count; learning rate, warm-up fraction and weight averaging (decay 0.999)
are unchanged. New recipe entry in `examples/ur3_dual_arm/recipes.py`, configurations in
`examples/ur3_dual_arm/configs.py` with the `quick_new` datasets and `state_input`, one GPU per family.

### Files

`examples/ur3_dual_arm/recipes.py`, `examples/ur3_dual_arm/configs.py`, `runs/pick_place_quick_new/train/`,
`openpi/checkpoints/pi05_ur3_quick_new_*_state_pp/`.

### Structures

`Recipe` entry.

### Affected Modules

Training recipes, UR3 example configuration.

### Dependencies

Phase 28.

### Observation

Loss at steps 0, 100 and the end, finite parameters at the last step, `arc_resample.json` in every checkpoint's assets,
batch loss against the real-only models of the previous recordings at the same pass count, two serve requests that differ only in the state return different actions.

### Launch

Recipe `full_5500` (5,500 steps, learning rate 2.5e-5, warm-up 150, decay 0.999, checkpoint every 1,000 steps: 10.0
passes over 17,605 rows). `runs/pick_place_quick_new/train/norm_stats.sh` computes the statistics,
`train/launch_one.sh` starts one run on one GPU (experiment `full5500`), `train/watch_one.sh` copies the label dataset's
`arc_resample.json` into each checkpoint and checks the final one for non-finite values.

### Result

The three runs finished 5,500 steps (about 5 h each, 3.2 s per step, three runs on one node). Batch loss at step 5,450: 0.0159
(per-frame), 0.0158 (two-phase) and 0.0145 (curvature), from 0.035-0.040 at step 950. The final checkpoints
(`full5500/5499`) have 51 arrays and 0 with non-finite values, and `arc_resample.json` is in the assets of every kept
checkpoint (1000, 2000, 3000, 4000, 5000, 5499). Serve with `--state-input`.

Serve check on a GPU of each final checkpoint with `--state-input` (random images; a request with the joint state entries shifted
by 5 % of their range against the same request twice): finite actions and spline points in all three; maximum difference of the 10
spline points, same state / shifted state: per-frame 0.076 / 0.181, two-phase 0.457 / 0.677, curvature 0.471 / 0.493 (the
sampling noise on random images is of the order of the state effect, so the check shows that the state is used for per-frame and
two-phase and is inconclusive for curvature).

## Phase 30

Phase Status: completed

Round: 25

### Goal

Simulated recordings of the moved workcell (`JingwuLuo/ur3_sim_pick_and_place_cube` revision 98ee80d7: 300 right-arm episodes,
the rig's setting of 2026-11-02): conversion to LeRobot v2.1, instructions, audit, timing, keyframes checked against the
simulator's recorded gripper commands, stratified held-out episodes (four per cent), as for revision 5cd89567 (Phases 17-19).

### Files

`runs/ur3_sim3_pick_and_place_cube/`, `data/ur3-sim3*`.

### Structures

None.

### Affected Modules

None.

### Dependencies

None.

### Observation

Episode counts after exclusion, audit findings, keyframe agreement with the recorded gripper commands, held-out list.

### Result

300 episodes, 43,487 rows (`data/ur3-sim3`), 1,500 instructions (180 distinct, no arm named). Audit: no non-finite values, largest
timestamp gap 1.00 frames, static joints 0-5 (the left arm), idle frames 22.7 % of the rows (0.22 % of the path); findings 379
frozen wrist-camera runs of the idle arm's camera and 23 episodes with idle ends; no episode excluded. Keyframes: the detected
grasps and releases equal the recorded closes and opens in all 300 episodes (250 episodes with one grasp, 50 late-close episodes
with two); 8 outlier issues on the converted data and 180 and 188 on the timed data (label specs) were accepted with reasons
(`make_keyframe_overrides.py`; a window clipped by up to 5 frames at the start of an episode is accepted when the close
command falls within the first 11 rows), 0 open. Timing (threshold 0.002, 5 frames, 60 padding rows): 3,644 still frames dropped,
18,000 padding rows, 57,843 rows in `data/ur3-sim3-timed`. Held-out: 12 episodes (34, 59, 113, 114, 122, 147, 169, 206, 232, 249,
278, 286) by `vla_tools.diversity`, stratified over two combined clusters, distance to the nearest training episode 0.86-1.4
times the median nearest-neighbour distance.

## Phase 31

Phase Status: completed

Round: 28

### Goal

Simulated label datasets for the three families and per-point statistics, state included, merged with Phase 28's real
statistics by union, as in Phase 20: sim robot spec with the gripper times measured on the revision (`switch_hold_s` 0.30 s
after a close and 0.30 s after an open), `perturbation_weight` 0 (pushed steps carry no label), the label configs of Phase 28
otherwise unchanged.

### Files

`runs/ur3_sim3_pick_and_place_cube/` (robot spec, build and statistics scripts), `openpi/assets/pi05_ur3_sim3_*_state_pp/`,
`openpi/assets/pi05_ur3_joint_*_state_pp/`, `data/ur3-sim3-*-{train,eval}`.

### Structures

None.

### Affected Modules

None.

### Dependencies

Phases 28 and 30.

### Observation

Reach medians per family, share of points with weight 0, action ranges of the merged statistics.

### Result

Label datasets built (`data/ur3-sim3-*-{train,eval}`: 288 training episodes with 55,359 rows, 300 evaluation episodes with 57,843
rows). Reach median 10.0 frames (per-frame), 48.0 (curvature) and 17.7 (two-phase), against 10.0, 48.0 and 32.9 on the real
recordings; points padded beyond the episode end 46.6 % (curvature) against 31.4 % on the real recordings; points with weight 0:
1.14 %, 1.26 % and 0.74 %. The simulated episodes last 134 rows at the median (4.4 s) against 295 rows of the real ones, so a
span of 48 frames covers about twice the motion. Gripper times measured on the revision: 350 closes 0.30 s, 350 opens 0.30 s
(sim2: 0.30 s and 0.50 s).

Per-point statistics of the three families (state included) merged with the real ones by union (`merge_stats_joint.sh`): the real
right shoulder_pan and wrist_3 ranges at the last chunk point widen (curvature: 1.577 to 2.448 rad and 2.266 to 3.236 rad), the
other joints keep the real range; the same file is stored at the asset paths of the real and of the simulated dataset.

## Phase 32

Phase Status: active

Round: 29

### Goal

One-stage training of the three families with the state input: the full action expert from `pi05_base` on the real rows
(primary dataset) mixed with the simulated rows (openpi patch 0004), without a simulation-only stage and without a second
stage. The fraction of simulated rows per batch moves linearly from 0.6 at the first step to 0.15 at the last. The numbers of
passes of the single-source recipes are kept: 10 over the real training rows (`full_5500`, 17,605 rows) and 2 over the simulated
training rows (`sim_10k`, 55,359 rows), 286,768 samples of batch 32, 8,962 steps rounded to 9,000; the fractions 0.6 to 0.15
give a mean of 0.375, that is 10.2 passes of the real rows and 1.95 of the simulated rows. Learning rate 2.5e-5 (the real-only
value), warm-up 300 steps, cosine decay to 10 %, decay 0.999 (a lower decay corrupts the frozen bfloat16 arrays of the saved
`params`, ISSUE-020). Checkpoints (15 GB each) are saved every 1,000 steps and kept every 3,000. The normalisation statistics are
the union of the real and the simulated per-point statistics (the simulated wrist_2 hardly moves, so the simulated statistics
alone would put real wrist_2 targets far outside the range). A checkpoint's `arc_resample.json` is the real dataset's: the
server needs the real gripper's hold time after a switch.

### Files

`examples/ur3_dual_arm/recipes.py` (`joint_9k`), `runs/ur3_sim3_pick_and_place_cube/train/` (`merge_stats_joint.sh`,
`launch_joint_one.sh`, `watch_joint_one.sh`), `openpi/checkpoints/pi05_ur3_joint_*_state_pp/`.

### Structures

`Recipe` entry.

### Affected Modules

Training recipes.

### Dependencies

Phase 31.

### Observation

Loss at steps 0, 100 and the end, finite parameters at the last step, `arc_resample.json` in every checkpoint's assets, the
logged row counts of the mixture, two serve requests that differ only in the state return different actions.

### Launch

Recipe `joint_9k` (9,000 steps, learning rate 2.5e-5, warm-up 300, decay 0.999, checkpoints every 1,000 steps, kept every 3,000),
`--mix-fraction 0.6 0.15`, experiment `joint9k`, config names `pi05_ur3_joint_<family>_state_pp`, one GPU per family on one
node. The logged mixture is 17,605 real + 55,359 simulated rows. `merge_stats_joint.sh` builds the statistics,
`launch_joint_one.sh` starts one run, `watch_joint_one.sh` copies the real dataset's `arc_resample.json` into each checkpoint and
checks the final one for non-finite values.

## Phase 33

Phase Status: pending

Round: 30

### Goal

Open-loop evaluation of the real-only and joint checkpoints on the real held-out episodes, and of the joint checkpoints on the
simulated held-out episodes, served with uniform timing at the recording's pace. Delivery of the six final checkpoints
(`full5500/5499` and `joint9k/8999` of the three families) to the Hub model repository `JingwuLuo/ur3_sim_pick_and_place_cube`
as `real_quick_new/<per_frame|two_phase|curvature>/5499` and `joint_sim300_real_quick_new/<per_frame|two_phase|curvature>/8999`,
with the model card extended by both groups (`runs/ur3_sim3_pick_and_place_cube/publish/`: `make_readme_v6.py`,
`upload_quick_new.py`, `verify_upload_quick_new.py`, `publish_when_done.sh`). The upload starts when the joint runs have finished
and passed the finite check; the checkpoints' files are scanned for internal paths before upload and compared by SHA-256 afterwards.

### Files

`runs/pick_place_quick_new/eval/`, `runs/ur3_sim3_pick_and_place_cube/eval/`, `runs/ur3_sim3_pick_and_place_cube/publish/`.

### Structures

None.

### Affected Modules

None.

### Dependencies

Phases 29 and 32.

### Observation

Points, head and stitched error, hold error and edge shifts per checkpoint and set.
