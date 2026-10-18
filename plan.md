# Plan

Plan Status: approved

# Problem

## Current

- Resampled labels and the server timing are measured only by the label replay, which takes the executed
  targets as the robot's motion: no model error, no tracking lag, no contact. LIBERO alphabet soup
  (libero_object task 0; 50 recordings, 20 Hz, 7 joints + binary gripper; 45 train, 5 held out): rate
  ceiling with velocity timing and switch holds of 0.533 s (close) / 0.667 s (open) 2.94× at L = 0
  (`curvature` s64 d0, cap 16, e 4), 2.69× at L = 0.2 s async; per-frame labels 2.82× at L = 0, 1.05× at
  L = 0.2 s async (90% of the episodes).
- The switch holds are DROID's measured gripper times. LIBERO's recordings hold the finger positions
  (`obs/gripper_states`), not yet measured.
- No model has been trained on these labels. The label datasets `curvature-s40-d2` and `time-s10`
  (train / eval) exist.
- The machine has no LIBERO simulator: openpi's `third_party/libero` submodule (pinned commit f78abd68) is
  not checked out. The recordings were made with an OSC_POSE controller; the labels are absolute joint
  targets. openpi pins robosuite 1.4.1 and mujoco 3.2.3 for its LIBERO evaluation.
- `label_replay.closed_loop` builds each chunk's targets inline (arrival rebuild, owed hold, async stall
  padding); a simulator loop needs the same construction.

## Problem

Whether a policy trained on resampled labels and executed with velocity timing and switch holds succeeds
at the replay's speed-ups is unknown: model error, tracking lag and contact are outside the replay.

## Goal

- Two pi0.5 policies trained identically from `pi05_base`, the VLM frozen (SigLIP and Gemma 2B), only the
  action expert (Gemma 300M) and the action and time projections trained; H = 10, batch 32, 10k steps:
  - S: `curvature` span 64, density 0 (best or near best with velocity timing and holds at L = 0–0.2 s);
  - F: per-frame `time` span 10 (the existing dataset).
- Simulator execution: absolute joint targets through robosuite's joint position controller at 20 Hz, kp
  chosen by replaying the recorded joint trajectories; the command 0 / 1 → LIBERO −1 / +1. Inference
  latency emulated in simulated time, d = round(L · 20) frames, sync and async as in the replay.
- Closed loop on libero_object task 0's 50 benchmark initial states, 10 settle frames then at most 280
  control frames (openpi's LIBERO budget). Per condition: success rate with a Wilson 95% interval,
  completion frames of the successful trials and the speed-up against the recordings' median length
  (152 frames) and against F at rate 1, failure phase (no grasp / lifted but not placed / timeout).
  Conditions, 16 × 50 trials:
  - F, uniform rate 1, e 5, L 0 (reference) and S, uniform rate 1, e 5, L 0;
  - S and F, velocity timing with LIBERO's holds, at the rate ceiling's best (cap, e) for each
    L ∈ {0, 0.1, 0.2 s} × {sync, async};
  - S, velocity timing with holds, L 0, caps 2, 4, 8;
  - S, velocity timing without holds, L 0, its best cap.
- The labels themselves in physics: the label replay's executed targets for S and F at the same settings,
  open loop from the 50 recordings' initial simulator states (`states[0]`), success rate.

# Structure

## Modules

- Chunk construction: `src/vla_tools/arc_interp.py` (new `chunk_targets`; standalone, numpy + scipy).
- Execution model: `src/vla_tools/label_replay.py` (`closed_loop` calls `chunk_targets`).
- Simulator integration, new package `src/vla_libero/` (imports LIBERO and robosuite, which are not
  package dependencies, as `vla_openpi` imports openpi):
  - `env.py`: the LIBERO task with a joint position or the recorded OSC controller;
  - `policy.py`: the chunk policy boundary and its websocket client;
  - `rollout.py`: one closed-loop episode with emulated latency;
  - `replay.py`: open-loop replays from recorded initial states (CLI);
  - `evaluate.py`: closed-loop trials of one condition over benchmark initial states (CLI).
- Serving points: `src/vla_openpi/launch.py` (`serve_points`).
- LIBERO training example: `examples/libero_object/configs.py` (TrainConfig), `run.py` (norm stats,
  training, serving points).
- LIBERO gripper actuation, outside the package: `FTpi05/reports/sim-eval-src/libero_gripper_actuation.py`.
- Environments, outside the repository: simulator venv `FTpi05/.work/libero-sim` (Python 3.11, mujoco 3.2.3,
  robosuite 1.4.1, numpy 1.26, bddl 1.0.1, CPU torch for LIBERO's initial-state files, openpi-client, this
  package); LIBERO at openpi's pinned commit; `LIBERO_CONFIG_PATH` set to `FTpi05/.work/libero-config`.
- Knowledge: `docs/validation.md`, `README.md`, `.claude/skills/vla-downstream/SKILL.md`.

## Responsibilities

`chunk_targets` turns one chunk's points into the targets it executes, given the commanded history; it
holds no state. `closed_loop` (replay) and `rollout` (simulator) own their commanded histories and call it,
so both execute chunks identically. `LiberoJointEnv` owns the simulator; it receives one target per control
frame and reports observations, success and the object's height. `ChunkPolicy` is the only path from an
observation to points. `evaluate` and `replay` own their result files.

## State Ownership

- Commanded history (start pose and every target sent) and the pending chunk's observation index:
  `rollout` in the simulator, `closed_loop` in the replay.
- Simulator state, controller gain: `LiberoJointEnv` (kp a constructor argument; the chosen value recorded
  in `runs/libero_alphabet_soup/sim_replay.json`).
- LIBERO holds: `runs/libero_alphabet_soup/gripper_actuation.json`; passed to `evaluate` and
  `rate_ceiling` as settings.
- Policies: `openpi/checkpoints/<config>/<exp>/<step>/`, with the label dataset's `arc_resample.json` in
  `assets/<repo_id>/`.
- Results: `runs/libero_alphabet_soup/rate_ceiling_libero_holds.json`, `sim_replay.json`,
  `sim_label_replay.json`, `sim_eval/<condition>.json`.

# Interface

## Interfaces

- `arc_interp.ChunkTargets(targets: np.ndarray, stalls: int)`: frozen dataclass; `targets` (n, D).
- `arc_interp.chunk_targets(points, times, command_dims, timing: VelocityTiming | None,
  history: Sequence[np.ndarray], observed: int, latency_frames: int, mode: Literal["sync", "async"],
  execute: int) -> ChunkTargets`: `history` holds the commanded poses so far, start pose first; `observed`
  is its length when the chunk's observation was taken. Returns the targets after the arrival: the owed
  hold, then the chunk (rebuilt after the latency in async mode), cut to `execute` and in async mode
  padded with its last target (`stalls` padded frames). The sync wait stays with the caller.
- `ArcInterpolator.chunk_targets(points, history, observed, latency_frames, mode, execute) -> ChunkTargets`.
- `vla_libero.env.Observation(agentview: np.ndarray, wrist: np.ndarray, state: np.ndarray)`: images
  (128, 128, 3) uint8 rotated 180°, as in the dataset; state (8,) = measured joints + commanded gripper.
- `vla_libero.env.LiberoJointEnv(bddl_file: Path, controller: Literal["joint", "osc"], kp: float)`:
  `reset(sim_state: np.ndarray) -> Observation`; `step(target: np.ndarray) -> Observation` (joint: 7 joint
  targets + command; osc: LIBERO's 7-dim action); `success() -> bool`; `object_height() -> float`;
  `joints() -> np.ndarray`.
- `vla_libero.policy.ChunkPolicy` (Protocol): `infer(observation: Observation, prompt: str) -> np.ndarray`,
  (H, D) absolute points; `WebsocketChunkPolicy(host: str, port: int)`.
- `vla_libero.rollout.Trial(success: bool, completion_frames: int | None, phase: str, inferences: int,
  stall_frames: int)`; `rollout(env, policy, interp: ArcInterpolator, prompt: str, execute: int,
  latency_frames: int, mode, settle_frames: int, budget: int) -> Trial`.
- `python -m vla_libero.replay --hdf5 <file> --source osc|joint|labels [--kp ...] [label and timing
  settings] --out <json>`.
- `python -m vla_libero.evaluate --host --port --report <arc_resample.json> --rate --execute --latency
  --mode [--velocity-fraction --switch-hold <rise s> <fall s>] --trials <first> <last> --out <json>`.
- `vla_openpi.launch.serve_points(config, checkpoint: Path, port: int) -> None`: the training config's
  outputs, (H, D) absolute points per request.
- `examples/libero_object/configs.py`: `train_config(name, data_root, repo_id, exp_name) -> TrainConfig`
  (8 dims; agentview → base_0_rgb, eye-in-hand → left_wrist_0_rgb; joints as deltas, gripper absolute;
  freeze filter: every parameter except the action expert, `action_in_proj`, `action_out_proj`,
  `time_mlp_*`); `run.py norm-stats | train | serve-points`.

## Inputs

`data/libero/raw/…alphabet_soup…_demo.hdf5` (states, actions, joint and finger positions);
`data/libero-alphabet-soup-timed`, keyframes and robot spec; `data/libero-alphabet-soup-time-s10-train`;
`pi05_base` (cached); LIBERO's bddl and initial-state files.

## Outputs

Label datasets `data/libero-alphabet-soup-curvature-s64-d0-{train,eval}`; checkpoints of S and F; the
result files under `runs/libero_alphabet_soup/` listed above.

## State Changes

None beyond the outputs.

# Flow

## Main Flow

1. Training (background, GPUs 0–3): `build_labels` S → `run.py norm-stats` (S, F) → `run.py train` (S on
   two GPUs, F on two) → `arc_resample.json` into each checkpoint's assets.
2. `chunk_targets` extracted; `closed_loop` calls it.
3. Simulator: venv, LIBERO, config → `replay --source osc` (simulator fidelity) → `replay --source joint`
   per kp (choice of kp).
4. `libero_gripper_actuation.py` → LIBERO holds → `rate_ceiling` on LIBERO with them → (cap, e) per L and
   mode for S and F → `replay --source labels` at those settings.
5. Per model, `run.py serve-points` on its own GPU; `evaluate` per condition, 50 trials split over
   parallel clients: reset to the initial state, settle, then per inference `policy.infer` →
   `chunk_targets` → `env.step` per target, the latency frames executed (async) or held (sync) in
   simulated time, until success or the budget.
6. Results into `docs/validation.md`.

# Code Mapping

## Modules

Chunk construction → `src/vla_tools/arc_interp.py`; execution model → `src/vla_tools/label_replay.py`;
simulator integration → `src/vla_libero/{__init__,env,policy,rollout,replay,evaluate}.py`; serving →
`src/vla_openpi/launch.py`; example → `examples/libero_object/{configs,run}.py`; gripper actuation →
`FTpi05/reports/sim-eval-src/libero_gripper_actuation.py`.

## Interfaces

`ChunkTargets`, `chunk_targets`, `ArcInterpolator.chunk_targets` → `arc_interp.py`; `Observation`,
`LiberoJointEnv` → `env.py`; `ChunkPolicy`, `WebsocketChunkPolicy` → `policy.py`; `Trial`, `rollout` →
`rollout.py`; replay CLI → `replay.py`; evaluation CLI → `evaluate.py`; `serve_points` → `launch.py`;
`train_config` → `configs.py`.

## State

Commanded history → `rollout.py`, `label_replay.py`; simulator and kp → `env.py`; holds →
`gripper_actuation.json`; results → `runs/libero_alphabet_soup/`.

# Implementation

## Phase 1

Phase Status: active

Round: 1

### Goal

Labels S, both trainings running.

### Files

`examples/libero_object/configs.py`, `examples/libero_object/run.py`; outside the repository
`runs/libero_alphabet_soup/label_config_curvature_s64_d0.json`, the S label datasets, norm stats.

### Structures

`train_config`.

### Affected Modules

LIBERO training example.

### Dependencies

None.

### Observation

S dataset: episodes, frames, `action_chunk` (80) and weight (10) shapes, `arc_resample.json` config.
Norm stats per dataset. Trainable and frozen parameter counts, GPU memory, seconds per step, loss at
steps 0, 100, 1000. The final checkpoints and loss curves are checked at the start of phase 5.

## Phase 2

Phase Status: pending

Round: 1

### Goal

`chunk_targets` in `arc_interp`; `closed_loop` builds its chunks with it.

### Files

`src/vla_tools/arc_interp.py`, `src/vla_tools/label_replay.py`, `tests/test_arc_interp.py`.

### Structures

`ChunkTargets`, `chunk_targets`, `ArcInterpolator.chunk_targets`.

### Affected Modules

Chunk construction, execution model.

### Dependencies

None.

### Observation

Tests. Parity: `closed_loop` outputs (executed targets, observations, stall and completion frames) on
LIBERO and DROID episodes for uniform and velocity timing, with and without holds, L 0 / 0.1 / 0.2 s,
sync and async, saved before the change and compared after (maximum difference printed).

## Phase 3

Phase Status: pending

Round: 1

### Goal

LIBERO simulator with absolute joint control; simulator fidelity; kp.

### Files

`src/vla_libero/__init__.py`, `src/vla_libero/env.py`, `src/vla_libero/replay.py` (sources osc, joint);
outside the repository the simulator venv, LIBERO checkout and config.

### Structures

`Observation`, `LiberoJointEnv`.

### Affected Modules

Simulator integration.

### Dependencies

None.

### Observation

Recorded OSC actions replayed from `states[0]`: successes of 50. Rendered images at `states[0]` against
the dataset's frame 0 (mean absolute pixel difference). Recorded joint targets replayed with the joint
controller for kp ∈ {50, 150, 400, 1000}: successes, joint tracking error (median, max) and lag frames;
the chosen kp and its reason in `sim_replay.json`.

## Phase 4

Phase Status: pending

Round: 2

### Goal

LIBERO holds, the (cap, e) per setting, the labels' success in physics.

### Files

`src/vla_libero/replay.py` (source labels); outside the package `libero_gripper_actuation.py`.

### Structures

None.

### Affected Modules

Simulator integration, reference figures.

### Dependencies

Phases 2 and 3.

### Observation

Close and open times (median, p10, p90 frames) from the finger positions. Rate ceiling for S and F with
those holds: (cap, e) and speed-up per L and mode. Open-loop label replay in the simulator per setting:
successes of 50, completion frames.

## Phase 5

Phase Status: pending

Round: 3

### Goal

Closed-loop success of S and F under the 16 conditions.

### Files

`src/vla_openpi/launch.py`, `src/vla_libero/policy.py`, `src/vla_libero/rollout.py`,
`src/vla_libero/evaluate.py`, `tests/test_vla_libero_rollout.py`.

### Structures

`ChunkPolicy`, `WebsocketChunkPolicy`, `Trial`, `rollout`, `serve_points`.

### Affected Modules

Simulator integration, serving.

### Dependencies

Phases 1, 3 and 4.

### Observation

Final checkpoints and loss curves of S and F. Rollout tests with a fake environment and policy: env steps
per inference, sync waits and async continuation, owed holds across chunks, stop on success and at the
budget. Per condition: successes of 50 with the Wilson interval, completion frames (median, p10, p90),
speed-ups, failure phases, measured inference time.

## Phase 6

Phase Status: pending

Round: 4

### Goal

Results recorded.

### Files

`docs/validation.md`, `README.md`, `.claude/skills/vla-downstream/SKILL.md`, `issues.md`,
`opportunities.md`; outside the repository `FTpi05/PROJECT.md` (simulator environment).

### Structures

None.

### Affected Modules

Knowledge.

### Dependencies

Phase 5.

### Observation

Every number in the docs traced to a result file under `runs/libero_alphabet_soup/`.
