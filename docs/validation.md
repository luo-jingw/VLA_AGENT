# Validation

Two kinds of evidence back the tools. Observations on real data were made on two datasets: one
dual-arm UR3 dataset (22 episodes, 3223 frames, 15 Hz, parquet-embedded PNG images), in an openpi fork
whose model, loader and config changes are equivalent to `integrations/openpi/patches`; and one LIBERO
task (50 demonstrations, simulated Franka Panda, 20 Hz), taken through the data stages S0–S7 with the
current tools (see [LIBERO run](#libero-run)). The test suite re-checks the mechanisms on synthetic data
on every run.

## Test suite

```bash
pip install -e ".[test]"
pytest                                    # data tools, labels, interpolation, LoRA merge: CPU only, no network
PYTHONPATH=src HF_HUB_OFFLINE=1 <openpi>/.venv/bin/python -m pytest   # adds the openpi-marked tests
```

`tests/synthetic_dataset.py` writes a LeRobot v2.1 dataset of a 3-joint arm with a binary gripper
picking a block and placing it: 10 episodes, two block regions, parquet-embedded images, an isolated
jump, a two-frame gripper pulse and a fast joint motion injected at known frames.
`tests/pipeline_chain.py` runs S0–S7 on it with the commands the skills document. The openpi-marked
tests run JAX on CPU and are skipped where openpi is not importable.

Measured on the synthetic run: next-state error 0 and same-state error 0.53 (index convention
`action[t] = state[t+1]`); the `time`, span = H labels and the label targets at their point frames
match the recording to within 1e-9 in float64 and 1e-6 after float32 storage; LoRA-merged weights
match the reference LoRA forward pass to within 1e-10 (float64), and match openpi's `lora.Einsum`
to within 1e-4 (float32, CPU).

Measured by the path tests: geometric curvature κ·R = 1 to within 2.1e-5 on circles sampled at uniform
and at 20× varying spacing, max κ 3.4e-10 on a straight line traversed while accelerating, and no change
from a stop on the path; `curvature` progress per unit length equals R^(−1/3) to within a relative 8.1e-6,
is 5.3e-5 on the accelerating straight line (`second_difference`: 10.3), and differs by at most 0.0027 of
the total between three recordings of one bent path at different speeds with a fixed chord length;
`path_metrics` reproduces constructed distances, keyframe offsets and speed ratios to within 1e-12.

Measured by the curve and window tests: the rebuilt curve passes through every recorded row (difference
0), is unchanged by 10 or 40 end-padding rows and stays at the last recorded row beyond it. On the bent
path recorded with acceleration, slowing and a pause, every frame whose window holds no keyframe and lies
within the recording has its H-th point at the recorded pose span frames after its state (920 windows,
max 5.6e-17 under `curvature`, 0 under `time`); inside a keyframe window at density 2 consecutive points
are (span/H) / 2 rows apart whatever the rest of the window holds (max deviation 5.7e-14), and windows
touching it reach at most span; a window that is still throughout is spaced in time and carries the
command edge inside it; 10 and 40 padding rows give identical progress, per-frame steps and labels of
unpadded frames under all three metrics. Targets at execution rate 2 equal the rate-1 targets at every
second frame (difference 0); the closed-loop replay of `time`, span = H labels at rate 2 on the synthetic
chain observes every 2e frames with path deviation 0 and executes 288 frames for 554 recorded (e = 4).

## vla_tools

Real-data observations in this table are on UR3 unless a row names LIBERO.

| Tool | Check | Observed on real data | Re-checked by |
|---|---|---|---|
| `arc_core` + `build_labels` | rebuild a dataset that a robot-specific builder had produced (`second_difference` progress, span 20, 3 points per gripper window, action anchor) | `action_chunk`, `action_loss_weight`, indices and observations identical element by element | — (the robot-specific builder is not part of the repository) |
| `arc_core` | `time`, span = H labels vs `action[t:t+H]`, both anchors | max difference 4.4e-9 over 3003 frames (`anchor="state"`) | `test_arc_core.py::test_time_with_span_equal_to_horizon_is_the_recorded_chunk`; `test_pipeline_chain.py::test_sweep_frame_baseline_is_exact_and_largest_admissible_span_is_selected` |
| `arc_core` | padding and keyframe-window weights; `time` gaps of 1 and 2 rows inside and outside a density-2 window; monotone frames under `curvature` and `second_difference`; refusal of a recorded config without `anchor`, of a `curvature` config without `curvature_length` (recorded with `curvature_window`, or with none), and of the metric name `arc` | — | `test_arc_core.py` |
| `arc_core` | per-window step: the H-th point at the window end; keyframe density independent of the rest of the window; a still window spaced in time; one recording with 10 and 40 end-padding rows under `time`, `curvature`, `second_difference`; a recorded-row count outside the episode | LIBERO reach and window ends, see [LIBERO run](#libero-run) | `test_arc_core.py::test_window_without_keyframes_ends_at_its_last_point`, `::test_keyframe_density_does_not_depend_on_the_rest_of_the_window`, `::test_still_window_is_spaced_in_time_and_keeps_its_command`, `::test_end_padding_does_not_change_labels_of_recorded_frames`, `::test_recorded_rows_must_be_within_the_episode` |
| `trajectory_curve` | the curve at the recorded rows; 10 vs 40 padding rows; a single recorded row | LIBERO curve selection, see [LIBERO run](#libero-run) | `test_trajectory_curve.py` |
| `path_curvature` | circle radius R at uniform and at 20× varying sample spacing; straight line traversed while accelerating; a stop on a circle and on a straight line | — | `test_path_curvature.py` |
| `arc_core` (`curvature`) | progress per unit length on circles R = 0.1, 0.2, 0.4 vs R^(−1/3); accelerating straight line; one bent path recorded at steady v, steady 2v and with acceleration, slowing and a 25-frame pause | LIBERO re-timing, see [LIBERO run](#libero-run) | `test_arc_core.py::test_curvature_on_a_circle_is_the_cube_root_of_inverse_radius`, `::test_curvature_is_zero_on_a_straight_segment`, `::test_curvature_progress_per_unit_length_is_independent_of_teleop_speed` |
| `path_metrics` | distances to an L-shaped path; a path against itself; a command switched two rows late; a doubled step; repeated frames in the step statistics | constructed values to within 1e-12 | `test_path_metrics.py` |
| `arc_interp` | against an independently written per-robot interpolator, all 3223 chunks | difference 0 | `test_arc_interp.py::test_to_frames_passes_through_the_points_and_takes_the_nearest_command` (the interpolator's rules, not the comparison) |
| `arc_interp.from_report` | outputs at the point frames vs the recording, `time` span 20 | max difference 4.4e-16 | `test_arc_interp.py::test_from_report_reproduces_the_recording_at_the_point_frames`; `test_pipeline_chain.py::test_train_and_eval_label_datasets` |
| `arc_interp` (execution rate) | point frames at rate 2; targets at rate 2 vs rate 1 | LIBERO replay at rate 2, see [LIBERO run](#libero-run) | `test_arc_interp.py::test_point_frames_by_anchor`, `::test_rate_traverses_the_same_path_faster` |
| `dataset_writer` | row drops, repeated rows, end padding, exclusion with renumbering, added and replaced columns | rows, image bytes, `frame_index`, `index`, `timestamp`, `episodes.jsonl` consistent; LeRobot loads the result | `test_dataset_writer.py::test_round_trip_renumbers_and_keeps_rows_consistent`; `test_openpi_integration.py::test_lerobot_loads_the_derived_datasets` |
| `dataset_writer` | row edits on a video dataset; an existing output directory | — | `test_dataset_writer.py` (both refused) |
| `audit` | jump classification vs inspection of joint traces and camera frames | 4 of 4 jumps classified "continuing", matching inspection | `test_cleaning_ops.py` (isolated, continuing, discontinuity, last frame); `test_pipeline_chain.py::test_audit_reports_alignment_and_injected_faults` |
| `audit` | the return frame of an isolated glitch | — | `test_cleaning_ops.py::test_isolated_glitch_is_one_jump_and_its_return_is_not_reported` |
| `clean` | a continuing jump without a decision | refuses to write, lists it, exit code 2 | `test_pipeline_chain.py::test_clean_refuses_undecided_jumps` |
| `clean` | isolated repair and pulse removal, mirrored into the next state; other episodes untouched | — | `test_pipeline_chain.py::test_clean_repairs_glitch_and_pulse_only`; `test_cleaning_ops.py` |
| `keyframe_detect` | `column` signal on the action column vs `action` signal | identical keyframes | `test_keyframe_detectors.py::test_column_signal_on_the_action_column_matches_the_action_signal` |
| `keyframe_detect` | every detector type; reference order in the spec; detected frames vs generated frames; clipped windows | — | `test_keyframe_detectors.py`; `test_pipeline_chain.py::test_first_keyframe_pass_matches_truth_and_flags_clipped_release` |
| keyframe consumers | a keyframes file from another dataset version; a keyframes file with open issues | — | `test_pipeline_chain.py::test_keyframes_from_another_version_are_refused`, `::test_labels_refuse_keyframes_with_open_issues` |
| `episode_timing` | audit after end padding | same findings as before padding; keyframe window clipping resolved | `test_pipeline_chain.py::test_audit_after_padding_reports_the_recorded_rows_only`, `::test_final_keyframes_follow_the_dropped_rows` |
| `episode_timing` | still-run classes, rows kept next to motion, whole keyframe runs, padding rows holding the last target | — | `test_timing_ops.py`; `test_pipeline_chain.py::test_timing_classifies_still_runs_and_pads_ends` |
| `diversity` | cluster cut on separated and identical descriptors; a stratified split on two block regions | — | `test_diversity_clusters.py`; `test_pipeline_chain.py::test_diversity_split_covers_both_block_regions` |
| `sweep`, `label_replay` | frame baseline exact; largest admissible span selected by the path criteria; one replay row per execution length; closed-loop replay of `time`, span = H labels reproduces the recording and observes every e frames, and at rate 2 every 2e frames; observations move forward under `curvature` | LIBERO, see [LIBERO run](#libero-run) | `test_pipeline_chain.py::test_sweep_frame_baseline_is_exact_and_largest_admissible_span_is_selected`, `::test_label_replay_reports_each_execution_length`, `::test_closed_loop_replay_of_frame_labels_is_the_recording`, `::test_closed_loop_observations_move_forward_for_progress_labels`, `::test_closed_loop_at_rate_two_skips_every_other_recorded_frame` |
| `replay_eval` | against an earlier replay script, same model and episodes | same errors (rad) and gripper shifts | — (needs a served policy) |
| `lora_merge` | merged attention deltas vs the LoRA forward pass | ≤ 2e-3 against openpi `lora.Einsum` for all four equations (float32 default matmul precision, GPU); a per-head `lora_a @ lora_b` is off by 8.6e-2 on the output projection | `test_lora_merge.py`; `test_openpi_integration.py::test_merged_weight_matches_openpi_lora_einsum` |
| `lora_to_pytorch` | merged weights in a non-LoRA JAX model vs the LoRA JAX model, same observations and noise | joints 2.2e-3 rad median, 3.5e-3 max; all 160 gripper values equal | — (needs a LoRA checkpoint) |
| `lora_to_pytorch` + `policy_parity` | converted PyTorch model vs JAX, 16 frames, same observations and noise | joints 2.8e-3 rad median, 8.7e-3 max; all 160 gripper values equal | — (needs a LoRA checkpoint) |
| `policy_parity` | a conversion with the per-head merge | joints 0.083 rad median, 0.14 max; 4 of 160 gripper values differ: detected | — (needs a LoRA checkpoint) |

## LIBERO run

Measured 2026-10-13 (S0–S6) and 2026-10-14 (curve selection, S7) on one LIBERO task, data stages only.

**Source.** `pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5` from the Hugging Face
dataset `yifengzhu-hf/LIBERO-datasets` (Apache-2.0; sha256
`42189d4415d4c51aaaf0708300653fccc39239cd3f2709079a713cd8d1678a8d`): 50 demonstrations of a simulated
Franka Panda, 136–196 steps at 20 Hz, end-effector delta actions plus a gripper command of −1 / +1, and
per-step joint angles. A conversion script outside this repository writes a LeRobot v2.1 dataset in joint
space: `state[t]` = the 7 joint angles at step t and the gripper command of step t−1, `action[t]` = the
joint angles at step t+1 and the command of step t (−1 / +1 mapped to 0 / 1), so a demonstration of T
steps has T−1 rows; both cameras are rotated by 180°, the orientation openpi's LIBERO examples use;
7758 rows. `robot_spec.json`: 8 action dims, joints 0–6, command 7, state dims identical, 20 fps.

**Commands**, from the project root, with `<task>` = `libero_alphabet_soup` and `<src>` =
`libero-alphabet-soup`; H = 10 and 5 executed frames per inference follow openpi's `pi05_libero`
config and LIBERO example:

```bash
python -m vla_tools.audit           --dataset data/<src> --spec runs/<task>/robot_spec.json --out runs/<task>/audit.json --review-dir runs/<task>/audit_review
python -m vla_tools.clean           --dataset data/<src> --spec runs/<task>/robot_spec.json --plan runs/<task>/cleaning_plan.json --out data/<src>-clean
python -m vla_tools.keyframe_detect --dataset data/<src>-clean --spec runs/<task>/robot_spec.json --keyframes-spec runs/<task>/keyframes_spec.json --out runs/<task>/keyframes_pass1.json
python -m vla_tools.episode_timing  --dataset data/<src>-clean --spec runs/<task>/robot_spec.json --keyframes runs/<task>/keyframes_pass1.json --plan runs/<task>/timing_plan.json --out data/<src>-timed
python -m vla_tools.keyframe_detect --dataset data/<src>-timed --spec runs/<task>/robot_spec.json --keyframes-spec runs/<task>/keyframes_spec.json --overrides runs/<task>/keyframe_overrides.json --out runs/<task>/keyframes.json
python -m vla_tools.keyframe_review --dataset data/<src>-timed --spec runs/<task>/robot_spec.json --keyframes runs/<task>/keyframes.json --type <grasp|lift|release> --camera observation.images.agentview --out runs/<task>/kf_review
python -m vla_tools.diversity       --dataset data/<src>-timed --spec runs/<task>/robot_spec.json --keyframes runs/<task>/keyframes.json --holdout 5 --out runs/<task>/diversity.json --plot runs/<task>/diversity.png --montage runs/<task>/montage.png
python -m vla_tools.sweep           --dataset data/<src>-timed --spec runs/<task>/robot_spec.json --keyframes runs/<task>/keyframes.json --horizon 10 --head 5 --metrics time curvature --spans 10 15 20 30 40 --event-density 0 1 2 --anchor state --out runs/<task>/sweep.json   # curvature chord 4 path steps = 0.1224 rad
python -m vla_tools.label_replay    --dataset data/<src>-timed --spec runs/<task>/robot_spec.json --keyframes runs/<task>/keyframes.json --config runs/<task>/label_config.json --execute 1 2 4 5 8 10 15 20 30 40 --rate 1 --out runs/<task>/replay_curvature-s40-d2.json
python -m vla_tools.label_replay    --dataset data/<src>-timed --spec runs/<task>/robot_spec.json --keyframes runs/<task>/keyframes.json --config runs/<task>/label_config.json --execute 1 2 4 5 8 10 15 20 30 40 --rate 2 --out runs/<task>/replay_curvature-s40-d2_r2.json
python -m vla_tools.build_labels    --dataset data/<src>-timed --spec runs/<task>/robot_spec.json --keyframes runs/<task>/keyframes.json --config runs/<task>/label_config.json --exclude 3 5 12 21 34 --out data/<src>-curvature-s40-d2-train
python -m vla_tools.data_report     runs/<task>
```

`run_index` registers each version; the `time`, span 10 baseline is replayed and built the same way.
`run_status` reports S0–S7 passed and `data_report.md` is written.

| Stage | Observed |
|---|---|
| S1 audit | max \|action[t] − state[t+1]\| = 0, max \|action[t] − state[t]\| = 6.7e-2; no jump above 8× the median step, no static joint, idle 0.4% of frames; 45 episodes with 2 gripper edges, 5 with 4 |
| S2 clean | nothing to repair or decide; 50 episodes kept |
| S3, S5 keyframes | grasp (close edge), lift (joint speed above 0.02 rad/frame for 2 frames within 30 frames after the grasp), release (open edge), counts allowed to vary; after end padding 16 consistency issues: the 5 four-edge episodes are re-grasps whose failed grasp opens over the floor (early release, short lift→release interval), 3 slow carries, 3 late releases; wrist-camera sheets of episodes 0, 1 and 29 show the can between the fingers at each grasp and dropping into the basket at the final release; all 16 accepted by a person |
| S4 timing | no leading or pause still run; 2 trailing runs (19 frames) kept; the last release comes 3–21 frames (median 18) before the recording ends; 40 padding rows per episode |
| S6 diversity | two combined clusters of 42 and 8 episodes, the small one the eight longest demonstrations (170–195 rows: four re-grasps, four slow demonstrations); stratified holdout [3, 5, 12, 21, 34] keeps both clusters in training, confirmed by a person |
| S7 sweep | all 30 `time` and `curvature` cells admissible; path deviation medians ≤ 0.21 and p90 ≤ 0.45 path steps (path step 0.0306 rad); no missed command edge; 110 of 110 edges kept in every cell |

Shortlist, distances in path steps, speed as a ratio to the recorded p95 step (0.0517 rad):

| span | cell | path head median | path full median / p90 | kf position median | speed median / p95 | reach median (frames) |
|---|---|---|---|---|---|---|
| 10 | `time`, density 0 | 0.00 | 0.00 / 0.00 | 0.00 | 0.64 / 1.07 | 10.0 |
| 15 | `time`, density 0 | 0.01 | 0.01 / 0.03 | 0.03 | 0.64 / 1.07 | 15.0 |
| 20 | `curvature`, density 2 | 0.01 | 0.02 / 0.06 | 0.00 | 0.63 / 1.01 | 20.0 |
| 30 | `curvature`, density 2 | 0.03 | 0.05 / 0.12 | 0.22 | 0.65 / 1.02 | 27.0 |
| 40 | `curvature`, density 2 | 0.05 | 0.11 / 0.23 | 0.30 | 0.66 / 1.03 | 33.7 |

Selected: `curvature`, span 40, density 2, the same cell as with the episode-wide step. At density 0 every
frame of `curvature` span 40 reaches exactly 40 recorded frames; density 2 places more points inside
keyframe windows (`time`, span 40, density 2: 34.0). Keyframe position medians at span 40: 0.30 at density
2, 0.88 at density 0. With `second_difference` added (45 cells, all admissible), span 40 selects
`second_difference`, density 2: path full median 0.0947 against 0.1053 for `curvature`. For the selected
cell the time-indexed full-chunk error, kept in `sweep.json` for information, has a median of 5.42 frames
of motion with the server's point times and 0.16 with the true ones.

Closed-loop replay, distances in path steps:

| labels | e | path median / p90 / max | seam median / max | kf position median / max | missed | hold median | speed median / max | executed / recorded frames |
|---|---|---|---|---|---|---|---|---|
| `curvature`-s40-d2, rate 1 | 2 | 0.06 / 0.13 / 0.60 | 0.84 / 3.05 | 0.66 / 3.08 | 0 | 2 | 1.14 / 1.80 | 9272 / 7758 |
| | 5 | 0.03 / 0.09 / 0.63 | 0.79 / 2.95 | 0.27 / 2.51 | 0 | 3 | 1.11 / 1.74 | 9945 / 7758 |
| | 10 | 0.03 / 0.08 / 0.50 | 0.86 / 2.84 | 0.26 / 2.88 | 0 | 3 | 1.05 / 1.68 | 9690 / 7758 |
| | 20 | 0.02 / 0.07 / 0.55 | 0.84 / 2.33 | 0.26 / 1.11 | 0 | 3 | 0.98 / 1.45 | 9920 / 7758 |
| | 40 | 0.02 / 0.06 / 0.54 | 1.07 / 2.42 | 0.23 / 2.10 | 0 | 3 | 0.98 / 1.72 | 10200 / 7758 |
| `curvature`-s40-d2, rate 2 | 2 | 0.01 / 0.08 / 0.46 | 1.70 / 5.68 | 0.12 / 1.41 | 0 | 1 | 2.10 / 3.36 | 4792 / 7758 |
| | 5 | 0.02 / 0.08 / 0.50 | 1.72 / 5.41 | 0.20 / 1.83 | 0 | 1 | 2.01 / 3.20 | 4845 / 7758 |
| | 10 | 0.01 / 0.07 / 0.55 | 1.69 / 4.58 | 0.19 / 1.56 | 0 | 1 | 1.90 / 2.83 | 4960 / 7758 |
| | 20 | 0.01 / 0.06 / 0.54 | 2.08 / 4.61 | 0.24 / 2.13 | 0 | 1 | 1.90 / 3.36 | 5100 / 7758 |
| `curvature`-s40-d0, rate 1 | 10 | 0.03 / 0.12 / 0.85 | 0.98 / 2.59 | 0.54 / 1.84 | 0 | 1 | 1.08 / 1.53 | 8380 / 7758 |
| `curvature`-s40-d0, rate 2 | 10 | 0.02 / 0.09 / 0.76 | 2.00 / 5.24 | 0.73 / 4.44 | 0 | 1 | 2.00 / 3.10 | 4170 / 7758 |
| `time`-s10, rate 1 | 2–10 | 0.00 / 0.00 / 0.00 | 1.00–1.01 / 2.93–3.31 | 0.00 / 0.00 | 0 | 1 | 1.08 / 1.98 | 7734–7930 / 7758 |

Path deviation and keyframe position of the selected labels do not grow from e = 2 to e = 40 at either
rate; the full tables (e = 1, 2, 4, 5, 8, 10, 15, 20, 30, 40) are in the run's
`replay_curvature-s40-d2.json`, `replay_curvature-s40-d2_r2.json` and data report. At rate 2 a chunk has
20 targets, so e = 30 and 40 repeat e = 20. At rate 1 the selected labels execute 1.20–1.31× the
recorded frames; the extra time is spent in the keyframe windows that density 2 fills with more points
(density 0: 1.08× at e = 10). At rate 2 they execute 0.62–0.66× the recorded frames, with the largest
per-frame step 1.90–2.10× (median) and 2.76–3.36× (max) the recorded p95 step. The e = 1 rows overstate
the error at rate 1 (path max 2.35 path steps; ISSUE-004). 23–31% of `curvature` points fall past the
episode end against 2–11% for `time`: labels taken on the 40 padding rows hold the final target and count
as padded.

**Curve selection.** Measured with a script outside this repository (50 episodes, recorded rows):
candidates are a parameter (row index, chord length, or its square root) and an interpolant through every
distinct row (PCHIP or not-a-knot cubic spline). Keeping every k-th row and rebuilding, distance of the
dropped rows to the rebuilt curve, and of the rebuilt curve to the full recorded polyline, p95 in path
steps:

| curve | dropped rows to curve, k = 2 / 4 | curve to recording, k = 2 / 4 |
|---|---|---|
| row index / PCHIP | 0.046 / 0.186 | 0.030 / 0.134 |
| row index / cubic spline | 0.024 / 0.105 | 0.021 / 0.081 |
| chord / PCHIP | 0.050 / 0.201 | 0.031 / 0.153 |
| chord / cubic spline | 0.046 / 0.219 | 0.028 / 0.183 |
| centripetal / PCHIP | 0.047 / 0.191 | 0.031 / 0.141 |
| centripetal / cubic spline | 0.029 / 0.134 | 0.022 / 0.099 |

On a constructed path (straight, arc R 0.25 over π, straight) sampled at steady, 2× steady and varying
speed with a pause, every candidate stays within 0.05–0.78 mrad of the true curve and the
parameterizations differ by ≤ 0.002 mrad. `trajectory_curve` uses the row-index cubic spline.

**Curvature chord.** Spread of the normalized `curvature` progress between the three constructed-path
recordings: 0.0577 with chords of 4 × each recording's median step, 0.0067 with a fixed chord, 0.010–0.018
with the fixed chord applied to the densely sampled reconstructions. Re-timing each LIBERO joint path
(PCHIP over arc length) at uniform speed, at uniform 2× speed with half the frames, and with a 0.4×–1.6×
sinusoidal speed; largest difference of the normalized progress to the recording along arc length,
median / max over episodes:

| metric | uniform | 2× | sine |
|---|---|---|---|
| `curvature`, fixed chord 0.1224 rad | 0.0036 / 0.0093 | 0.0082 / 0.0310 | 0.0036 / 0.0077 |
| `curvature`, chord 4 × each recording's median step | 0.0040 / 0.0111 | 0.0162 / 0.0450 | 0.0035 / 0.0076 |
| `second_difference` | 0.0443 / 0.0773 | 0.0479 / 0.0873 | 0.0486 / 0.0824 |

**Window against an episode-wide step.** `curvature` span 40 on the 50 episodes, unpadded labels of
recorded frames, distances in path steps; jitter is the largest joint distance between the deployed
targets (rate 1) of the labels of frames t and t + d over the frames both cover, median / p90 / max:

| density | step | reach p05 / median / p95 (frames) | window end − span, median / max (1685 windows) | jitter d = 1 | jitter d = 10 |
|---|---|---|---|---|---|
| 0 | episode-wide | 30.0 / 37.6 / 45.0 | 2.43 / 16.05 | 0.40 / 0.80 / 2.19 | 2.06 / 5.20 / 13.24 |
| 0 | per window | 40.0 / 40.0 / 40.0 | 0 / 0 | 0.47 / 0.74 / 2.02 | 2.46 / 4.65 / 8.80 |
| 2 | episode-wide | 22.0 / 32.8 / 42.9 | 1.85 / 16.26 | 0.38 / 1.31 / 2.09 | 2.37 / 10.11 / 16.95 |
| 2 | per window | 22.9 / 33.6 / 40.0 | 0 / 0 | 0.41 / 1.33 / 2.99 | 2.24 / 9.55 / 22.35 |

In the sweep the path full medians change by at most 0.03 path steps (span 40, density 0: 0.18 → 0.21)
and the keyframe position median at span 40 rises from 0.65 to 0.88 at density 0 (density 2: 0.30 both).
The closed-loop replay at e = 10 executes 8020 → 8380 frames at density 0 and 9530 → 9690 at density 2
(rate 1), 4090 → 4170 and 4800 → 4960 at rate 2: with the episode-wide step the straighter final
stretches were traversed faster than recorded, while every window now takes span frames at rate 1
(inferences with the state in the last 40 recorded frames, density 0: 183 → 249).

## integrations/openpi

| Check | Observed | Re-checked by |
|---|---|---|
| patches 0001–0003 on upstream openpi `15a9616` | apply in order with `git apply` | — |
| `LeRobotArcDataConfig` + `ArcInputs` + `RawOutputs` vs a hand-written robot config | identical repack structure, norm stats, model transforms; inputs (state, label, loss weight, images, masks) and outputs identical on 4 sampled frames | `test_openpi_integration.py::test_arc_inputs_map_cameras_label_and_weights`, `::test_outputs_trim_model_padding_and_interpolate_frames` (the transforms alone) |
| `vla_openpi.launch.norm_stats` + `launch.train` (3 steps, batch 4) on a `build_labels` dataset | norm stats written; `action_loss_weight` reaches the model as (4, 10); finite losses; checkpoint saved | — |
| `vla_openpi.launch.serving_config` vs a hand-written serving config, same checkpoint, observations and noise | per-frame targets and points identical over 4 frames; output (21, 14) | — |

## Not yet observed

- video datasets: `FrameReader`'s video path;
- the `settle` and referenced `threshold_cross` detectors on real data; `settle` and the referenced search
  that `threshold_cross` shares with `motion_onset` are tested on constructed arrays;
- a full training run started from `vla_openpi.launch`, and closed-loop robot results of any model
  trained through this harness;
- execution of resampled labels in the LIBERO simulator, at rate 1 or faster, and any training comparison
  between `curvature` and `time` labels or between per-window and episode-wide steps; joint speed limits for rates above 1 are not checked against a
  robot or simulator;
- the integration in a clean upstream openpi environment with the patches applied (every UR3 check above
  ran on a fork with equivalent changes).
