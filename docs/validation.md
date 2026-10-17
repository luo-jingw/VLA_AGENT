# Validation

Two kinds of evidence back the tools. Observations on real data were made on two datasets: one
dual-arm UR3 dataset (22 episodes, 3223 frames, 15 Hz, parquet-embedded PNG images), in an openpi fork
whose model, loader and config changes are equivalent to `integrations/openpi/patches`; and one LIBERO
task (50 demonstrations, simulated Franka Panda, 20 Hz), taken through the data stages S0–S7 with the
current tools (see [LIBERO run](#libero-run)). A subset of DROID (60 teleoperated pick-and-place episodes,
Franka Panda, 15 Hz, joint positions only) and the LIBERO task measure the rate ceiling (see
[Rate ceiling](#rate-ceiling)). The test suite re-checks the mechanisms on synthetic data on every run.

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

Measured by the label, window and rebuild tests: every label point lies on the recorded polyline (8008
points, max 1.1e-16 under `curvature` and `time`; points past the last recorded row equal it). The
server's rebuild reproduces the chord through two knots, the parabola through three and a cubic in time
through four or more (max 3.6e-15). On the bent
path recorded with acceleration, slowing and a pause, every frame whose window holds no keyframe and lies
within the recording has its H-th point at the recorded pose span frames after its state (921 windows,
max 1.1e-16 under all three metrics; the last of them ends on the last recorded row and has no padded
point); inside a keyframe window at density 2 consecutive points
are (span/H) / 2 rows apart whatever the rest of the window holds (max deviation 5.7e-14), and windows
touching it reach at most span; a window that is still throughout is spaced in time and carries the
command edge inside it; 10 and 40 padding rows give identical progress, per-frame steps and labels of
unpadded frames under all three metrics. Targets at execution rate 2 equal the rate-1 targets at every
second frame (difference 0); the closed-loop replay of `time`, span = H labels at rate 2 on the synthetic
chain observes every 2e frames and then the last recorded frame, with path deviation 0, and completes in
265 frames for 554 recorded rows (e = 4). Per-frame labels at rate 1, e = 1 execute every recorded action
including the last and complete in the recorded frames; `time` span 40 at e = 1 observes the last recorded
frame 5 times before the executed pose is within the final tolerance.

## vla_tools

Real-data observations in this table are on UR3 unless a row names LIBERO.

| Tool | Check | Observed on real data | Re-checked by |
|---|---|---|---|
| `arc_core` + `build_labels` | rebuild a dataset that a robot-specific builder had produced (`second_difference` progress, span 20, 3 points per gripper window, action anchor) | `action_chunk`, `action_loss_weight`, indices and observations identical element by element | — (the robot-specific builder is not part of the repository) |
| `arc_core` | `time`, span = H labels vs `action[t:t+H]`, both anchors | max difference 4.4e-9 over 3003 frames (`anchor="state"`) | `test_arc_core.py::test_time_with_span_equal_to_horizon_is_the_recorded_chunk`; `test_pipeline_chain.py::test_sweep_frame_baseline_is_exact_and_largest_admissible_span_is_selected` |
| `arc_core` | padding and keyframe-window weights; `time` gaps of 1 and 2 rows inside and outside a density-2 window; monotone frames under `curvature` and `second_difference`; refusal of a recorded config without `anchor`, of a `curvature` config without `curvature_length` (recorded with `curvature_window`, or with none), and of the metric name `arc` | — | `test_arc_core.py` |
| `arc_core` | per-window step: the H-th point at the window end; keyframe density independent of the rest of the window; a still window spaced in time; one recording with 10 and 40 end-padding rows under `time`, `curvature`, `second_difference`; a recorded-row count outside the episode | LIBERO reach and window ends, see [LIBERO run](#libero-run) | `test_arc_core.py::test_window_without_keyframes_ends_at_its_last_point`, `::test_keyframe_density_does_not_depend_on_the_rest_of_the_window`, `::test_still_window_is_spaced_in_time_and_keeps_its_command`, `::test_end_padding_does_not_change_labels_of_recorded_frames`, `::test_recorded_rows_must_be_within_the_episode` |
| `arc_core` | label points on the recorded polyline, points past the last recorded row on it | LIBERO: max 4.7e-16 rad over every label point of `curvature` span 40 | `test_arc_core.py::test_label_points_lie_on_the_recorded_path` |
| `path_curvature` | circle radius R at uniform and at 20× varying sample spacing; straight line traversed while accelerating; a stop on a circle and on a straight line | — | `test_path_curvature.py` |
| `arc_core` (`curvature`) | progress per unit length on circles R = 0.1, 0.2, 0.4 vs R^(−1/3); accelerating straight line; one bent path recorded at steady v, steady 2v and with acceleration, slowing and a 25-frame pause | LIBERO re-timing, see [LIBERO run](#libero-run) | `test_arc_core.py::test_curvature_on_a_circle_is_the_cube_root_of_inverse_radius`, `::test_curvature_is_zero_on_a_straight_segment`, `::test_curvature_progress_per_unit_length_is_independent_of_teleop_speed` |
| `path_metrics` | distances to an L-shaped path; a path against itself; a command switched two rows late; a doubled step; repeated frames in the step statistics | constructed values to within 1e-12 | `test_path_metrics.py` |
| `arc_interp` | not-a-knot cubic spline through the current pose and the points: chord, parabola and cubic from 2, 3 and ≥ 4 knots; points as knots, nearest-point commands, a straight line kept straight | LIBERO rebuild selection, see [LIBERO run](#libero-run) | `test_arc_interp.py::test_to_frames_is_the_not_a_knot_cubic_spline`, `::test_to_frames_passes_through_the_points_and_takes_the_nearest_command` |
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
| `sweep`, `label_replay` | frame baseline exact; largest admissible span selected by the path criteria; one replay row per execution length; closed-loop replay of `time`, span = H labels reproduces the recording and observes every e frames, at rate 2 every 2e frames, then the last recorded frame; observations move forward under `curvature` until the last recorded frame, observed again until the robot is at the final row | LIBERO, see [LIBERO run](#libero-run) | `test_pipeline_chain.py::test_sweep_frame_baseline_is_exact_and_largest_admissible_span_is_selected`, `::test_label_replay_reports_each_execution_length`, `::test_closed_loop_replay_of_frame_labels_is_the_recording`, `::test_closed_loop_observations_move_forward_for_progress_labels`, `::test_closed_loop_at_rate_two_skips_every_other_recorded_frame` |
| `label_replay` (latency, end, completion) | asynchronous and synchronous latency; the arrival rebuild without a seam for per-frame labels; a chunk that runs out holds; a chunk spent by its arrival is refused; per-frame labels at rate 1, e = 1 complete in the recorded frames; completion counts frames until the final row is reached for good | LIBERO replays, see [LIBERO run](#libero-run) | `test_label_replay.py` |
| `rate_ceiling` | velocity and acceleration ratios scale with the rate; the largest rate within the velocity limit; latency caps the rate of short chunks; a sparse chunk that cuts a corner fails the path bound; the summary selects a rate feasible on 90% but not all episodes | DROID and LIBERO, see [Rate ceiling](#rate-ceiling) | `test_rate_ceiling.py` |
| `replay_eval` | against an earlier replay script, same model and episodes | same errors (rad) and gripper shifts | — (needs a served policy) |
| `lora_merge` | merged attention deltas vs the LoRA forward pass | ≤ 2e-3 against openpi `lora.Einsum` for all four equations (float32 default matmul precision, GPU); a per-head `lora_a @ lora_b` is off by 8.6e-2 on the output projection | `test_lora_merge.py`; `test_openpi_integration.py::test_merged_weight_matches_openpi_lora_einsum` |
| `lora_to_pytorch` | merged weights in a non-LoRA JAX model vs the LoRA JAX model, same observations and noise | joints 2.2e-3 rad median, 3.5e-3 max; all 160 gripper values equal | — (needs a LoRA checkpoint) |
| `lora_to_pytorch` + `policy_parity` | converted PyTorch model vs JAX, 16 frames, same observations and noise | joints 2.8e-3 rad median, 8.7e-3 max; all 160 gripper values equal | — (needs a LoRA checkpoint) |
| `policy_parity` | a conversion with the per-head merge | joints 0.083 rad median, 0.14 max; 4 of 160 gripper values differ: detected | — (needs a LoRA checkpoint) |

## LIBERO run

Measured 2026-10-13 (S0–S6) and 2026-10-14 (rebuild selection, S7) on one LIBERO task, data stages only.

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
| S7 sweep | all 30 `time` and `curvature` cells admissible; path deviation medians ≤ 0.16 and p90 ≤ 0.34 path steps (path step 0.0306 rad); no missed command edge; 110 of 110 edges kept in every cell |

Shortlist, distances in path steps, speed as a ratio to the recorded p95 step (0.0517 rad):

| span | cell | path head median | path full median / p90 | kf position median | speed median / p95 | reach median (frames) |
|---|---|---|---|---|---|---|
| 10 | `time`, density 0 | 0.00 | 0.00 / 0.00 | 0.00 | 0.64 / 1.07 | 10.0 |
| 15 | `curvature`, density 2 | 0.01 | 0.01 / 0.03 | 0.08 | 0.62 / 1.01 | 15.0 |
| 20 | `curvature`, density 1 | 0.01 | 0.02 / 0.07 | 0.22 | 0.69 / 1.09 | 20.0 |
| 30 | `curvature`, density 2 | 0.02 | 0.04 / 0.11 | 0.22 | 0.65 / 1.03 | 27.0 |
| 40 | `curvature`, density 2 | 0.05 | 0.08 / 0.20 | 0.29 | 0.67 / 1.04 | 33.6 |

Selected: `curvature`, span 40, density 2. At density 0 every frame of `curvature` span 40 reaches exactly
40 recorded frames; density 2 places more points inside keyframe windows (`time`, span 40, density 2:
34.0). Keyframe position medians at span 40: 0.29 at density 2, 0.79 at density 0. With
`second_difference` added (45 cells, all admissible), span 40 selects `second_difference`, density 2: path
full median 0.0799 against 0.0824 for `curvature`. For the selected cell the time-indexed full-chunk error,
kept in `sweep.json` for information, has a median of 5.44 frames of motion with the server's point times
and 0.10 with the true ones.

Closed-loop replay, distances in path steps; frames are completion frames (until the executed targets, or
the recorded actions, reach the final recorded row and stay within 0.25 path steps of it):

| labels | e | path median / p90 / max | seam median / max | kf position median / max | missed | hold median | speed median / max | executed / recorded frames |
|---|---|---|---|---|---|---|---|---|
| `curvature`-s40-d2, rate 1 | 2 | 0.04 / 0.12 / 0.57 | 0.82 / 3.21 | 0.70 / 3.13 | 0 | 2 | 1.16 / 1.90 | 9460 / 7702 |
| | 5 | 0.02 / 0.08 / 0.49 | 0.78 / 2.99 | 0.29 / 2.47 | 0 | 3 | 1.10 / 1.77 | 9654 / 7702 |
| | 10 | 0.02 / 0.07 / 0.43 | 0.85 / 3.04 | 0.29 / 2.93 | 0 | 3 | 1.04 / 1.80 | 9485 / 7702 |
| | 20 | 0.01 / 0.05 / 0.47 | 0.83 / 2.33 | 0.26 / 1.12 | 0 | 3 | 0.96 / 1.42 | 9415 / 7702 |
| | 40 | 0.01 / 0.05 / 0.50 | 1.05 / 2.55 | 0.23 / 2.09 | 0 | 3 | 1.00 / 1.67 | 9304 / 7702 |
| `curvature`-s40-d2, rate 2 | 2 | 0.00 / 0.08 / 0.46 | 1.67 / 5.74 | 0.14 / 1.45 | 0 | 1 | 2.14 / 3.40 | 4768 / 7702 |
| | 5 | 0.01 / 0.06 / 0.42 | 1.71 / 5.59 | 0.23 / 1.87 | 0 | 1 | 1.99 / 3.30 | 4755 / 7702 |
| | 10 | 0.00 / 0.05 / 0.47 | 1.66 / 4.25 | 0.20 / 1.56 | 0 | 1 | 1.89 / 2.82 | 4725 / 7702 |
| | 20 | 0.00 / 0.04 / 0.50 | 2.06 / 4.72 | 0.24 / 2.16 | 0 | 1 | 1.91 / 3.33 | 4663 / 7702 |
| `curvature`-s40-d0, rate 1 | 10 | 0.02 / 0.10 / 0.57 | 0.94 / 2.60 | 0.50 / 1.89 | 0 | 1 | 1.06 / 1.54 | 8162 / 7702 |
| `curvature`-s40-d0, rate 2 | 10 | 0.00 / 0.07 / 0.65 | 1.98 / 5.16 | 0.72 / 4.46 | 0 | 1 | 1.98 / 3.05 | 4007 / 7702 |
| `time`-s10, rate 1 | 2–10 | 0.00 / 0.00 / 0.00 | 1.00–1.01 / 2.93–3.31 | 0.00 / 0.00 | 0 | 1 | 1.08 / 1.98 | 7702 / 7702 |

Path deviation and keyframe position of the selected labels do not grow from e = 2 to e = 40 at either
rate; the full tables (e = 1, 2, 4, 5, 8, 10, 15, 20, 30, 40) are in the run's
`replay_curvature-s40-d2.json`, `replay_curvature-s40-d2_r2.json` and data report. At rate 2 a chunk has
20 targets, so e = 30 and 40 repeat e = 20. Per-frame labels complete in the recorded frames at every e. At
rate 1 the selected labels take 1.21–1.25× the recorded frames (e ≥ 2); the extra time is spent in the
keyframe windows that density 2 fills with more points (density 0: 1.06× at e = 10). At rate 2 they take
0.61–0.62× the recorded frames, with the largest
per-frame step 1.89–2.14× (median) and 2.82–3.40× (max) the recorded p95 step. The e = 1 rows overstate
the error at rate 1 (path max 1.97 path steps; ISSUE-004). 23–31% of `curvature` points fall past the
episode end against 2–11% for `time`: labels taken on the 40 padding rows hold the final target and count
as padded.

**Rebuild from the label points.** Measured with a script outside this repository
(`sparse_rebuild_check.py`, 50 episodes, every recorded frame whose label has no padded point): the
current pose and the 10 label points are the nodes, rebuilt with a polyline or with PCHIP or a not-a-knot
cubic spline over a node parameter (uniform k, i.e. the server's knots; chord length; its square root; the
points' true recorded rows, which the server does not know). Error per frame: the larger of the distance
of the recorded rows of the window to the rebuilt curve and of the rebuilt curve to the recorded path,
path steps, median / p90 / max over frames:

| rebuild from the 10 points | `curvature` s40 d2 (6313 frames) | `curvature` s40 d0 (5808 frames) |
|---|---|---|
| polyline | 0.22 / 0.37 / 0.99 | 0.34 / 0.74 / 1.36 |
| PCHIP, uniform | 0.12 / 0.25 / 0.78 | 0.24 / 0.62 / 1.24 |
| cubic spline, uniform | 0.09 / 0.21 / 0.72 | 0.17 / 0.45 / 0.76 |
| PCHIP, chord | 0.12 / 0.26 / 0.82 | 0.25 / 0.65 / 1.28 |
| cubic spline, chord | 0.11 / 0.30 / 1.41 | 0.22 / 0.73 / 2.40 |
| cubic spline, centripetal | 0.09 / 0.21 / 0.71 | 0.19 / 0.56 / 1.13 |
| PCHIP, true rows | 0.12 / 0.25 / 0.76 | 0.24 / 0.60 / 1.22 |
| cubic spline, true rows | 0.08 / 0.18 / 2.67 | 0.16 / 0.42 / 4.42 |

The interpolant matters more than the parameter: uniform knots and the true rows give nearly the same
error, chord-length parameters the largest tails. `arc_interp` uses the cubic spline over uniform knots
(the server's point times); `time` labels at span 40 give 0.13 / 0.28 / 1.02 (density 0) against 0.18 /
0.35 / 0.80 for PCHIP. A joint of the spline leaves the range of its two neighbouring nodes by up to 0.06
/ 0.13 / 0.53 path steps (density 2) and 0.09 / 0.20 / 0.51 (density 0); the recorded rows between the
same nodes leave it by 0.06 / 0.15 / 0.66 and 0.12 / 0.42 / 0.85, so the excursion follows the recorded
path's own bends; PCHIP has none by construction. Label points are read linearly between the two
neighbouring recorded rows: max 4.7e-16 rad from the recorded polyline, and the 10 points of a label are
the only knots of the rebuild.

**Curvature chord.** Spread of the normalized `curvature` progress between the three constructed-path
recordings: 0.0577 with chords of 4 × each recording's median step, 0.0067 with a fixed chord, 0.010–0.018
with the fixed chord on dense resamplings of interpolations through every recorded row. Re-timing each
LIBERO joint path (PCHIP over arc length) at uniform speed, at uniform 2× speed with half the frames, and
with a 0.4×–1.6× sinusoidal speed; largest difference of the normalized progress to the recording along arc length,
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
| 0 | episode-wide | 30.0 / 37.6 / 45.0 | 2.43 / 16.05 | 0.36 / 0.78 / 2.15 | 2.04 / 5.20 / 13.26 |
| 0 | per window | 40.0 / 40.0 / 40.0 | 0 / 1.4e-14 | 0.43 / 0.72 / 1.90 | 2.45 / 4.64 / 8.84 |
| 2 | episode-wide | 22.0 / 32.8 / 42.9 | 1.85 / 16.26 | 0.36 / 1.31 / 2.16 | 2.34 / 10.13 / 16.91 |
| 2 | per window | 22.9 / 33.6 / 40.0 | 0 / 4.3e-14 | 0.39 / 1.31 / 2.89 | 2.21 / 9.57 / 22.30 |

The episode-wide rows are the labels of one step per episode (chord 4 × each recording's median step,
PCHIP over the rows), deployed with the same spline rebuild. The closed-loop replay at e = 10, ending when an
observation reaches the last recorded frame, executes 8040 (episode-wide) and 8410 (per window) frames at density 0 and 9510 and 9670 at density 2 (rate 1),
4090 / 4170 and 4800 / 4950 at rate 2: with one step per episode the straighter final stretches are
traversed faster than recorded, while every window takes span frames at rate 1.

## Rate ceiling

The highest task speed-up the labels allow within the robot's joint velocity limits, at an inference
latency L, with the labels standing in for a perfect model (`rate_ceiling`, closed-loop replay through
`label_replay`). A run is feasible when the executed targets stay within the velocity limits (finite
differences at the dataset rate, holds and chunk switches included), every recorded gripper edge is
executed, and no executed chunk target leaves the recorded segment its chunk covers by more than 2 path
steps. Speed-up is recorded / executed completion frames (until the final recorded row is reached and held
within 0.25 path steps). Per configuration, latency and mode the result is the largest speed-up among the
(rate, e) feasible on at least 90% of the episodes. Acceleration is reported, not constrained.

**Data.** DROID (`lerobot/droid_1.0.1`, revision 0eabc778, LeRobot v3.0), data files 000, 078 and 155:
2335 episodes, 1951 successful, 891 with a pick-and-place instruction (`put` / `place` … `in` / `on`), 580
with one grasp and one release; 60 kept, spread over 44 buildings and 38 collectors, 63–743 frames (median
216). Extracted by a script outside this repository (`extract_droid.py`) to the LeRobot v2.1 layout without
video: actions are the commanded joint positions and the gripper command thresholded at 0.5; the state is
the previous command, since DROID's measured joints follow the commands about 4 frames later. Panda limits
from libfranka 0.9.2 (`rate_limiting.h`): velocity 2.175 rad/s (joints 1–4), 2.61 rad/s (5–7);
acceleration 15, 7.5, 10, 12.5, 15, 20, 20 rad/s². LIBERO: the run above with the same limits at 20 fps.

| Stage (DROID) | Observed |
|---|---|
| audit | `action[t] = state[t+1]` exactly; 17 jumps above 8× the median step in 10 episodes; the recordings reach median 0.54 of the velocity limit at their fastest frame, 5 exceed it (max 1.41, all at jumps) and are left out of the rate ceiling |
| keyframes | grasp and release from the command edges, 60 / 60 each; 5 interval outliers left open (no camera review) |
| timing | leading still frames trimmed in 37 episodes (up to 41 frames), trailing in 8 (up to 41), pauses kept, 256 padding rows |
| sweep | path full median / p90 in path steps (0.0241 rad): span 32 `time` 0.34–0.37 / 0.89–0.94, `curvature` 0.31 / 0.72–0.73; span 64 `time` 1.01–1.07 / 2.46–2.56, `curvature` 0.78–0.79 / 1.70–1.71; spans 128 and 256 inadmissible (path head median 1.4–4.5); selected `curvature` s64 d2 |

**Commands**, DROID (LIBERO: the same `rate_ceiling` call with `--spans 32 40 64 128 256`):

```bash
python -m vla_tools.audit           --dataset data/droid-pick-place --spec runs/droid_pick_place/robot_spec.json --out runs/droid_pick_place/audit.json
python -m vla_tools.keyframe_detect --dataset data/droid-pick-place --spec runs/droid_pick_place/robot_spec.json --keyframes-spec runs/droid_pick_place/keyframes_spec.json --out runs/droid_pick_place/keyframes_pass1.json
python -m vla_tools.episode_timing  --dataset data/droid-pick-place --spec runs/droid_pick_place/robot_spec.json --keyframes runs/droid_pick_place/keyframes_pass1.json --plan runs/droid_pick_place/timing_plan.json --out data/droid-pick-place-timed
python -m vla_tools.keyframe_detect --dataset data/droid-pick-place-timed --spec runs/droid_pick_place/robot_spec.json --keyframes-spec runs/droid_pick_place/keyframes_spec.json --out runs/droid_pick_place/keyframes.json
python -m vla_tools.sweep           --dataset data/droid-pick-place-timed --spec runs/droid_pick_place/robot_spec.json --keyframes runs/droid_pick_place/keyframes.json --horizon 10 --head 8 --metrics time curvature --spans 10 32 64 128 256 --event-density 0 2 --anchor state --out runs/droid_pick_place/sweep.json
python -m vla_tools.rate_ceiling    --dataset data/droid-pick-place-timed --spec runs/droid_pick_place/robot_spec.json --keyframes runs/droid_pick_place/keyframes.json --horizon 10 --per-frame --metrics time curvature --spans 32 64 128 256 --event-density 0 2 --latency 0 0.05 0.1 0.2 0.4 --modes async sync --path-bound 2 --feasible-fraction 0.9 --out runs/droid_pick_place/rate_ceiling.json
```

**Results.** Largest speed-up feasible on at least 90% of the episodes (DROID 55, LIBERO 50), best
configuration, asynchronous / synchronous; per-frame labels (`time`, span 10) beside:

| L (s) | DROID best, async | DROID best, sync | DROID per-frame | LIBERO best, async | LIBERO best, sync | LIBERO per-frame |
|---|---|---|---|---|---|---|
| 0 | 1.96 `time` s32 d0, rate 2, e 16 (50) | 1.96 | 1.59 | 2.44 `curvature` s64 d2, rate 3, e 21 (46) | 2.44 | 1.99 |
| 0.05 | 1.96 `time` s32 d0, rate 2, e 15 (50) | 1.84 `time` s32 d0 | 1.50 / 1.28 | 2.87 `curvature` s64 d0, rate 3, e 20 (45) | 2.32 `curvature` s64 d2 | 1.97 / 1.66 |
| 0.1 | 1.93 `time` s32 d0, rate 2, e 14 (50) | 1.73 `time` s32 d0 | 1.48 / 1.12 | 2.65 `curvature` s128 d0, rate 3, e 40 (45) | 2.21 `curvature` s128 d2 | 1.94 / 1.42 |
| 0.2 | 1.45 `time` s32 d0, rate 1.5, e 9 (52) | 1.64 `time` s32 d0 | 0.99 / 0.99 | 2.29 `curvature` s64 d2, rate 3, e 17 (45) | 2.09 `curvature` s128 d2 | 0.97 / 1.10 |
| 0.4 | 0.97 `time` s32 d0, rate 1, e 26 (51) | 1.40 `time` s32 d0 | none / 0.74 | 2.03 `curvature` s128 d2, rate 3, e 34 (45) | 1.88 `curvature` s128 d2 | none / 0.76 |

The figure in parentheses is the number of feasible episodes at the selected setting.

- DROID: `time` s32 d0 is the best configuration at every latency in both modes. `curvature` s32 reaches
  rate 1.5 (1.30–1.45×); spans 64–256 fail the path bound on more than 10% of the episodes (`curvature` s64
  at L = 0: 43–47 of 55 feasible at rates 1–2; `time` s64: 30–33), except `curvature` s64 at L = 0.1–0.2 s
  (rate 1–1.5). The next rate fails on velocity (36–38 episodes at rate 3 for `time` s32).
- LIBERO: `curvature` at spans 64 and 128 reaches rate 3 (2.0–2.9×) where `time` labels stay at rate 2
  (1.8–2.1× at density 0, 1.5–1.8× at density 2). Spans 256 fail the path bound on every episode in both data sets (median path max 5.6–9.3
  path steps at rate 1). Per-frame labels fall to rate 1 at L = 0.2 s; at 0.4 s no asynchronous setting is
  feasible: a span-10 chunk keeps 2 (LIBERO) or 4 (DROID) targets after the latency, the robot holds once
  they are spent and the next chunk's catch-up exceeds the velocity limit (median 1.99× and 1.11×).
- The 90% figure sits at the threshold: most selected settings have exactly 50 of 55 or 45–46 of 50
  feasible episodes, so it moves between neighbouring latencies where a grid rate gains or loses one episode
  (LIBERO `curvature` s64 d0: 1.95, 2.87, 1.91 at L = 0, 0.05, 0.1 s). The largest feasible speed-up per
  episode varies smoothly: DROID `time` s32 d0, asynchronous, median 2.07, 2.00, 1.96, 1.95, 1.86 over the
  five latencies; per-frame 2.00, 1.98, 1.95, 1.47, 0.
- On DROID at L ≥ 0.2 s the asynchronous figure is below the synchronous one: at rate 2 asynchronous
  execution is feasible on 49 of 55 episodes (5 velocity, 1 path), synchronous on 50; per episode the
  asynchronous median stays higher (1.95 and 1.86 against 1.66 and 1.43).
- Velocity bounds of the recorded paths (a script outside this repository, `velocity_bound.py`): the
  recording replayed at one uniform rate is feasible on every episode up to 1.07× (DROID) and 1.96×
  (LIBERO), per episode median 1.88× and 3.30×; every recorded step at the speed of its limiting joint gives
  7.65× and 5.95×.
- End-effector deviation of the executed targets before completion at the selected asynchronous settings
  (Panda forward kinematics to the Franka Hand's tool centre point; `ee_deviation.py`, outside this
  repository), median / p90 / max: DROID `time` s32 d0 0.3 / 0.9–1.0 / 6.0–15.6 mm; LIBERO `curvature` s64 0.3–0.4
  / 1.2–1.3 / 7.3–8.5 mm, s128 1.6–2.0 / 5.4–5.5 / 17.2–18.5 mm.
- Acceleration at the selected settings, as a fraction of the nominal limits: 2.1–5.2 (DROID), 2.4–5.0
  (LIBERO); DROID's recordings themselves reach median 2.52, max 5.60.

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
  between `curvature` and `time` labels or between per-window and episode-wide steps;
- the rate ceiling on a robot: it holds the commanded targets to the Panda's nominal joint velocity limits
  in replay; tracking lag and tracking error (DROID's measured joints follow the commands about 4 frames
  later) and the controller's acceleration are not modelled;
- the integration in a clean upstream openpi environment with the patches applied (every UR3 check above
  ran on a fork with equivalent changes).
