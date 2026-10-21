# Validation

Two kinds of evidence back the tools. Observations on real data were made on two datasets: one
dual-arm UR3 dataset (22 episodes, 3223 frames, 15 Hz, parquet-embedded PNG images), in an openpi fork
whose model, loader and config changes are equivalent to `integrations/openpi/patches`; and one LIBERO
task (50 demonstrations, simulated Franka Panda, 20 Hz), taken through the data stages S0–S7 with the
current tools (see [LIBERO run](#libero-run)). A subset of DROID (60 teleoperated pick-and-place episodes,
Franka Panda, 15 Hz, joint positions only) and the LIBERO task measure the rate ceiling (see
[Rate ceiling](#rate-ceiling)). In the LIBERO simulator the labels and two policies trained on them run
closed loop (see [LIBERO simulator](#libero-simulator)). A second dual-arm UR3 dataset (40 episodes) went
through S0–S8 with a frozen-VLM policy evaluated open loop on held-out episodes (see
[UR3 optical bench](#ur3-optical-bench)). The test suite re-checks the mechanisms on synthetic data on every
run.

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
| `arc_interp` (velocity timing) | without a binding limit the targets equal the uniform ones; a fast straight slowed to the fraction of the limit (50 frames of 0.040000 rad against the closed form); a still chunk at the rate; 60 random chunks at rates 1, 4, 16 within the limit (largest 0.9999); rise and fall holds of 5 and 2 frames, also against the start pose; the owed hold after a switch; the arrival chunk time from the observed pose | DROID and LIBERO, see [Velocity timing](#velocity-timing) | `test_arc_interp.py::test_velocity_timing_without_a_binding_limit_is_the_uniform_timing`, `::test_velocity_timing_slows_a_fast_straight_to_the_fraction_of_the_limit`, `::test_still_chunk_runs_at_the_rate`, `::test_velocity_timing_keeps_every_joint_within_the_fraction`, `::test_command_switches_hold_the_arm`, `::test_owed_hold_is_what_remains_of_the_last_switch`, `::test_velocity_timed_arrival_rebuild_starts_at_the_commanded_pose` |
| `label_replay` (velocity timing) | arriving 2 frames late at cap 4 under a 0.5 rad/s limit: velocity ratio 0.9999 at e = 2, 4, 8, the arm stays 6 and 3 frames at the close and the open, observations repeat during the holds; completion lasts until the last hold is over | DROID and LIBERO, see [Velocity timing](#velocity-timing) | `test_label_replay.py::test_velocity_timing_with_latency_stays_within_the_limit_and_holds_at_switches`, `::test_completion_counts_frames_until_the_final_row_is_reached_for_good` |
| `label_replay` (latency, end, completion) | asynchronous and synchronous latency; the arrival rebuild without a seam for per-frame labels; a chunk that runs out holds; a chunk spent by its arrival is refused; per-frame labels at rate 1, e = 1 complete in the recorded frames; completion counts frames until the final row is reached for good | LIBERO replays, see [LIBERO run](#libero-run) | `test_label_replay.py` |
| `arc_interp` (chunk construction) | first and synchronous chunks are the rebuild from the observed pose cut to e; an asynchronous chunk 2 frames late is the arrival rebuild at chunk time 2, and at e 12 with 8 targets left repeats its last target 4 times; a switch at the last commanded frame owes 5 hold frames, an asynchronous chunk starts with 5, a synchronous one after a 3-frame wait with 2 | LIBERO and DROID replays identical before and after the extraction, see [LIBERO simulator](#libero-simulator) | `test_arc_interp.py::test_first_and_sync_chunks_start_at_the_observed_pose`, `::test_async_chunk_is_rebuilt_after_the_latency_and_padded`, `::test_owed_hold_leads_the_next_chunk_less_the_wait`, `::test_interpolator_chunk_targets_matches_the_function` |
| `arc_interp` (deployment default) | a report with joint velocity limits and `switch_hold_s` gives the interpolator and targets of the explicit `VelocityTiming` at cap 4, fraction 1 (peak joint speed 1.000 of the limit, 5 still rows after a 0.4 s rise at 10 Hz); a report without either field is refused naming it; `RobotSpec.switch_hold_s` loads and is checked | LIBERO S3b report with the holds added: `launch.serving_config` defaults to the same interpolator as the evaluated velocity-timing runs (cap 4, limits 2.175 / 2.61 rad/s, holds 0.2 / 0.55 s), equal targets; `velocity_fraction=None` gives uniform timing | `test_arc_interp.py::test_deployment_interpolator_is_velocity_timing_from_the_report`, `::test_deployment_needs_limits_and_holds_in_the_report`, `::test_robot_spec_switch_holds_round_trip_and_are_checked` |
| `vla_libero.rollout` | with a perfectly tracking fake environment: asynchronous at L 2, e 4 holds 2 frames for the first inference, then observes every 4 frames; synchronous holds 2 frames before every chunk; stop at the first success or the budget, failure phases from the object height; a close holds 6 frames across chunks of 2 targets; e < L asynchronous refused | LIBERO closed loop, see [LIBERO simulator](#libero-simulator) | `test_vla_libero_rollout.py` |
| `rate_ceiling` | velocity timing keeps every cap within the limit and passes the uniform ceiling (5.28 against 3.00 on a minimum-jerk profile); velocity and acceleration ratios scale with the rate; the largest rate within the velocity limit; latency caps the rate of short chunks; a sparse chunk that cuts a corner fails the path bound; the summary selects a rate feasible on 90% but not all episodes | DROID and LIBERO, see [Rate ceiling](#rate-ceiling) | `test_rate_ceiling.py` |
| `replay_eval` | against an earlier replay script, same model and episodes; edge shifts: an edge shifted +2 and −1 frames, a missed edge, a switch past the midpoint to the next edge not counted, two command dims scored on their own edges | same errors (rad) and gripper shifts; UR3 optical bench end to end, see [UR3 optical bench](#ur3-optical-bench) | `test_replay_eval.py` (where `openpi_client` is installed) |
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

### Velocity timing

With `--velocity-fraction` the server times every chunk by its geometry (`arc_interp.VelocityTiming`): the
chunk's spline is traversed so that no joint exceeds that fraction of its velocity limit, never faster than
the rate, which becomes a cap on the slow and still stretches. Each command switch holds the arm still for
the gripper to act; a chunk that replaces one still holding starts with the rest of the hold, the
re-observation stays on its frame while the arm holds, and a run ends, and completes, only once the last
hold is over (for the recorded actions too: last recorded edge plus its hold).

**Gripper actuation.** DROID's measured gripper position reaches 90% of its travel a median 8 frames after
a close command (p10 / p90 4 / 10; 1374 edges of file 000) and 10 frames after an open command (7 / 11;
1316 edges), measured by a script outside this repository (`gripper_actuation.py`). Holds 0.533 s (close,
rising command) and 0.667 s (open): 8 and 10 frames at 15 Hz, 11 and 13 at LIBERO's 20 Hz. LIBERO's
finger width gives 0.2 s / 0.55 s (see [LIBERO simulator](#libero-simulator)); the table below applies
DROID's times to both datasets.

**Commands**: the rate-ceiling call above with `--velocity-fraction 1 --switch-hold 0 0 --switch-hold 0.533
0.667 --out runs/<task>/rate_ceiling_velocity.json`.

**Results.** Largest speed-up feasible on at least 90% of the episodes, asynchronous / synchronous (one
figure at L = 0, where both are the same); α = 1:

| L (s) | DROID uniform | DROID velocity | DROID velocity + holds | LIBERO uniform | LIBERO velocity | LIBERO velocity + holds |
|---|---|---|---|---|---|---|
| 0 | 1.96 | 7.46 | 4.58 | 2.44 | 5.83 | 2.94 |
| 0.05 | 1.96 / 1.84 | 7.20 / 5.89 | 4.53 / 3.52 | 2.87 / 2.32 | 5.60 / 5.11 | 2.87 / 2.71 |
| 0.1 | 1.93 / 1.73 | 5.77 / 4.89 | 3.91 / 3.49 | 2.65 / 2.21 | 5.38 / 4.71 | 2.83 / 2.58 |
| 0.2 | 1.45 / 1.64 | 5.55 / 4.41 | 3.27 / 2.84 | 2.29 / 2.09 | 4.97 / 4.18 | 2.69 / 2.39 |
| 0.4 | 0.97 / 1.40 | 1.39 / 3.41 | 1.54 / 2.36 | 2.03 / 1.88 | 3.95 / 3.41 | 2.28 / 2.19 |

- Bounds: every recorded step at the speed of its limiting joint gives 7.65× (DROID) and 5.95× (LIBERO);
  with the same holds at every recorded edge, 4.68× and 2.96×. At L = 0 velocity timing reaches 98% of each.
- Best configurations: `curvature` s32–s64 and `time` s32 on DROID, `curvature` s64–s128 on LIBERO, at caps
  6–16. The next cap fails on the path bound or keeps no target after the latency (cap ≤ span / (d + 1));
  velocity never binds (largest ratio 1.00).
- Per-frame labels: DROID 5.73× at L = 0 and 0.82× at 0.4 s (asynchronous); a 10-frame chunk caps the rate
  at 10 / (d + 1). At 0.4 s the best labels of spans 32–128 keep 3.41× without holds and 2.19–2.36× with
  them (synchronous; LIBERO asynchronous 3.95× and 2.28×).
- With holds the arm stays at least 9 frames (DROID) and 12 (LIBERO) at every switch of every feasible run;
  the recordings stay a median 2 and 1–1.5 frames within 0.25 path steps of the switch pose; without holds
  the arm stays 1 frame.
- Asynchronous execution at 0.4 s fails the path bound on DROID (9–21 episodes at the next cap) under both
  timings; synchronous execution keeps 3.41× and 2.36×.
- End-effector deviation at the selected settings: median 0.2–2.2 mm, p90 0.9–6.3 mm, max 5.4–29.1 mm.

## LIBERO simulator

Measured 2026-10-17 and 2026-10-18 on the LIBERO task of [LIBERO run](#libero-run) (libero_object task 0),
with `vla_libero` in a simulator environment outside this repository: mujoco 3.2.3, robosuite 1.4.1, LIBERO
at openpi's pinned commit f78abd68.

**Execution.** `vla_libero.env.LiberoJointEnv` drives the Panda with absolute joint targets through
robosuite's joint position controller (critically damped, goal set to the target each control frame) or
with the recorded actions through the recorded OSC_POSE controller; the command 0 / 1 is LIBERO's gripper
action −1 / +1. Observations are the agentview and wrist images (128 × 128, rotated 180° as in the
dataset) and the state, the measured joints and the command in effect. One closed-loop trial
(`vla_libero.rollout`) starts from a benchmark initial state, holds still for 10 settling frames, then
executes each chunk's targets one per control frame through `ArcInterpolator.chunk_targets` — the
construction `label_replay.closed_loop` uses (1260 replay runs on LIBERO and DROID episodes give the same
outputs, maximum difference 0, or the same refusal before and after the extraction) — with the latency in simulated frames,
d = round(L · 20), until the task's success predicate holds or 280 control frames pass. A failure is
classed by the object's height above its settled height: not lifted (never 5 cm), dropped (lifted, below
2 cm at the end), not placed (still lifted).

**Fidelity.** The recorded OSC actions from each recording's first simulator state succeed 46 of 50
(first success a median 140.5 frames into 153 recorded actions; 45 of 50 under mujoco 2.3.7). Rendered
images at that state differ from the dataset's first frames by a mean absolute 5.5 (median per episode,
max 9.0; against unrotated images 57–60). The recorded joint angles as targets of the joint controller:

| kp | Successes of 50 | Tracking error median (rad) | Lag (frames) |
|---|---|---|---|
| 50 | 40 | 0.105 | 5 |
| 150 | 48 | 0.057 | 3 |
| 400 | 48 | 0.031 | 1 |
| 1000 | 47 | 0.016 | 1 |

kp 400 is used for the label replays and the closed loop: equal to kp 1000 at the recorded pace and more
successful when labels run fast (`curvature` s64 d0 at cap 16: 39 against 26 of 50, cap 8: 39 against 30).

**Gripper actuation.** The finger width of the 50 recordings reaches 90% of its travel a median 4 frames
after a close command (p10 / p90 3 / 16.8; 55 edges) and 11 frames after an open command (10 / 11; 53
edges): holds of 0.2 s (close) and 0.55 s (open). With these holds the rate ceiling (90% of the episodes)
selects, as cap / e / speed-up:

| L (s) | `curvature` s64 d0 async | sync | `time` s10 async | sync |
|---|---|---|---|---|
| 0 | 16 / 4 / 3.60 | — | 8 / 1 / 3.43 | — |
| 0.1 | 16 / 2 / 3.43 | 6 / 10 / 2.95 | 3 / 2 / 1.80 | 4 / 2 / 1.62 |
| 0.2 | 12 / 4 / 3.26 | 6 / 10 / 2.68 | 2 / 4 / 1.11 | 4 / 2 / 1.17 |

Without holds `curvature` s64 d0 reaches 5.78 (cap 16, e 2) at L = 0.

**Training.** Two pi0.5 policies from `pi05_base` with `examples/libero_object/configs.py`: SigLIP and
Gemma 2B frozen (2923.3 M parameters), the action expert and the action and time projections trained
(430.1 M); agentview → `base_0_rgb`, wrist → `left_wrist_0_rgb`; joints as deltas, the command absolute;
no state in the prompt; H = 10, batch 32, 10k steps, learning rate 2.5e-5 after 500 warm-up steps, cosine
decay to 2.5e-6. S is trained on `curvature`
span 64 density 0 labels, F on per-frame `time` span 10 labels, both from the 45 training recordings.
One L40S per run: 3.0 s per step. Loss at steps 0 / 1000 / 5000 / 9900: S 0.157 / 0.021 / 0.010 / 0.0063,
F 0.132 / 0.017 / 0.008 / 0.0046 (the label sets differ, so the values do not compare across runs).

**Closed loop.** Each policy served by `run.py serve-points` (the (H, D) points; the client times them),
50 trials per condition from the task's 50 benchmark initial states, kp 400. The open-loop column executes
the labels themselves at the same settings from the recordings' first states (`vla_libero.replay --source
labels`). Speed-ups are median completion frames of the successes against the recordings' median length
(152 frames) and against F at rate 1 (212 frames); intervals are Wilson 95%.

| Policy, timing | L (s), mode | Cap, e | Successes of 50 | Frames | Speed-up | Not lifted / dropped / not placed | Open loop |
|---|---|---|---|---|---|---|---|
| F, rate 1 | 0 | 1, 5 | 31 (0.48–0.74) | 212 | 0.72 / 1.00 | 5 / 11 / 3 | 48 |
| S, rate 1 | 0 | 1, 5 | 9 (0.10–0.31) | 193 | 0.79 / 1.10 | 22 / 15 / 4 | 45 |
| S, velocity + holds | 0 | 16, 4 | 22 (0.31–0.58) | 41 | 3.71 / 5.17 | 20 / 8 / 0 | 39 |
| S, velocity + holds | 0.1 async | 16, 2 | 28 (0.42–0.69) | 43 | 3.53 / 4.93 | 18 / 3 / 1 | 37 |
| S, velocity + holds | 0.1 sync | 6, 10 | 20 (0.28–0.54) | 65 | 2.34 / 3.26 | 28 / 2 / 0 | 40 |
| S, velocity + holds | 0.2 async | 12, 4 | 10 (0.11–0.33) | 56 | 2.74 / 3.82 | 33 / 7 / 0 | 39 |
| S, velocity + holds | 0.2 sync | 6, 10 | 22 (0.31–0.58) | 64 | 2.38 / 3.31 | 24 / 4 / 0 | 39 |
| S, velocity + holds | 0 | 2, 4 | 23 (0.33–0.60) | 136 | 1.12 / 1.56 | 13 / 13 / 1 | 46 |
| S, velocity + holds | 0 | 4, 4 | 31 (0.48–0.74) | 81 | 1.88 / 2.62 | 12 / 7 / 0 | 44 |
| S, velocity + holds | 0 | 8, 4 | 29 (0.44–0.71) | 42 | 3.62 / 5.05 | 17 / 4 / 0 | 39 |
| S, velocity, no holds | 0 | 16, 2 | 6 (0.06–0.24) | 40 | 3.85 / 5.37 | 42 / 2 / 0 | 11 |
| F, velocity + holds | 0 | 8, 1 | 42 (0.71–0.92) | 92 | 1.64 / 2.29 | 2 / 5 / 1 | 41 |
| F, velocity + holds | 0.1 async | 3, 2 | 41 (0.69–0.90) | 95 | 1.60 / 2.23 | 5 / 3 / 1 | 48 |
| F, velocity + holds | 0.1 sync | 4, 2 | 41 (0.69–0.90) | 170 | 0.89 / 1.25 | 4 / 4 / 1 | 48 |
| F, velocity + holds | 0.2 async | 2, 4 | 37 (0.60–0.84) | 142 | 1.07 / 1.49 | 4 / 8 / 1 | 46 |
| F, velocity + holds | 0.2 sync | 4, 2 | 30 (0.46–0.72) | 211 | 0.72 / 1.00 | 7 / 9 / 4 | 48 |

- Every condition succeeds less closed loop than its labels open loop; the labels themselves lose
  successes as the cap rises (cap 2 46, cap 16 39) and almost all of them without holds (11).
- S succeeds less than F at every matched setting; most of S's failures leave the object on the floor
  (ISSUE-011).
- Velocity timing with holds raises F from 31 to 42 of 50 and completes in 0.44 of F's rate-1 frames
  (L = 0); at L = 0.1 s asynchronous F keeps 41 at 2.23×. S at cap 8 succeeds 29 of 50 (interval
  overlapping F at rate 1) at 5.05× F's rate-1 pace.
- No initial state fails in all 15 conditions with holds; per-state success counts of S and F correlate
  at 0.33.
- Two runs of the same open-loop condition differ by up to 1 success.

**Whole chunks.** Measured 2026-10-19, same policies and task, kp 400, L = 0, sync (at L = 0 sync and async
run the same code), 50 trials per cell; "full" executes every target of a chunk before the next observation
(`--execute 1000`). Labels: the same setting open loop from the recordings' first states.

| Policy | Timing | Execute | Successes of 50 | Not lifted / dropped / not placed | Frames (successes) | Inferences per trial | Labels |
|---|---|---|---|---|---|---|---|
| S | rate 1 | 5 | 17 | 17 / 10 / 6 | 192 | 56 | 45 |
| S | rate 1 | full | 22 | 25 / 3 / 0 | 144 | 5 | 46 |
| S | velocity cap 4 + holds | 4 | 33 | 13 / 4 / 0 | 82 | 28 | 44 |
| S | velocity cap 4 + holds | full | 26 | 21 / 3 / 0 | 54 | 6 | 46 |
| F | rate 1 | 5 | 32 | 5 / 8 / 5 | 192 | 48 | 48 |
| F | rate 1 | full | 39 | 3 / 7 / 1 | 159 | 18 | 48 |
| F | velocity cap 4 + holds | 4 | 42 | 4 / 4 / 0 | 108 | 47 | 45 |
| F | velocity cap 4 + holds | full | 43 | 4 / 3 / 0 | 98 | 38 | 46 |

S at rate 1, e = 5 succeeded 9, 17 and 14 of 50 in three runs of the same setting (the last in the traced
runs below); S at cap 4, e = 4 31 and 33; F at rate 1, e = 5 31, 32 and 28.

**Close attempts.** A script outside this repository (`reports/sim-eval-src/gripper_trace.py`) runs the same
closed loop and records per control frame the commanded gripper, the finger opening, and the grip site and
object positions; `gripper_attempts.py` classes every commanded close up to the next open. Rate 1, sync,
L = 0; attempts of 50 trials:

| Policy, execute | Successes | Close attempts | Lifted the object | Reopened within 6 frames | Fingers closed on nothing | Fingers on the object, not lifted |
|---|---|---|---|---|---|---|
| S, 5 | 14 | 193 | 45, at 0.036 m | 33 | 85, at 0.068 m | 30 |
| S, full | 25 | 115 | 34, at 0.029 m | 2 | 72, at 0.060 m | 7 |
| F, 5 | 28 | 187 | 90, at 0.021 m | 73 | 18, at 0.041 m | 6 |
| F, full | 36 | 165 | 70, at 0.024 m | 70 | 21, at 0.042 m | 4 |

Distances are the median from the grip site to the object's centre at the close. Of S's empty closes, the
grip site never came within the lifting closes' 90th-percentile distance (0.050 / 0.042 m) from 10 frames
before to 5 frames after the close in 48 of 85 (e = 5) and 54 of 72 (full); in 9 and 5 it came within that
distance at another frame. No close lifted the object in 19 of S's 25 failed full-chunk trials and 13 of its
36 failed trials at e = 5. A missed close is followed by an open and another approach: a median 3 (full)
and 4.5 (e = 5) closes per failed S trial.

**Event density 2.** S2: `curvature` span 64 with event density 2, otherwise S's labels (points inside
keyframe windows 24.9% against 16.1%; the 10th point a median 51.5 recorded frames ahead against 64.0) and
S's training on two L40S data parallel (global batch 32, 1.7 s per step, 4 h 52 min; loss 0.1458 / 0.0203 /
0.0096 / 0.0062 at steps 0 / 1000 / 5000 / 9900). Measured 2026-10-19 in the whole-chunk cells; successes of
50 (not lifted / dropped / not placed) and the successes' median completion frames:

| Timing | Execute | S2 | S | F | Labels S2 / S / F |
|---|---|---|---|---|---|
| rate 1 | 5 | 12 (22 / 10 / 6), 196 | 17 (17 / 10 / 6), 192 | 32 (5 / 8 / 5), 192 | 45 / 45 / 48 |
| rate 1 | full | 26 (13 / 8 / 3), 166 | 22 (25 / 3 / 0), 144 | 39 (3 / 7 / 1), 159 | 45 / 46 / 48 |
| velocity cap 4 + holds | 4 | 30 (14 / 5 / 1), 72 | 33 (13 / 4 / 0), 82 | 42 (4 / 4 / 0), 108 | 46 / 44 / 45 |
| velocity cap 4 + holds | full | 28 (20 / 2 / 0), 70 | 26 (21 / 3 / 0), 54 | 43 (4 / 3 / 0), 98 | 48 / 46 / 46 |

S2's successes differ from S's by −5 to +4 per cell, within the spread of repeated runs of one setting (S
at rate 1, e = 5: 9, 17, 14). Close attempts at rate 1 (traced runs, S2 13 and 26 of 50):

| Policy, execute | Close attempts | Lifted the object | Reopened within 6 frames | Fingers closed on nothing | Fingers on the object, not lifted |
|---|---|---|---|---|---|
| S2, 5 | 123 | 40, at 0.032 m | 21 | 47, at 0.050 m | 15 |
| S2, full | 107 | 40, at 0.028 m | 4 | 51, at 0.053 m | 12 |

Closes on nothing fall from 44% to 38% (e = 5) and from 63% to 48% (full) of S's closes, against 10–13%
for F. In 44 of S2's 51 full-chunk empty closes the grip site never came within 0.040 m of the object from
10 frames before to 5 after the close.

**Grasp accuracy open loop and close timing.** A script outside this repository
(`reports/sim-eval-src/grasp_diagnostic.py`) sends recorded observations, from 16 frames before each
episode's first recorded close up to the close, to each checkpoint with four seeded noise draws (the same
for every checkpoint), rebuilds the per-frame targets at rate 1, and compares the target for the recorded
close frame with the recorded target there through the Panda's kinematics. Episodes held out of training
3, 5, 12, 21, 34 and training episodes 0, 10, 20, 30, 40; tool-centre-point distance, median (held-out /
training):

| Lookahead | F | S | S at step 5000 | S2 |
|---|---|---|---|---|
| 1–5 frames | 8.0 / 5.8 mm | 9.0 / 7.7 mm | 8.7 / 9.0 mm | 11.0 / 11.8 mm |
| 6–10 frames | 18.5 / 11.1 mm | 18.4 / 13.1 mm | 24.2 / 16.6 mm | 24.3 / 22.1 mm |
| 11–16 frames | — | 27.6 / 19.6 mm | 29.3 / 22.7 mm | 34.2 / 25.9 mm |

S's points against its labels, held-out: 16 mm at the first point, 48–69 mm at points 6–9 (F 3 mm at the
first, 19 mm at the tenth); the mean over noise draws is off by nearly as much (14 and 43–67 mm), the draws
spread 5–21 mm around it.

The rebuild gives every target the command of the nearest point in time; the observed pose is not a knot
for commands, so at rate 1 with S's spacing (6.4 frames) targets 1–9 carry point 1's command. Built from the
labels themselves, observation by observation, the first target commands the close 4 frames before the
recorded close for S (p10 / p90 3 / 6; the recorded tool centre point is then 22 mm from where the
recording closes, p90 36 mm), 1 frame for S2 (4 mm) and on the recorded frame for F. Closed loop at rate 1
(traces of the whole-chunk grid): at e = 5 every first close of S and S2 is the first target of a chunk, the
grip site 49 mm (S) and 45 mm (S2) from the object at first closes that close on nothing against 34 and
32 mm at first closes that lift it, while F's first closes come a median 2–4 targets into the chunk, 23 mm
from the object when they lift it. With whole chunks S's first
close comes a median 41 frames after the observation that produced it (S2 54), taken when the grip site was
0.32 m from the object.

**Command rule, execution speed, settling.** Scripts outside this repository: `command_rule.py` replaces the
rebuild's command assignment at run time (`nearest`, the rebuild's own rule, reproduces it exactly on 200
random chunks; `midpoint`: the commanded command at time 0 is a knot, a switch to point 1 comes half way;
`hold`: a switch comes at the point that carries it; `settle`: `nearest` plus 3 frames at the switching pose
with the old command before every switch); `gripper_trace.py` now also records the commanded joints;
`close_lag.py` compares the commanded and the actual tool centre point at each first close. Same task, kp
400, sync, L = 0, 50 trials; half speed with a 560-frame budget (its labels succeed a median 276–325 frames
in). Successes closed loop (labels open loop); `hold` at e = 5, rate 1 never switches (its first switch
falls after the next observation) and is left out:

| Policy | Cell | nearest | midpoint | hold | settle |
|---|---|---|---|---|---|
| S | rate 1, e 5 | 15 (45) | 14 (50) | — | 13 (44) |
| S | rate 0.5, e 10 | 19 (38) | — | — | — |
| S | rate 1, full | 17 (46) | 22 (47) | 22 (42) | 19 (46) |
| S | velocity cap 4 + holds, e 4 | 26 (44) | 30 (44) | 27 (40) | 32 (44) |
| S2 | rate 1, e 5 | 12 (45) | 13 (47) | — | 7 (45) |
| S2 | rate 0.5, e 10 | 13 (42) | — | — | — |
| S2 | rate 1, full | 31 (45) | 17 (45) | 29 (49) | 25 (48) |
| S2 | velocity cap 4 + holds, e 4 | 33 (46) | 32 (45) | 32 (47) | 26 (45) |
| F | rate 1, e 5 | 30 (48) | — | — | 33 |
| F | rate 0.5, e 10 | 35 (44) | — | — | — |

At the first close (median over failed / successful trials), distance to the object's centre:

| Policy, cell | Commanded tool centre point | Grip site | Grip site to commanded | Commanded height above the centre | Actual height | Object pushed before the close |
|---|---|---|---|---|---|---|
| F, rate 1, e 5 | 24 / 19 mm | 38 / 21 mm | 12 / 5 mm | 19 / 14 mm | 34 / 17 mm | 6 / 0 mm |
| F, rate 0.5, e 10 | 23 / 19 mm | 29 / 20 mm | 9 / 2 mm | 15 / 12 mm | 26 / 13 mm | 15 / 0 mm |
| S, rate 1, e 5 | 28 / 21 mm | 43 / 36 mm | 24 / 16 mm | 13 / 12 mm | 37 / 29 mm | 6 / 5 mm |
| S, rate 0.5, e 10 | 27 / 24 mm | 43 / 38 mm | 30 / 11 mm | 11 / 15 mm | 37 / 33 mm | 8 / 3 mm |
| S2, rate 1, e 5 | 27 / 26 mm | 36 / 42 mm | 16 / 17 mm | 14 / 20 mm | 27 / 37 mm | 20 / 2 mm |
| S2, rate 0.5, e 10 | 25 / 18 mm | 39 / 26 mm | 19 / 11 mm | 13 / 14 mm | 27 / 18 mm | 22 / 4 mm |

The commanded height at the close is the same for every policy; in failed trials S's and S2's grip site
stays 13–26 mm above it, at half speed as at full speed, a median 21–24 mm horizontally from the object's
centre against 10 mm for F's successful closes. Over the whole approach the grip site trails the commanded tool centre point by
1.2–1.7 frames of commanded motion for every policy. The commanded path does not dwell at the close pose for
S and S2 at e = 5 (0 frames within 3 mm before and after; F 1 and 1). The kinematic tool centre point is
placed in the world frame by an offset fitted at a 1–2-frame lag (residual median 2.3–4.3 mm).

**Label grid in the simulator.** The labels open loop (`vla_libero.replay --source labels`, kp 400, sync,
L = 0, e = 4, holds 0.2 / 0.55 s) over `curvature` span 16, 24, 32, 48 × event density 0, 1, 2 × velocity fraction
0.6, 0.8, 1.0 × cap 4, 8 (72 cells, `runs/libero_alphabet_soup/sim/label_grid/`). Successes of 50, mean over the
12 label configurations (range), and the mean of the cells' median success frames:

| | Fraction 0.6 | Fraction 0.8 | Fraction 1.0 |
|---|---|---|---|
| Cap 4 | 47.3 (45–48), 54 | 46.8 (45–48), 48 | 45.0 (41–48), 47 |
| Cap 8 | 46.6 (45–48), 51 | 45.0 (43–47), 41 | 38.3 (35–40), 36 |

Over all fractions and caps the span changes the mean by less than one success (16: 45.3, 24: 44.6, 32: 45.1,
48: 44.4) and the density by about one (0: 44.4, 1: 44.5, 2: 45.6); at cap 8, fraction 0.8, density 2 gives 46–47
against 43–45 for density 0. Span 32, density 1 at cap 4, fraction 1 gives 44, as measured before.

**Simulator pipeline checks** (scripts outside this repository; same task, kp 400, sync, L = 0).

- Observations (`obs_fidelity.py`): the simulator set to the recording's own state at 170 frames (16 before
  each first close up to it, episodes 3, 5, 12, 21, 34, 0, 10, 20, 30, 40) renders images that differ from
  the recorded ones by a mean absolute 4.4 (agentview) and 7.4 (wrist) of 255. The same checkpoint, state
  and noise on rendered against recorded images: tool-centre-point error of the points against the labels,
  median, point 1 / 5 / 10 — S 13.9 / 34.7 / 51.2 mm against 13.5 / 28.4 / 48.2 mm, S2 12.6 / 26.2 /
  61.1 against 12.9 / 24.8 / 52.3, F 2.4 / 9.7 / 15.9 against 2.4 / 9.3 / 16.1.
- Execution (`recorded_trace.py`, `gripper_trace.py` with measured joints, the grip site at the commanded
  joints from the simulator's own kinematics, and gripper contacts): the recordings' joint targets succeed
  48 of 50; at their close the grip site is 7 mm from the commanded one (joint gap 0.022 rad), 14 mm off the
  object's centre and 12 mm above it.
- Closed loop at the first close, successful / failed trials:

| Source | Successes | Commanded height above the centre | Actual height | Grip site to commanded | Joint gap | Touching the object in the 10 frames before |
|---|---|---|---|---|---|---|
| recordings | 48 | +7 / +25 mm | +12 / +26 mm | 7 / 3 mm | 0.022 / 0.010 rad | 10 / 0 |
| F, rate 1, e 5 | 33 | +13 / +14 | +16 / +27 | 4 / 13 | 0.021 / 0.023 | 15 / 12 |
| F, rate 0.5, e 10 | 33 | +11 / +13 | +14 / +22 | 3 / 8 | 0.011 / 0.011 | 15 / 11 |
| S, rate 1, e 5 | 15 | +18 / +15 | +33 / +40 | 21 / 25 | 0.026 / 0.029 | 12 / 32 |
| S, rate 0.5, e 10 | 14 | +19 / +13 | +34 / +39 | 10 / 27 | 0.018 / 0.030 | 9 / 34 |
| S2, rate 1, e 5 | 12 | +16 / +17 | +27 / +34 | 7 / 19 | 0.012 / 0.021 | 8 / 35 |
| S2, rate 0.5, e 10 | 14 | +15 / +11 | +31 / +26 | 21 / 22 | 0.017 / 0.020 | 14 / 34 |

  In failed trials S's and S2's gripper first touches the object with its grip site 45–48 mm above the
  object's centre (S at half speed: a median 12 frames before the close, fingers fully open at 80 mm) and
  stays 19–27 mm from the commanded grip site, at half speed as at full speed. F's grip site meets its command within 3–4 mm in
  successful trials although half of them touch the object on the way down at the same height.
- Rendering fails with "Offscreen framebuffer is not complete" when `MUJOCO_EGL_DEVICE_ID` names a device
  whose GPU holds a preallocating JAX policy; the EGL device numbering does not follow nvidia-smi's (device
  0 failed with nvidia-smi GPU 0 idle).

**Replanning more often.** Rate 1 with e = 1, 2, 3 (closed loop / labels): S 3 / 8 / 11 (37 / 41 / 42), S2
6 / 9 / 9 (43 / 45 / 45), F 0 / 7 / 18 (48 / 48 / 48); velocity cap 4 with holds, e = 1, 2: S 14 / 20, S2
22 / 17, F 38 / 39. The training data define state[t + 1] = action[t] exactly (0 difference over the recorded
rows); closed loop the joint controller covers about half of a commanded step per frame (F at e = 1: 0.0098 rad
commanded per frame, 0.005 rad left at the next observation), and each chunk is anchored on that lagging
state. With the commanded joints and command as the observation state, at rate 1: S 22 (e 5) / 6 (e 1) / 17
(full), S2 22 / 9 / 23, F 29 / 14 / 39; at velocity cap 4, e 4, every policy falls to 0–2 of 50.

**Span 32, event density 1, per-point statistics.** S3a: `curvature` span 32, event density 1 (keyframe windows
at the free part's density, the recorded pace at rate 1), otherwise S's labels and training (two L40S, 10k
steps; loss 0.0055 at step 9900). S3b: the same labels with action normalization statistics per point index
(`launch.norm_stats(per_point_actions=True)`; joint 6's q99 − q01 0.22 rad at point 1 to 1.53 rad at point 10
against 1.23 rad pooled; loss 0.0070 at step 9900). Open loop (held-out / training), point 1 against its label:
S3a 11 / 9 mm, S3b 10 / 8 mm (S 16 / 12, S2 14 / 12; F's point 3, the same lookahead, 7 / 6 mm). Labels open
loop, rate 1, e = 1 / 5 / full: 47 / 47 / 49. Closed loop at rate 1, e = 5 / full / e = 1 / e = 5 with the
commanded state: S3a 9 / 26 / 0 / 22, S3b 27 / 31 / 22 / 35.

Velocity timing with holds, measured state, e = 4 (successes; median completion frames of the successes; the
recordings take a median 152):

| Cap | S3a | S3b | F | S3 labels |
|---|---|---|---|---|
| 2 | 27 (165) | 33 (113) | — | 49 |
| 4, three runs | 33, 34, 33 (80, 73, 91) | 42, 41, 39 (64, 78, 66) | 42 (108); e 2: 39 (109) | 44 |
| 8 | 28 (58) | 43 (47) | — | 37 |
| 16 | 31 (49) | 40 (59) | — | 38 |
| 4, L = 0.1 s async | 31 (84) | 37 (61) | — | 46 |
| 8, L = 0.1 s async | 28 (72) | 32 (56) | — | 39 |

At cap 4 S averaged 30 and S2 31.5 of 50 over their runs.

**Matched comparison of the strategies.** Per-frame F at the recorded pace (uniform timing, rate 1, no holds)
against S3b with velocity timing and holds; task, the 50 initial states, kp 400, measured state and the
280-frame budget equal; three runs per cell (F at rate 1, e = 5: five), earlier runs under the same conditions
included (`runs/libero_alphabet_soup/sim/matched/`; servers allocate memory on demand). Successes of 50 per run,
their mean, and the successes' median completion frames (the recordings: 152):

| Latency | F at the recorded pace | Mean | Frames | S3b, velocity timing | Mean | Frames |
|---|---|---|---|---|---|---|
| 0 | e 10: 39, 36, 35 | 36.7 | 166 | cap 4: 42, 41, 39 | 40.7 | 73 |
| 0 | e 5: 31, 32, 28, 30, 33 | 30.8 | 199 | cap 8: 43, 37, 36 | 38.7 | 50 |
| 0.1 s async | e 8: 39, 38, 37 | 38.0 | 178 | cap 4: 37, 43, 39 | 39.7 | 65 |
| 0.1 s async | — | | | cap 8: 32, 35, 36 | 34.3 | 52 |
| 0.1 s sync | e 10: 33, 35, 30 | 32.7 | 204 | cap 4: 40, 38, 37 | 38.3 | 103 |

Labels open loop in the new cells: F e 10 48, e 8 at 0.1 s async 48, e 10 at 0.1 s sync 47; S3 cap 4 at 0.1 s
sync 44. F was trained with pooled action statistics on one GPU, S3b with per-point statistics on two.

Fpp: F's labels with per-point action statistics, trained on two L40S as S3b (loss 0.0058 at step 9900;
joint 6's q99 − q01 0.063 rad at point 1 against 0.456 pooled). At the recorded pace, same cells:

| Latency | Fpp at the recorded pace | Mean | Frames | S3b, velocity timing, cap 4 | Mean | Frames |
|---|---|---|---|---|---|---|
| 0 | e 10: 42, 45, 46 | 44.3 | 158 | 42, 41, 39 | 40.7 | 73 |
| 0 | e 5: 36, 43, 38 | 39.0 | 189 | — | | |
| 0.1 s async | e 8: 44, 42, 47 | 44.3 | 168 | 37, 43, 39 | 39.7 | 65 |
| 0.1 s sync | e 10: 40, 44, 40 | 41.3 | 189 | 40, 38, 37 | 38.3 | 103 |

Over the 150 trials of a cell, Fpp against S3b: 133 against 122 (z = 1.8), 133 against 119 (z = 2.2), 124 against
115 (z = 1.3); the three runs share the 50 initial states, so the trials are not independent. With the same
training recipe the per-frame policy at the recorded pace succeeds 6–10 points more often, and S3b completes
2.1–2.6 times sooner (1.5 times with 0.1 s sync latency).

**Data checks** (`label_data_check.py`, outside this repository). For S, S2 and F, the training datasets'
`action_chunk` rows equal the labels recomputed from the source recordings with each dataset's report (8772
rows, largest difference 1.2e-7). Through openpi's loader and training transforms, on 300 rows each: the
target's joints are the label's minus the row's `observation.state` (6e-8), its command the label's, the
loader's state the parquet row's; with the training norm stats the targets fall within −1.25…+1.19 (p0.5 /
p99.5), commands −1 / +1; the serving outputs give the labels back (6e-8).

training and serving from the openpi root with `examples/libero_object` on `PYTHONPATH`):

```bash
python -m vla_libero.replay --hdf5 <demo.hdf5> --source osc --out runs/<task>/sim/replay_osc.json
python -m vla_libero.replay --hdf5 <demo.hdf5> --source joint --kp 50 150 400 1000 --out runs/<task>/sim/replay_joint_kp.json
python -m vla_libero.replay --hdf5 <demo.hdf5> --source labels --kp 400 --dataset data/<src>-timed --spec runs/<task>/robot_spec.json \
    --keyframes runs/<task>/keyframes.json --label-config <label_config.json> --rate 16 --execute 4 --latency 0 --mode async \
    --velocity-fraction 1 --switch-hold 0.2 0.55 --out runs/<task>/sim/label_replay/<condition>.json
python examples/libero_object/run.py train --name <config> --data-root <data> --repo-id <dataset>-train --exp-name run
python examples/libero_object/run.py serve-points --name <config> --data-root <data> --repo-id <dataset>-train \
    --checkpoint checkpoints/<config>/run/9999 --port 8000
python -m vla_libero.evaluate --policy websocket --ports 8000 8001 --report <ckpt>/assets/<repo_id>/arc_resample.json \
    --spec runs/<task>/robot_spec.json --rate 16 --execute 4 --latency 0 --mode async --velocity-fraction 1 \
    --switch-hold 0.2 0.55 --kp 400 --workers 25 --out runs/<task>/sim/closed_loop/<condition>.json
```

Each server answers one request before `evaluate` starts (ISSUE-012).

## UR3 optical bench

Measured 2026-10-18 (data stages, training) and 2026-10-19 (open-loop evaluation) on
`LGG100/optical-bench-lerobot-cleaned`: dual-arm UR3, 14 dims, 15 Hz, 40 episodes, 9588 frames (176–388 per
episode), three AV1 cameras at 240 × 320, task "pull the metal rod out of the post holder and place it in
the white box". The left arm and its gripper never move (gripper open in 11 recordings, closed in 29); the
right arm closes and opens once per episode.

**Data stages.** The videos were decoded once into parquet-embedded PNG (`dataset_writer` refuses row
changes on video datasets; every stored image read back equal to its decoded frame). S0–S7 with the current
tools: 12 joint jumps kept as real fast turns, every still frame kept, 45 padding frames per episode,
keyframes grasp / pre-grasp stop / lift / release with 35 consistency flags accepted, holdout
[8, 15, 24, 27]. Labels `curvature` span 30, event density 2, H 10, anchor state, chord 0.1389 rad; train 36
episodes / 10191 rows (1620 padding), eval 40. Label replay at rate 1, e = 8: path deviation median / p90 /
max 0.0003 / 0.0032 / 0.092 rad, switch-pose error max 0.116 rad, no missed switch. Rate ceiling with the UR3
limits at L = 0.1 s asynchronous (90% of 37 episodes): uniform 0.89×, velocity timing 4.50× without holds,
3.80× with 0.533 / 0.667 s holds.

**Training.** `examples/ur3_optical_bench`: pi0.5 from `pi05_base`, SigLIP and Gemma 2B frozen (430M
trainable, 2923M frozen), no state input, joints and the left gripper as deltas, right gripper absolute,
quantile norm stats with dims 0–6 floored to a 0.05 range (`norm_floor`), batch 32 over two L40S
(`NCCL_P2P_DISABLE=1`), 10k steps in 5 h 01 min (1.7 s per step). Loss 0.2236 / 0.1293 / 0.0159 / 0.0061 /
0.0039 at steps 0 / 100 / 1000 / 5000 / 9900.

**Open loop.** `replay_eval` against `run.py serve` (rate 1) on the eval dataset, e = 8; worst arm joint in
rad, pooled median / p90 over real rows; one frame of motion 0.017 rad:

| | Held-out 8, 15, 24, 27 | Training 0, 10, 20, 30 |
|---|---|---|
| points vs label | 0.340 / 0.967 | 0.120 / 0.333 |
| head (first 8 targets) | 0.159 / 0.414 | 0.112 / 0.306 |
| stitched | 0.077 / 0.285 | 0.057 / 0.214 |
| hold (padding rows) | 0.005 / 0.125 | 0.004 / 0.007 |
| static (left arm), max | 1.2e-3 | 7.4e-4 |
| close shift, frames | +8, +5, +5, +5 | +2, +1, +7, +3 |
| open shift, frames | −1, −2, −4; 1 missed | −2, −8; 2 missed |
| round trip, median | 98 ms | 101 ms |

Stitched / points median by phase: held-out approach 0.071 / 0.215, carry (grasp to release)
0.081 / 0.457; training approach 0.064 / 0.113, carry 0.052 / 0.131. The first request to a server takes
15 s (compilation). Missed opens: ISSUE-014 (scoring), ISSUE-015 (model).

**Serving outside the harness.** A self-contained serving config (upstream openpi plus `arc_interp.py`,
published with the checkpoint) against `vla_openpi.launch.serving_config`, one loaded model, 6 recorded
frames, identical noise: per-frame targets (30, 14) float32, points (10, 14) and point frames identical
(maximum difference 0). Built in two separate processes, the two differ by up to 2.9e-3 rad; each
configuration repeats its own outputs exactly across processes.

**PyTorch conversion.** openpi's `convert_pi0_checkpoint` with the serving config's model config, bf16,
published with the JAX weights. The model has no LoRA, and the converted weights set every `PI0Pytorch`
parameter except the two `lm_head` matrices, which action prediction does not read. Through the
self-contained serving config, 16 recorded frames (1/3 and 2/3 of episodes 8, 15, 24, 27, 0, 10, 20, 30),
identical noise; largest joint difference over one answer's 10 points, median / max over the 16 (rad):

| Comparison | Joints | Right gripper commands that differ |
|---|---|---|
| JAX vs PyTorch, both fp32 (full fp32 matmuls) | 0.0004 / 0.0029 | 0 of 160 |
| JAX bf16 vs JAX fp32 | 0.0056 / 0.029 | 0 of 160 |
| PyTorch bf16 compiled vs PyTorch fp32 | 0.0072 / 0.062 | 0 of 160 |
| JAX bf16 vs PyTorch bf16, eager | 0.0073 / 0.043 | 0 of 160 |
| JAX bf16 vs PyTorch bf16, compiled (`max-autotune`, openpi's default) | 0.014 / 0.080 | 0 of 160 |
| JAX bf16, two noise draws | 0.089 / 0.43 | 4 of 160 |

`policy.infer` on one L40S: PyTorch compiled 62 ms median after 72 s loading and 25.6 s / 4.5 s for the
first two calls; eager 357 ms; JAX 84 ms.

**Commands**, from the project root (training and serving from the openpi root with
`examples/ur3_optical_bench` on `PYTHONPATH`):

```bash
python -m vla_tools.build_labels --dataset data/<src>-timed --spec runs/<task>/robot_spec.json --keyframes runs/<task>/keyframes.json \
    --config runs/<task>/label_config_curvature_s30_d2.json --exclude 8 15 24 27 --out data/<src>-curvature-s30-d2-train
python examples/ur3_optical_bench/run.py norm-stats --name <config> --data-root <data> --repo-id <dataset>-train
python -m vla_tools.norm_floor --stats <norm_stats.json> --key actions --dims 0 1 2 3 4 5 6 --min-range 0.05 --out <floored.json>
NCCL_P2P_DISABLE=1 python examples/ur3_optical_bench/run.py train --name <config> --data-root <data> --repo-id <dataset>-train --exp-name run
python examples/ur3_optical_bench/run.py serve --name <config> --data-root <data> --repo-id <dataset>-train \
    --checkpoint checkpoints/<config>/run/9999 --port 8100 --rate 1
python -m vla_tools.replay_eval --dataset data/<src>-curvature-s30-d2-eval --spec runs/<task>/robot_spec.json \
    --replay-spec runs/<task>/replay_spec.json --episodes 8 15 24 27 --execute 8 --port 8100 --out runs/<task>/replay_eval_holdout.json
```

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
- closed-loop results on a robot of any model trained through this harness; in simulation one task, one
  training budget and one `curvature` configuration (s64 d0) against per-frame labels;
- a training comparison between per-window and episode-wide steps;
- the rate ceiling on a robot: it holds the commanded targets to the Panda's nominal joint velocity limits
  in replay; tracking lag and tracking error (DROID's measured joints follow the commands about 4 frames
  later; LIBERO's joint controller 1–5 frames) and the controller's acceleration are not modelled; velocity
  timing is not passed through `vla_openpi.launch` (the simulator client applies it), and the switch hold on
  a robot needs the commanded gripper value as the start pose's command, not a lagging echo;
- the integration in a clean upstream openpi environment with the patches applied (every UR3 check above
  ran on a fork with equivalent changes).
