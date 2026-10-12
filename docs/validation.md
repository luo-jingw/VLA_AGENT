# Validation

Two kinds of evidence back the tools. Observations on real data were made once, on one dual-arm UR3
dataset (22 episodes, 3223 frames, 15 Hz, parquet-embedded PNG images), in an openpi fork whose model,
loader and config changes are equivalent to `integrations/openpi/patches`. The test suite re-checks the
mechanisms on synthetic data on every run.

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

## vla_tools

| Tool | Check | Observed on UR3 | Re-checked by |
|---|---|---|---|
| `arc_core` + `build_labels` | rebuild a dataset that a robot-specific builder had produced (curvature, span 20, 3 points per gripper window, action anchor) | `action_chunk`, `action_loss_weight`, indices and observations identical element by element | — (the robot-specific builder is not part of the repository) |
| `arc_core` | `time`, span = H labels vs `action[t:t+H]`, both anchors | max difference 4.4e-9 over 3003 frames (`anchor="state"`) | `test_arc_core.py::test_time_with_span_equal_to_horizon_is_the_recorded_chunk`; `test_pipeline_chain.py::test_sweep_frame_baseline_is_exact_and_largest_admissible_span_is_selected` |
| `arc_core` | padding and keyframe-window weights; event density inside windows; monotone frames under `arc` and `curvature`; a recorded config without `anchor` | — | `test_arc_core.py` |
| `arc_interp` | against an independently written per-robot interpolator, all 3223 chunks | difference 0 | `test_arc_interp.py::test_to_frames_passes_through_the_points_and_takes_the_nearest_command` (the interpolator's rules, not the comparison) |
| `arc_interp.from_report` | outputs at the point frames vs the recording, `time` span 20 | max difference 4.4e-16 | `test_arc_interp.py::test_from_report_reproduces_the_recording_at_the_point_frames`; `test_pipeline_chain.py::test_train_and_eval_label_datasets` |
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
| `sweep`, `label_replay` | frame baseline exact; largest admissible span selected; one row per replan interval | — | `test_pipeline_chain.py` |
| `replay_eval` | against an earlier replay script, same model and episodes | same errors (rad) and gripper shifts | — (needs a served policy) |
| `lora_merge` | merged attention deltas vs the LoRA forward pass | ≤ 2e-3 against openpi `lora.Einsum` for all four equations (float32 default matmul precision, GPU); a per-head `lora_a @ lora_b` is off by 8.6e-2 on the output projection | `test_lora_merge.py`; `test_openpi_integration.py::test_merged_weight_matches_openpi_lora_einsum` |
| `lora_to_pytorch` | merged weights in a non-LoRA JAX model vs the LoRA JAX model, same observations and noise | joints 2.2e-3 rad median, 3.5e-3 max; all 160 gripper values equal | — (needs a LoRA checkpoint) |
| `lora_to_pytorch` + `policy_parity` | converted PyTorch model vs JAX, 16 frames, same observations and noise | joints 2.8e-3 rad median, 8.7e-3 max; all 160 gripper values equal | — (needs a LoRA checkpoint) |
| `policy_parity` | a conversion with the per-head merge | joints 0.083 rad median, 0.14 max; 4 of 160 gripper values differ: detected | — (needs a LoRA checkpoint) |

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
- `diversity` cluster cuts on a second real dataset;
- a full training run started from `vla_openpi.launch`, and closed-loop robot results of any model
  trained through this harness;
- the integration in a clean upstream openpi environment with the patches applied (every UR3 check above
  ran on a fork with equivalent changes).
