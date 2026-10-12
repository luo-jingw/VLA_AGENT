# Validation

What has been observed, and on what. Every check ran on one dual-arm UR3 dataset (22 episodes, 3223
frames, 15 Hz, parquet-embedded PNG images) in an openpi fork whose model, loader and config changes are
equivalent to `integrations/openpi/patches`.

## vla_tools

| Tool | Check | Observed |
|---|---|---|
| `arc_core` + `build_labels` | rebuild a dataset that a robot-specific builder had produced (curvature, span 20, 3 points per gripper window, action anchor) | `action_chunk`, `action_loss_weight`, indices and observations identical element by element |
| `arc_core` | `time`, span = H, `anchor="state"` labels vs `action[t:t+H]` | max difference 4.4e-9 over 3003 frames |
| `arc_interp` | against an independently written per-robot interpolator, all 3223 chunks | difference 0 |
| `arc_interp.from_report` | outputs at the point frames vs the recording, `time` span 20 | max difference 4.4e-16 |
| `dataset_writer` | row drops, end padding, exclusion with renumbering | rows, image bytes, `frame_index`, `index`, `timestamp`, `episodes.jsonl` consistent; LeRobot loads the result |
| `audit` | jump classification vs inspection of joint traces and camera frames | 4 of 4 jumps classified "continuing", matching inspection |
| `clean` | a continuing jump without a decision | refuses to write, lists it, exit code 2 |
| `keyframe_detect` | `column` signal on the action column vs `action` signal | identical keyframes |
| `episode_timing` | audit after 20 rows of end padding | same findings as before padding; keyframe window clipping resolved |
| `replay_eval` | against an earlier replay script, same model and episodes | same errors (rad) and gripper shifts |
| `lora_to_pytorch` | merged attention deltas vs openpi `lora.Einsum` on random weights | ≤ 2e-3 for all four equations (float32 default matmul precision); a per-head `lora_a @ lora_b` is off by 8.6e-2 on the output projection |
| `lora_to_pytorch` | merged weights in a non-LoRA JAX model vs the LoRA JAX model, same observations and noise | joints 2.2e-3 rad median, 3.5e-3 max; all 160 gripper values equal |
| `lora_to_pytorch` + `policy_parity` | converted PyTorch model vs JAX, 16 frames, same observations and noise | joints 2.8e-3 rad median, 8.7e-3 max; all 160 gripper values equal |
| `policy_parity` | a conversion with the per-head merge | joints 0.083 rad median, 0.14 max; 4 of 160 gripper values differ: detected |

## integrations/openpi

| Check | Observed |
|---|---|
| patches 0001–0003 on upstream openpi `15a9616` | apply in order with `git apply` |
| `LeRobotArcDataConfig` + `ArcInputs` + `RawOutputs` vs a hand-written robot config | identical repack structure, norm stats, model transforms; inputs (state, label, loss weight, images, masks) and outputs identical on 4 sampled frames |
| `vla_openpi.launch.norm_stats` + `launch.train` (3 steps, batch 4) on a `build_labels` dataset | norm stats written; `action_loss_weight` reaches the model as (4, 10); finite losses; checkpoint saved |
| `vla_openpi.launch.serving_config` vs a hand-written serving config, same checkpoint, observations and noise | per-frame targets and points identical over 4 frames; output (21, 14) |

## Not yet observed

- video datasets: `FrameReader`'s video path, and the writer's refusal of row edits on them;
- the `settle` and referenced `threshold_cross` detectors on real data;
- `diversity` cluster cuts on a second dataset;
- a full training run started from `vla_openpi.launch`, and closed-loop robot results of any model
  trained through this harness.
