# openpi integration

What openpi needs in order to train and serve on datasets built by `vla_tools.build_labels`.

## Patches

Against upstream openpi (checked to apply in order on commit `15a9616`):

| Patch | Needed for | Change |
|---|---|---|
| `0001-per-step-loss-weight.patch` | training with `key_weight` / `pad_weight` | `Observation.action_loss_weight`; `Pi0.compute_loss` multiplies the per-step loss by it. JAX only: the PyTorch training path does not take weights |
| `0002-local-dataset-root.patch` | datasets outside the Hugging Face cache | `DataConfig.local_root`; the loader reads `local_root / repo_id` |
| `0003-pyav-video-backend.patch` | optional | forces LeRobot's PyAV video backend where torchcodec imports but cannot load FFmpeg |

```bash
cd <openpi>
git apply <VLA_AGENT>/integrations/openpi/patches/0001-per-step-loss-weight.patch
git apply <VLA_AGENT>/integrations/openpi/patches/0002-local-dataset-root.patch
```

## Transforms and data config

`vla_openpi` (installed with this repository, `pip install -e .`, into the openpi environment):

| Object | Role |
|---|---|
| `ArcInputs` | robot observation → model inputs; reshapes the flat label to (H, D) and passes the loss weight through |
| `RawOutputs` | the model's (H, D) points, for training and evaluation of points |
| `ArcOutputs` | per-frame targets via `vla_tools.arc_interp`, for serving |
| `LeRobotArcDataConfig` | an openpi `DataConfigFactory` assembling the above |

Register a training config in openpi's `config.py` (or build it in your own module and pass it to
openpi's training entry point):

```python
from vla_openpi import ArcInputs, LeRobotArcDataConfig, RawOutputs

TrainConfig(
    name="pi05_<robot>_<labels>",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=H, ...),
    data=LeRobotArcDataConfig(
        repo_id="<dataset>-train",
        base_config=DataConfig(local_root=pathlib.Path("<data root>"), prompt_from_task=True,
                               action_sequence_keys=("action",)),
        observation_columns=(("observation/image", "observation.images.<top camera>"),
                             ("observation/state", "observation.state")),
        inputs=ArcInputs(cameras=(("observation/image", "base_0_rgb"),), action_dim=D, action_horizon=H),
        delta_mask=tuple(transforms.make_bool_mask(...)),
        outputs=RawOutputs(action_dim=D),
    ),
    ...
)
```

Then, from the openpi root:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_<robot>_<labels>
python -m vla_tools.norm_floor ...            # static dims, see the vla-downstream skill
WANDB_MODE=offline uv run scripts/train.py pi05_<robot>_<labels> --exp-name <run>
cp <dataset>-train/meta/arc_resample.json checkpoints/<config>/<run>/<step>/assets/<asset_id>/
```

openpi saves only the norm stats into a checkpoint's assets; the last line puts the label settings next
to them, where the server reads them.

## Serving

Swap the outputs for `ArcOutputs` built from the checkpoint's report, then serve as usual:

```python
interp = ArcInterpolator.from_report(ckpt / "assets" / asset_id / "arc_resample.json")
config = dataclasses.replace(train_config, data=dataclasses.replace(train_config.data, outputs=ArcOutputs(interp)))
policy = policy_config.create_trained_policy(config, ckpt)
websocket_policy_server.WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=8000,
                                              metadata=policy.metadata).serve_forever()
```

`from_report(path, rate)` takes the execution rate, a deployment choice that `arc_resample.json` does not
record: point k is placed at (offset + k·span/H)/rate frames, so rate 1 executes a chunk at the recording's
average pace and rate r traverses the same interpolated path r times faster (⌊span/rate⌋ targets per chunk).
`examples/ur3_dual_arm/run.py serve --rate <r>` is a complete instance (`vla_openpi.launch.serve(..., rate)`).

## Verified

On a dual-arm UR3 dataset, a config assembled from these classes matches a hand-written robot-specific
config: identical repack structure, norm stats and model transforms; identical inputs (state, label,
loss weight, images, masks) and outputs on sampled frames (max difference 0).
