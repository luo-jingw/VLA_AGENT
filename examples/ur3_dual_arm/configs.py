"""openpi TrainConfig for a dual-arm UR3 (14 dims: per arm 6 joints + binary gripper, 3 cameras).

pi0.5 from pi05_base with the VLM frozen (SigLIP and Gemma 2B): only the action expert (Gemma 300M) and the
action and time projections are trained, as in examples/libero_object. action_horizon 10, quantile
normalisation, joints as deltas from the state and grippers absolute, no state input. Each training sample's
prompt is drawn from its episode's instructions (meta/episodes.jsonl `tasks`). The dataset is one written by
vla_tools.build_labels; its control rate is carried by its arc_resample.json.
"""

from __future__ import annotations

import pathlib

from openpi import transforms
from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders

from vla_openpi import ArcInputs
from vla_openpi import LeRobotArcDataConfig
from vla_openpi import RawOutputs
from vla_openpi import vlm_filter

ACTION_DIM = 14
ACTION_HORIZON = 10
CAMERAS = (
    ("observation/image", "observation.images.cam_high", "base_0_rgb"),
    ("observation/left_wrist_image", "observation.images.cam_left_wrist", "left_wrist_0_rgb"),
    ("observation/right_wrist_image", "observation.images.cam_right_wrist", "right_wrist_0_rgb"),
)


def train_config(name: str, data_root: pathlib.Path, repo_id: str, exp_name: str) -> _config.TrainConfig:
    model = pi0_config.Pi0Config(pi05=True, action_horizon=ACTION_HORIZON, discrete_state_input=False)
    data = LeRobotArcDataConfig(
        repo_id=repo_id,
        base_config=_config.DataConfig(local_root=data_root, prompt_from_task=False, action_sequence_keys=("action",)),
        observation_columns=(
            *((key, column) for key, column, _ in CAMERAS),
            ("observation/state", "observation.state"),
        ),
        inputs=ArcInputs(
            cameras=tuple((key, camera) for key, _, camera in CAMERAS),
            action_dim=ACTION_DIM,
            action_horizon=ACTION_HORIZON,
        ),
        delta_mask=tuple(transforms.make_bool_mask(6, -1, 6, -1)),
        outputs=RawOutputs(action_dim=ACTION_DIM),
        rotate_prompts=True,
    )
    return _config.TrainConfig(
        name=name,
        exp_name=exp_name,
        model=model,
        data=data,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        batch_size=32,
        num_workers=8,
        num_train_steps=10_000,
        save_interval=1000,
        log_interval=50,
        keep_period=5000,  # keeps the half-way (5000) checkpoint besides the latest
        wandb_enabled=False,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=2.5e-5, decay_steps=10_000, decay_lr=2.5e-6
        ),
        optimizer=_optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-4, clip_gradient_norm=1.0),
        freeze_filter=vlm_filter(),
        ema_decay=None,
    )
