"""openpi TrainConfig for a LIBERO Panda task in joint space (8 dims: 7 joints + binary gripper, 2 cameras, 20 Hz).

pi0.5 from pi05_base with the VLM frozen (SigLIP and Gemma 2B): only the action expert (Gemma 300M) and the
action and time projections are trained. action_horizon 10, quantile normalisation, joints as deltas and the
gripper absolute. No state input (discrete_state_input=False), as openpi's pi05_libero. The dataset is one
written by vla_tools.build_labels from a LIBERO recording converted to joint space (images rotated 180°).
"""

from __future__ import annotations

import pathlib

import flax.nnx as nnx
from openpi import transforms
from openpi.models import pi0_config
from openpi.shared import nnx_utils
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders

from vla_openpi import ArcInputs
from vla_openpi import LeRobotArcDataConfig
from vla_openpi import RawOutputs

ACTION_DIM = 8
ACTION_HORIZON = 10
NUM_TRAIN_STEPS = 10_000
CAMERAS = (
    ("observation/image", "observation.images.agentview", "base_0_rgb"),
    ("observation/wrist_image", "observation.images.eye_in_hand", "left_wrist_0_rgb"),
)


def vlm_filter() -> nnx.filterlib.Filter:
    """Parameters of the VLM: SigLIP (img) and every Gemma parameter that is not the action expert's (suffix _1)."""
    gemma_2b = nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")))
    return nnx.All(nnx.Param, nnx.Any(nnx_utils.PathRegex(".*img.*"), gemma_2b))


def train_config(name: str, data_root: pathlib.Path, repo_id: str, exp_name: str) -> _config.TrainConfig:
    model = pi0_config.Pi0Config(pi05=True, action_horizon=ACTION_HORIZON, discrete_state_input=False)
    data = LeRobotArcDataConfig(
        repo_id=repo_id,
        base_config=_config.DataConfig(local_root=data_root, prompt_from_task=True, action_sequence_keys=("action",)),
        observation_columns=(
            *((key, column) for key, column, _ in CAMERAS),
            ("observation/state", "observation.state"),
        ),
        inputs=ArcInputs(
            cameras=tuple((key, camera) for key, _, camera in CAMERAS),
            action_dim=ACTION_DIM,
            action_horizon=ACTION_HORIZON,
        ),
        delta_mask=tuple(transforms.make_bool_mask(7, -1)),
        outputs=RawOutputs(action_dim=ACTION_DIM),
    )
    return _config.TrainConfig(
        name=name,
        exp_name=exp_name,
        model=model,
        data=data,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        batch_size=32,
        num_workers=8,
        num_train_steps=NUM_TRAIN_STEPS,
        save_interval=1000,
        log_interval=100,
        keep_period=5000,
        wandb_enabled=False,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=2.5e-5, decay_steps=NUM_TRAIN_STEPS, decay_lr=2.5e-6
        ),
        optimizer=_optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-4, clip_gradient_norm=1.0),
        freeze_filter=vlm_filter(),
        ema_decay=None,
    )
