"""openpi TrainConfig for a dual-arm UR3 (14 dims: per arm 6 joints + binary gripper, 3 cameras).

pi0.5 from pi05_base with the VLM frozen (SigLIP and Gemma 2B): only the action expert (Gemma 300M) and the
action and time projections are trained, as in examples/libero_object. action_horizon 10, quantile
normalisation, joints as deltas from the state and grippers absolute, no state input. Each training sample's
prompt is drawn from its episode's instructions (meta/episodes.jsonl `tasks`). The full recipe: 20,000 steps (about 11 passes over
206 episodes) with an exponential moving average of the weights: a checkpoint's `params` are the averaged
weights. Steps, learning rate, warm-up and the EMA decay come from a recipe (recipes.py). The dataset is one written by vla_tools.build_labels; its control rate is carried by its arc_resample.json.
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
from vla_openpi import ZeroLoraBLoader
from vla_openpi import non_lora_filter
from vla_openpi import vlm_filter

from recipes import Recipe

ACTION_DIM = 14
ACTION_HORIZON = 10
CAMERAS = (
    ("observation/image", "observation.images.cam_high", "base_0_rgb"),
    ("observation/left_wrist_image", "observation.images.cam_left_wrist", "left_wrist_0_rgb"),
    ("observation/right_wrist_image", "observation.images.cam_right_wrist", "right_wrist_0_rgb"),
)


def train_config(
    name: str,
    data_root: pathlib.Path,
    repo_id: str,
    exp_name: str,
    absolute_joints: bool = False,
    state_input: bool = False,
    init_checkpoint: pathlib.Path | None = None,
    lora: bool = False,
    mix_repo_id: str | None = None,
    mix_fraction: tuple[float, float] = (0.0, 0.0),
    mix_skip_steps: int = 0,
    *,
    recipe: Recipe,
) -> _config.TrainConfig:
    """`absolute_joints`: the joints are predicted as absolute positions instead of deltas from the state.
    `state_input`: the normalised state is written into the prompt as 256-bin tokens (pi0.5's discrete state).
    `init_checkpoint`: a saved checkpoint step directory of an earlier run; training starts from its `params`
    instead of pi05_base (its model must have the same structure, so the same `state_input`).
    `lora`: fine-tune LoRA adapters of the action expert (attention and feed-forward, rank 32) on top of
    `init_checkpoint` and nothing else: SigLIP, Gemma 2B, the action expert's own weights and the projections stay
    frozen; the adapters start as a no-op (ZeroLoraBLoader).
    `mix_repo_id`, `mix_fraction`, `mix_skip_steps`: rows of a second dataset (`repo_id` is the primary one) are mixed
    into every batch with a fraction moving linearly from mix_fraction[0] to mix_fraction[1] over the recipe's steps
    (patch 0004); mix_skip_steps are the steps already trained when resuming.
    `recipe`: steps, learning rate, warm-up, EMA decay and kept checkpoints (recipes.py)."""
    if lora and init_checkpoint is None:
        raise ValueError("lora fine-tuning adapts a trained model: init_checkpoint is required")
    model = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=ACTION_HORIZON,
        discrete_state_input=state_input,
        action_expert_variant="gemma_300m_lora" if lora else "gemma_300m",
    )
    if lora:
        weight_loader: weight_loaders.WeightLoader = ZeroLoraBLoader(str(init_checkpoint / "params"))
    else:
        weight_loader = weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
            if init_checkpoint is None
            else str(init_checkpoint / "params")
        )
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
        delta_mask=() if absolute_joints else tuple(transforms.make_bool_mask(6, -1, 6, -1)),
        outputs=RawOutputs(action_dim=ACTION_DIM),
        rotate_prompts=True,
        mix_repo_id=mix_repo_id,
        mix_fraction=mix_fraction,
        mix_steps=recipe.num_train_steps if mix_repo_id is not None else 0,
        mix_skip_steps=mix_skip_steps,
    )
    return _config.TrainConfig(
        name=name,
        exp_name=exp_name,
        model=model,
        data=data,
        weight_loader=weight_loader,
        batch_size=32,
        num_workers=8,
        num_train_steps=recipe.num_train_steps,
        save_interval=1000,
        log_interval=50,
        keep_period=recipe.keep_period,
        wandb_enabled=False,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=recipe.warmup_steps,
            peak_lr=recipe.peak_lr,
            decay_steps=recipe.num_train_steps,
            decay_lr=recipe.decay_lr,
        ),
        optimizer=_optimizer.AdamW(b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-4, clip_gradient_norm=1.0),
        freeze_filter=non_lora_filter() if lora else vlm_filter(),
        ema_decay=recipe.ema_decay,
    )
