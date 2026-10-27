"""A robot-agnostic openpi DataConfigFactory for LeRobot datasets with a resampled label.

Requires patches 0001 (per-step loss weight) and, for datasets outside the HF cache, 0002 (local
dataset root) applied to openpi.

    observation_columns  (repacked key, dataset column) pairs for cameras and state; the label and its
                         weights are always read from `action_chunk` / `action_loss_weight`
    inputs               ArcInputs for this robot
    delta_mask           True for dims trained as deltas from the state (joints), False for absolute
                         ones (binary commands); as for a per-frame config of the same robot
    outputs              RawOutputs for training and evaluation of points, ArcOutputs for serving
    rotate_prompts       each training sample's prompt drawn from its episode's `tasks` in meta/episodes.jsonl
                         (EpisodePrompt) instead of its `task_index`; needs prompt_from_task=False and
                         local_root in the base config

`action_sequence_keys` in the base config must still name a per-frame action column that exists in
the dataset (e.g. ("action",)): the loader builds `delta_timestamps` from it. That slice is not used
as the label.
"""

from __future__ import annotations

import dataclasses
import pathlib

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.training import config as _config
from typing_extensions import override

from vla_openpi.arc_transforms import ArcInputs
from vla_openpi.episode_prompt import EpisodePrompt


@dataclasses.dataclass(frozen=True)
class LeRobotArcDataConfig(_config.DataConfigFactory):
    observation_columns: tuple[tuple[str, str], ...] = ()
    inputs: ArcInputs | None = None
    delta_mask: tuple[bool, ...] = ()
    outputs: _transforms.DataTransformFn | None = None
    rotate_prompts: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        if self.inputs is None or self.outputs is None or not self.observation_columns:
            raise ValueError("LeRobotArcDataConfig needs observation_columns, inputs and outputs")
        base = self.create_base_config(assets_dirs, model_config)
        structure = dict(self.observation_columns) | {
            "actions": "action_chunk",
            "action_loss_weight": "action_loss_weight",
        }
        repack: list[_transforms.DataTransformFn] = []
        if self.rotate_prompts:
            if base.prompt_from_task:
                raise ValueError("rotate_prompts replaces prompt_from_task; set prompt_from_task=False")
            if base.local_root is None:
                raise ValueError("rotate_prompts reads meta/episodes.jsonl under local_root; set local_root")
            structure["episode_index"] = "episode_index"
            repack.append(EpisodePrompt(base.local_root / base.repo_id))
        else:
            structure["prompt"] = "prompt"
        data_transforms = _transforms.Group(inputs=[self.inputs], outputs=[self.outputs])
        if self.delta_mask:
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(self.delta_mask)],
                outputs=[_transforms.AbsoluteActions(self.delta_mask)],
            )
        return dataclasses.replace(
            base,
            repack_transforms=_transforms.Group(inputs=[_transforms.RepackTransform(structure), *repack]),
            data_transforms=data_transforms,
            model_transforms=_config.ModelTransformFactory()(model_config),
        )
