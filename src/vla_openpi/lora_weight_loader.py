"""Initial weights for LoRA fine-tuning of a model that was trained without adapters."""

from __future__ import annotations

import dataclasses

import flax.traverse_util
import numpy as np
from openpi.shared import array_typing as at
from openpi.training import weight_loaders

LORA_B_SUFFIX = "lora_b"


@dataclasses.dataclass(frozen=True)
class ZeroLoraBLoader(weight_loaders.WeightLoader):
    """Loads the saved params of a trained checkpoint (`params_path`) into a model that has LoRA adapters.

    The checkpoint holds no adapters, so openpi's CheckpointWeightLoader leaves them to the model's own
    initialisation, which draws both factors of an adapter from a normal distribution: the adapted model would
    start slightly different from the checkpoint. Here every `lora_b` factor is initialised to zero, so the
    adapters add nothing at step 0 and training starts from the checkpoint's outputs.
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        merged = weight_loaders.CheckpointWeightLoader(self.params_path).load(params)
        flat = flax.traverse_util.flatten_dict(merged, sep="/")
        zeroed = {
            key: np.zeros(value.shape, dtype=value.dtype) if key.endswith(LORA_B_SUFFIX) else value
            for key, value in flat.items()
        }
        return flax.traverse_util.unflatten_dict(zeroed, sep="/")
