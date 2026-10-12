"""Convert a LoRA-finetuned openpi pi0 / pi0.5 JAX checkpoint to PyTorch, folding the adapters in first.

openpi's examples/convert_jax_model_to_pytorch.py maps only base weight names and loads them with
strict=False, so every lora_a / lora_b is dropped without an error: the result is the base model with
fine-tuned projection heads. The PyTorch model has no LoRA (it reads the variant name only for layer
sizes), so the adapters are folded into the weights they adapt by `lora_merge` -- the exact sum the JAX
forward pass computes -- and the merged tree is handed to that converter.

The converter copies assets from the parent of the checkpoint directory; this copies <step>/assets.

Runs in the openpi environment, with openpi's examples/ directory on PYTHONPATH:
    PYTHONPATH=<openpi>/examples uv run python -m vla_tools.lora_to_pytorch \
        --checkpoint-dir <run>/<step> --config-name <train_config> --output-path <run>/<step>_pytorch
Verify the result with policy_parity before using it.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import convert_jax_model_to_pytorch as converter
import numpy as np
from flax.nnx import traversals
from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models import pi0_config as _pi0_config
from openpi.training import config as _config

from vla_tools.lora_merge import AdapterScaling
from vla_tools.lora_merge import merge_lora


def expert_scalings(model_config: _pi0_config.Pi0Config) -> tuple[AdapterScaling, AdapterScaling]:
    """Rank and output scaling of each expert's attention adapter, from the model's variants."""
    configs = [
        _gemma.get_config(model_config.paligemma_variant).lora_configs["attn"],
        _gemma.get_config(model_config.action_expert_variant).lora_configs["attn"],
    ]
    return (
        AdapterScaling(rank=configs[0].rank, scaling=configs[0].scaling_value),
        AdapterScaling(rank=configs[1].rank, scaling=configs[1].scaling_value),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--precision", choices=["float32", "bfloat16"], default="bfloat16")
    args = parser.parse_args()

    model_config = _config.get_config(args.config_name).model
    params = _model.restore_params(f"{args.checkpoint_dir}/params/", restore_type=np.ndarray, dtype="float32")
    params, merged = merge_lora(params, expert_scalings(model_config))
    print(f"merged {len(merged)} adapters: " + ", ".join(merged))
    converter.slice_initial_orbax_checkpoint = lambda **_: {
        "paligemma_params": traversals.flatten_mapping(params["PaliGemma"], sep="/"),
        "projection_params": params,
    }
    converter.convert_pi0_checkpoint(str(args.checkpoint_dir), args.precision, str(args.output_path), model_config)
    assets = args.checkpoint_dir / "assets"
    if assets.is_dir():
        shutil.copytree(assets, args.output_path / "assets", dirs_exist_ok=True)


if __name__ == "__main__":
    main()
