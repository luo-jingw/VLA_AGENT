"""Convert a LoRA-finetuned openpi pi0 / pi0.5 JAX checkpoint to PyTorch, folding the adapters in first.

openpi's examples/convert_jax_model_to_pytorch.py maps only base weight names and loads them with
strict=False, so every lora_a / lora_b is dropped without an error: the result is the base model with
fine-tuned projection heads. The PyTorch model has no LoRA (it reads the variant name only for layer
sizes), so the adapters are folded into the weights they adapt -- the exact sum the JAX forward pass
computes (openpi/models/lora.py) -- and the merged tree is handed to that converter.

    Einsum (attention)   w + scaling * delta, scaling = alpha / rank (alpha / sqrt(rank) under rslora)
    FeedForward          w + lora_a @ lora_b; openpi's FeedForward._dot applies no scaling

The attention delta is NOT a per-head lora_a @ lora_b. lora.Einsum derives two equations from the
layer's own equation, and a label of lora_b that the first equation's output lacks is summed inside
lora_b. For the output projection "BTNH,NHD->BTD" the first equation is "BTNH,NHL->BTL", so lora_b is
summed over heads: delta[N,H,D] = sum_L lora_a[N,H,L] * sum_M lora_b[M,L,D]. `einsum_delta` derives
the delta from the equation the same way, for every attention einsum.

An adapter without suffix belongs to expert 0 (the VLM), one ending "_1" to expert 1 (the action expert).
The converter copies assets from the parent of the checkpoint directory; this copies <step>/assets.

Runs in the openpi environment, with openpi's examples/ directory on PYTHONPATH next to this package:
    PYTHONPATH=<openpi>/examples:<tools> uv run python -m vla_tools.lora_to_pytorch \
        --checkpoint-dir <run>/<step> --config-name <train_config> --output-path <run>/<step>_pytorch
Verify the result with policy_parity before using it.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import convert_jax_model_to_pytorch as converter
import numpy as np
from flax import traverse_util
from flax.nnx import traversals
from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models import pi0_config as _pi0_config
from openpi.training import config as _config

# The equation each attention einsum is called with, from openpi/models/gemma.py.
ATTENTION_EQUATIONS = {
    "qkv_einsum": "BSD,3KDH->3BSKH",
    "q_einsum": "BTD,NDH->BTNH",
    "kv_einsum": "BSD,2KDH->2BSKH",
    "attn_vec_einsum": "BTNH,NHD->BTD",
}
FFN_LORA = {
    "gating_einsum": ("gating_einsum_lora_a", "gating_einsum_lora_b"),
    "linear": ("linear_lora_a", "linear_lora_b"),
}


def einsum_delta(equation: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The weight change lora.Einsum adds for `equation`, with axes (-2, -1) and a leading layer axis.

    Mirrors lora.Einsum._make_lora_eqns: eqn_a = lhs,a_rhs->a_out and eqn_b = a_out,b_rhs->out. A label
    of b_rhs in neither a_out nor out is contracted inside lora_b alone, so it gets its own letter here.
    """
    lhs, rhs, out = re.match("(.*),(.*)->(.*)", equation).groups()
    digits = {c: "PQRS"[i] for i, c in enumerate(sorted({c for c in equation if c.isdigit()}))}
    lhs, rhs, out = ("".join(digits.get(c, c) for c in part) for part in (lhs, rhs, out))
    a_label, b_label = rhs[-2], rhs[-1]
    a_rhs = rhs.replace(b_label, "L")
    a_out = out.replace(b_label, "L")
    b_rhs = rhs.replace(a_label, "L")
    fresh = iter("UVWXY")
    b_rhs = "".join(c if (c in a_out or c in out) else next(fresh) for c in b_rhs)
    return np.einsum(f"Z{a_rhs},Z{b_rhs}->Z{rhs}", a, b)


def merge_lora(params: dict, model_config: _pi0_config.Pi0Config) -> tuple[dict, list[str]]:
    variants = [
        _gemma.get_config(model_config.paligemma_variant),
        _gemma.get_config(model_config.action_expert_variant),
    ]
    flat = traverse_util.flatten_dict(params)
    merged: list[str] = []
    for key in [k for k in flat if k[-1] == "lora_a"]:
        module, prefix = key[-2], key[:-1]
        config = variants[1 if module.endswith("_1") else 0].lora_configs["attn"]
        a, b = flat.pop(key), flat.pop((*prefix, "lora_b"))
        if a.shape[-1] != config.rank:
            raise ValueError(f"{'/'.join(key)}: rank {a.shape[-1]}, variant says {config.rank}")
        equation = ATTENTION_EQUATIONS[module.removesuffix("_1")]
        flat[(*prefix, "w")] = flat[(*prefix, "w")] + config.scaling_value * einsum_delta(equation, a, b)
        merged.append(f"{'/'.join(prefix)} x{config.scaling_value:g}")
    for key in [k for k in flat if k[-1] in FFN_LORA]:
        a_key, b_key = ((*key[:-1], name) for name in FFN_LORA[key[-1]])
        if a_key in flat:
            flat[key] = flat[key] + np.matmul(flat.pop(a_key), flat.pop(b_key))
            merged.append(f"{'/'.join(key)} (ffn)")
    leftover = [k for k in flat if any("lora" in part for part in k)]
    if leftover:
        raise ValueError(f"unmerged LoRA weights: {leftover}")
    return traverse_util.unflatten_dict(flat), merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--precision", choices=["float32", "bfloat16"], default="bfloat16")
    args = parser.parse_args()

    model_config = _config.get_config(args.config_name).model
    params = _model.restore_params(f"{args.checkpoint_dir}/params/", restore_type=np.ndarray, dtype="float32")
    params, merged = merge_lora(params, model_config)
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
