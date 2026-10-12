"""Fold openpi LoRA adapters into the weights they adapt. numpy only; the openpi-facing CLI is lora_to_pytorch.

openpi's attention layers are `lora.Einsum` modules whose adapter output is computed with two equations
derived from the layer's own equation (`lora.Einsum._make_lora_eqns`):

    eqn_a = lhs,a_rhs->a_out        a_rhs, a_out: the layer's output-side label replaced by L
    eqn_b = a_out,b_rhs->out        b_rhs: the layer's input-side label replaced by L

A label of b_rhs that is in neither a_out nor out is contracted inside lora_b alone. For the output
projection "BTNH,NHD->BTD" that is the head label N: its adapter is a @ (sum over heads of b), not a
per-head a @ b. `einsum_delta` derives the weight change from the equation, so every attention einsum
is folded exactly. Feed-forward adapters (`lora.FeedForward`) are plain x @ a @ b and are not scaled.

Parameter trees use openpi's nesting; a leading layer axis (from nn.scan) is carried through.
"""

from __future__ import annotations

import dataclasses
import re

import numpy as np

# The equation each attention einsum is called with, from openpi/models/gemma.py.
ATTENTION_EQUATIONS: dict[str, str] = {
    "qkv_einsum": "BSD,3KDH->3BSKH",
    "q_einsum": "BTD,NDH->BTNH",
    "kv_einsum": "BSD,2KDH->2BSKH",
    "attn_vec_einsum": "BTNH,NHD->BTD",
}
FFN_LORA: dict[str, tuple[str, str]] = {
    "gating_einsum": ("gating_einsum_lora_a", "gating_einsum_lora_b"),
    "linear": ("linear_lora_a", "linear_lora_b"),
}


@dataclasses.dataclass(frozen=True)
class AdapterScaling:
    """One expert's attention adapter: rank and the factor applied to its output (alpha / rank)."""

    rank: int
    scaling: float


def lora_equations(equation: str) -> tuple[str, str]:
    """The two equations lora.Einsum uses for `equation` (axes (-2, -1), label L)."""
    lhs, rhs, out = re.match("(.*),(.*)->(.*)", equation).groups()
    a_label, b_label = rhs[-2], rhs[-1]
    a_rhs, a_out = rhs.replace(b_label, "L"), out.replace(b_label, "L")
    b_rhs = rhs.replace(a_label, "L")
    return f"{lhs},{a_rhs}->{a_out}", f"{a_out},{b_rhs}->{out}"


def einsum_delta(equation: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The weight change the adapter adds to `equation`'s weight; a and b carry a leading layer axis."""
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


def _flatten(tree: dict, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], np.ndarray]:
    out: dict[tuple[str, ...], np.ndarray] = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            out.update(_flatten(value, (*prefix, key)))
        else:
            out[(*prefix, key)] = value
    return out


def _unflatten(flat: dict[tuple[str, ...], np.ndarray]) -> dict:
    tree: dict = {}
    for key, value in flat.items():
        node = tree
        for part in key[:-1]:
            node = node.setdefault(part, {})
        node[key[-1]] = value
    return tree


def merge_lora(params: dict, experts: tuple[AdapterScaling, AdapterScaling]) -> tuple[dict, list[str]]:
    """Return the tree with every adapter folded in, and the list of folded modules.

    experts[0] is the VLM (modules without suffix), experts[1] the action expert (suffix "_1").
    """
    flat = _flatten(params)
    merged: list[str] = []
    for key in [k for k in flat if k[-1] == "lora_a"]:
        module, prefix = key[-2], key[:-1]
        expert = experts[1 if module.endswith("_1") else 0]
        a, b = flat.pop(key), flat.pop((*prefix, "lora_b"))
        if a.shape[-1] != expert.rank:
            raise ValueError(f"{'/'.join(key)}: rank {a.shape[-1]}, expected {expert.rank}")
        equation = ATTENTION_EQUATIONS[module.removesuffix("_1")]
        flat[(*prefix, "w")] = flat[(*prefix, "w")] + expert.scaling * einsum_delta(equation, a, b)
        merged.append(f"{'/'.join(prefix)} x{expert.scaling:g}")
    for key in [k for k in flat if k[-1] in FFN_LORA]:
        a_key, b_key = ((*key[:-1], name) for name in FFN_LORA[key[-1]])
        if a_key in flat:
            flat[key] = flat[key] + np.matmul(flat.pop(a_key), flat.pop(b_key))
            merged.append(f"{'/'.join(key)} (ffn)")
    leftover = [k for k in flat if any("lora" in part for part in k)]
    if leftover:
        raise ValueError(f"unmerged LoRA weights: {leftover}")
    return _unflatten(flat), merged
