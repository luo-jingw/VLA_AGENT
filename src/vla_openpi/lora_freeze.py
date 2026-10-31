"""Freeze filter for fine-tuning with LoRA adapters only: every parameter that is not an adapter stays at the
weights it was loaded with."""

from __future__ import annotations

import flax.nnx as nnx
from openpi.shared import nnx_utils


def non_lora_filter() -> nnx.filterlib.Filter:
    """Every parameter that is not a LoRA adapter (`lora_a` / `lora_b` of an attention einsum, or the
    `*_lora_a` / `*_lora_b` of a feed-forward layer)."""
    return nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex(".*lora.*")))
