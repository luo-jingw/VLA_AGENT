"""Freeze filter for pi0.5 trained with the VLM frozen: SigLIP and Gemma 2B stay at the base weights, the action
expert (Gemma 300M) and the action and time projections are trained."""

from __future__ import annotations

import flax.nnx as nnx
from openpi.shared import nnx_utils


def vlm_filter() -> nnx.filterlib.Filter:
    """Parameters of the VLM: SigLIP (img) and every Gemma parameter that is not the action expert's (suffix _1)."""
    gemma_2b = nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")))
    return nnx.All(nnx.Param, nnx.Any(nnx_utils.PathRegex(".*img.*"), gemma_2b))
