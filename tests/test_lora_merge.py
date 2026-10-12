"""lora_merge against a reference LoRA forward pass written from openpi's lora.Einsum, in numpy."""

from __future__ import annotations

import numpy as np
import pytest

from vla_tools.lora_merge import ATTENTION_EQUATIONS
from vla_tools.lora_merge import AdapterScaling
from vla_tools.lora_merge import einsum_delta
from vla_tools.lora_merge import lora_equations
from vla_tools.lora_merge import merge_lora

SIZES = {"B": 2, "S": 3, "T": 3, "D": 8, "K": 2, "N": 2, "H": 4, "3": 3, "2": 2, "L": 2}
# The two equations openpi's lora.Einsum._make_lora_eqns derives for each attention einsum.
OPENPI_LORA_EQUATIONS = {
    "qkv_einsum": ("BSD,3KDL->3BSKL", "3BSKL,3KLH->3BSKH"),
    "q_einsum": ("BTD,NDL->BTNL", "BTNL,NLH->BTNH"),
    "kv_einsum": ("BSD,2KDL->2BSKL", "2BSKL,2KLH->2BSKH"),
    "attn_vec_einsum": ("BTNH,NHL->BTL", "BTL,NLD->BTD"),
}


def _letters(equation: str) -> str:
    """numpy einsum takes letters only; each equation has at most one digit label."""
    return equation.replace("3", "P").replace("2", "P")


def _shape(labels: str) -> tuple[int, ...]:
    return tuple(SIZES[c] for c in labels)


def _operands(equation: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lhs, rest = equation.split(",")
    rhs = rest.split("->")[0]
    a_rhs = rhs[:-1] + "L"
    b_rhs = rhs[:-2] + "L" + rhs[-1]
    return (rng.normal(size=_shape(lhs)), rng.normal(size=_shape(rhs)), rng.normal(size=_shape(a_rhs)),
            rng.normal(size=_shape(b_rhs)))  # fmt: skip


@pytest.mark.parametrize("module", sorted(ATTENTION_EQUATIONS))
def test_lora_equations_match_openpi(module: str) -> None:
    assert lora_equations(ATTENTION_EQUATIONS[module]) == OPENPI_LORA_EQUATIONS[module]


@pytest.mark.parametrize("module", sorted(ATTENTION_EQUATIONS))
def test_merged_weight_reproduces_the_lora_forward_pass(module: str) -> None:
    equation, scaling = ATTENTION_EQUATIONS[module], 0.5
    x, w, a, b = _operands(equation, np.random.default_rng(0))
    eqn_a, eqn_b = (_letters(e) for e in lora_equations(equation))
    reference = np.einsum(_letters(equation), x, w) + scaling * np.einsum(eqn_b, np.einsum(eqn_a, x, a), b)
    merged = w + scaling * einsum_delta(equation, a[None], b[None])[0]
    difference = float(np.abs(np.einsum(_letters(equation), x, merged) - reference).max())
    print(f"{module}: merged vs LoRA forward max difference {difference:.1e}")
    assert difference < 1e-10


def test_per_head_merge_of_the_output_projection_differs() -> None:
    equation = ATTENTION_EQUATIONS["attn_vec_einsum"]
    x, w, a, b = _operands(equation, np.random.default_rng(1))
    exact = np.einsum(equation, x, w + einsum_delta(equation, a[None], b[None])[0])
    per_head = np.einsum(equation, x, w + np.einsum("NHL,NLD->NHD", a, b))
    difference = float(np.abs(exact - per_head).max())
    print(f"per-head a @ b on the output projection: max difference {difference:.2f}")
    assert difference > 1e-3


def _tree(rng: np.random.Generator, layers: int) -> dict:
    def attention(rank: int) -> dict[str, np.ndarray]:
        return {"w": rng.normal(size=(layers, 2, 4, 8)), "lora_a": rng.normal(size=(layers, 2, 4, rank)),
                "lora_b": rng.normal(size=(layers, 2, rank, 8))}  # fmt: skip

    mlp = {
        "gating_einsum": rng.normal(size=(layers, 2, 8, 16)),
        "gating_einsum_lora_a": rng.normal(size=(layers, 2, 8, 2)),
        "gating_einsum_lora_b": rng.normal(size=(layers, 2, 2, 16)),
        "linear": rng.normal(size=(layers, 16, 8)),
        "linear_lora_a": rng.normal(size=(layers, 16, 2)),
        "linear_lora_b": rng.normal(size=(layers, 2, 8)),
    }
    return {"layers": {"attn": {"attn_vec_einsum": attention(2), "attn_vec_einsum_1": attention(3)}, "mlp": mlp}}


def test_merge_lora_folds_every_adapter_with_its_experts_scaling() -> None:
    tree = _tree(np.random.default_rng(2), layers=3)
    experts = (AdapterScaling(rank=2, scaling=0.5), AdapterScaling(rank=3, scaling=2.0))
    merged, modules = merge_lora(tree, experts)
    print(modules)
    equation = ATTENTION_EQUATIONS["attn_vec_einsum"]
    for name, expert in (("attn_vec_einsum", experts[0]), ("attn_vec_einsum_1", experts[1])):
        source = tree["layers"]["attn"][name]
        expected = source["w"] + expert.scaling * einsum_delta(equation, source["lora_a"], source["lora_b"])
        np.testing.assert_allclose(merged["layers"]["attn"][name]["w"], expected)
        assert set(merged["layers"]["attn"][name]) == {"w"}
    mlp = tree["layers"]["mlp"]
    np.testing.assert_allclose(
        merged["layers"]["mlp"]["linear"], mlp["linear"] + mlp["linear_lora_a"] @ mlp["linear_lora_b"]
    )
    assert set(merged["layers"]["mlp"]) == {"gating_einsum", "linear"}
    assert len(modules) == 4


def test_merge_lora_refuses_a_rank_mismatch_and_unknown_adapters() -> None:
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="rank"):
        merge_lora(_tree(rng, 1), (AdapterScaling(rank=4, scaling=1.0), AdapterScaling(rank=3, scaling=1.0)))
    tree = _tree(rng, 1)
    tree["layers"]["mlp"]["router_lora_c"] = rng.normal(size=(1, 2))
    with pytest.raises(ValueError, match="unmerged"):
        merge_lora(tree, (AdapterScaling(rank=2, scaling=1.0), AdapterScaling(rank=3, scaling=1.0)))
