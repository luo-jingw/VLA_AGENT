"""Checks that need an openpi environment: LoRA merge vs openpi's lora.Einsum, the vla_openpi transforms, LeRobot loading.

Skipped when openpi is not importable. JAX runs on CPU here unless JAX_PLATFORMS is already set, so the
tests do not take GPU memory.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
pytest.importorskip("openpi.models.lora")

import jax.numpy as jnp  # noqa: E402
from openpi.models import lora  # noqa: E402

from pipeline_chain import HORIZON  # noqa: E402
from pipeline_chain import SPAN  # noqa: E402
from pipeline_chain import ChainRun  # noqa: E402
from synthetic_dataset import SPEC  # noqa: E402
from vla_openpi import ArcInputs  # noqa: E402
from vla_openpi import ArcOutputs  # noqa: E402
from vla_openpi import RawOutputs  # noqa: E402
from vla_tools.arc_interp import ArcInterpolator  # noqa: E402
from vla_tools.lora_merge import ATTENTION_EQUATIONS  # noqa: E402
from vla_tools.lora_merge import einsum_delta  # noqa: E402

pytestmark = pytest.mark.openpi
SIZES = {"B": 2, "S": 3, "T": 3, "D": 8, "K": 2, "N": 2, "H": 4, "3": 3, "2": 2, "L": 2}


def _shape(labels: str) -> tuple[int, ...]:
    return tuple(SIZES[c] for c in labels)


@pytest.mark.parametrize("module", sorted(ATTENTION_EQUATIONS))
def test_merged_weight_matches_openpi_lora_einsum(module: str) -> None:
    equation = ATTENTION_EQUATIONS[module]
    lhs, rest = equation.split(",")
    rhs = rest.split("->")[0]
    config = lora.LoRAConfig(rank=SIZES["L"], alpha=1.5)
    rng = np.random.default_rng(0)
    w = rng.normal(0.0, 0.3, _shape(rhs)).astype(np.float32)
    a = rng.normal(0.0, 0.3, _shape(rhs[:-1] + "L")).astype(np.float32)
    b = rng.normal(0.0, 0.3, _shape(rhs[:-2] + "L" + rhs[-1])).astype(np.float32)
    x = rng.normal(0.0, 1.0, _shape(lhs)).astype(np.float32)

    layer = lora.Einsum(shape=w.shape, lora_config=config)
    with_lora = layer.apply({"params": {"w": w, "lora_a": a, "lora_b": b}}, equation, jnp.asarray(x))
    merged = w + config.scaling_value * einsum_delta(equation, a[None], b[None])[0]
    plain = lora.Einsum(shape=w.shape).apply({"params": {"w": merged.astype(np.float32)}}, equation, jnp.asarray(x))
    difference = float(np.abs(np.asarray(with_lora) - np.asarray(plain)).max())
    print(f"{module}: openpi LoRA forward vs merged weight, max difference {difference:.1e}")
    assert difference < 1e-4


def _sample(images: np.ndarray) -> dict:
    return {
        "observation/image": images,
        "observation/state": np.zeros(SPEC.action_dim, dtype=np.float32),
        "actions": np.arange(HORIZON * SPEC.action_dim, dtype=np.float32),
        "action_loss_weight": np.ones(HORIZON, dtype=np.float32),
        "prompt": "pick",
    }


def test_arc_inputs_map_cameras_label_and_weights() -> None:
    inputs = ArcInputs(
        cameras=(("observation/image", "base_0_rgb"),), action_dim=SPEC.action_dim, action_horizon=HORIZON
    )
    from_dataset = inputs(_sample(np.full((3, 24, 32), 0.5, dtype=np.float32)))
    from_client = inputs(_sample(np.full((24, 32, 3), 127, dtype=np.uint8)))
    assert from_dataset["image"]["base_0_rgb"].shape == (24, 32, 3)
    assert from_dataset["image"]["base_0_rgb"].dtype == np.uint8
    np.testing.assert_array_equal(from_dataset["image"]["base_0_rgb"], from_client["image"]["base_0_rgb"])
    assert {k: bool(v) for k, v in from_dataset["image_mask"].items()} == {
        "base_0_rgb": True,
        "left_wrist_0_rgb": False,
        "right_wrist_0_rgb": False,
    }
    assert from_dataset["actions"].shape == (HORIZON, SPEC.action_dim)
    assert from_dataset["action_loss_weight"].shape == (HORIZON,)
    wrong = ArcInputs(cameras=inputs.cameras, action_dim=SPEC.action_dim, action_horizon=HORIZON + 1)
    with pytest.raises(ValueError, match="action_horizon"):
        wrong(_sample(np.zeros((24, 32, 3), dtype=np.uint8)))


def test_outputs_trim_model_padding_and_interpolate_frames(chain: ChainRun) -> None:
    points = np.zeros((HORIZON, 32), dtype=np.float32)
    assert RawOutputs(action_dim=SPEC.action_dim)({"actions": points})["actions"].shape == (HORIZON, SPEC.action_dim)
    interp = ArcInterpolator.from_report(chain.labels_train / "meta" / "arc_resample.json")
    out = ArcOutputs(interp=interp)({"actions": points, "state": np.zeros(SPEC.action_dim)})
    assert out["actions"].shape == (SPAN, SPEC.action_dim)
    np.testing.assert_allclose(out["arc_point_frames"], np.arange(2, SPAN + 1, 2))


def test_lerobot_loads_the_derived_datasets(chain: ChainRun) -> None:
    lerobot_dataset = pytest.importorskip("lerobot.common.datasets.lerobot_dataset")
    for root in (chain.clean, chain.timed, chain.labels_train, chain.labels_eval):
        dataset = lerobot_dataset.LeRobotDataset(f"local/{root.name}", root=root)
        item = dataset[len(dataset) - 1]
        print(f"{root.name}: {dataset.num_episodes} episodes, {len(dataset)} frames, keys {sorted(item)}")
        assert len(dataset) == dataset.meta.total_frames
        if "action_chunk" in item:
            assert tuple(item["action_chunk"].shape) == (HORIZON * SPEC.action_dim,)


def test_per_point_stats_normalize_each_point_and_invert_through_model_padding() -> None:
    from openpi import transforms as _transforms
    from openpi.shared import normalize

    from vla_openpi.per_point_stats import per_point

    rng = np.random.default_rng(0)
    scale = np.linspace(0.1, 2.0, 10)[:, None] * np.ones((1, 8))  # far points move more than near ones
    chunks = rng.normal(0.0, 1.0, (500, 10, 8)) * scale
    running = normalize.RunningStats()
    running.update(chunks.reshape(len(chunks), -1))
    stats = per_point(running.get_statistics(), (10, 8))
    assert stats.q01.shape == stats.q99.shape == stats.mean.shape == stats.std.shape == (10, 8)
    data = {"actions": chunks[0]}
    normalized = _transforms.Normalize({"actions": stats}, use_quantiles=True)(dict(data))["actions"]
    spread = np.abs(_transforms.Normalize({"actions": stats}, use_quantiles=True)({"actions": chunks})["actions"])
    near, far = np.percentile(spread[:, 0], 98), np.percentile(spread[:, -1], 98)
    assert abs(near - far) < 0.2  # each point uses its own range
    padded = np.concatenate([normalized, rng.normal(size=(10, 24))], axis=-1)  # the model's 32-dim output
    back = _transforms.Unnormalize({"actions": stats}, use_quantiles=True)({"actions": padded})["actions"]
    assert np.abs(back[:, :8] - chunks[0]).max() < 1e-9
    assert np.array_equal(back[:, 8:], padded[:, 8:])
    loaded = normalize.deserialize_json(normalize.serialize_json({"actions": stats}))["actions"]
    assert np.asarray(loaded.q99).shape == (10, 8)


def test_episode_prompt_draws_every_instruction_of_the_sample_episode(tmp_path) -> None:
    from vla_openpi import EpisodePrompt

    (tmp_path / "meta").mkdir()
    rows = [
        {"episode_index": 0, "tasks": ["a"], "length": 3},
        {"episode_index": 1, "tasks": ["b", "c", "d"], "length": 3},
    ]
    (tmp_path / "meta" / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    transform = EpisodePrompt(tmp_path)
    drawn = [transform({"episode_index": np.int64(1), "x": 7}) for _ in range(300)]
    counts = {p: sum(d["prompt"] == p for d in drawn) for p in ("b", "c", "d")}
    print(counts)
    assert set(counts) == {d["prompt"] for d in drawn} and min(counts.values()) > 50
    assert all("episode_index" not in d and d["x"] == 7 for d in drawn)


def test_rotate_prompts_config_reads_no_file_when_built(tmp_path) -> None:
    from openpi.models import pi0_config
    from openpi.training import config as _config

    from vla_openpi import EpisodePrompt
    from vla_openpi import LeRobotArcDataConfig

    def factory(prompt_from_task: bool) -> LeRobotArcDataConfig:
        return LeRobotArcDataConfig(
            repo_id="missing-dataset",
            base_config=_config.DataConfig(local_root=tmp_path, prompt_from_task=prompt_from_task),
            observation_columns=(("observation/state", "observation.state"),),
            inputs=ArcInputs(cameras=(), action_dim=2, action_horizon=2),
            outputs=RawOutputs(action_dim=2),
            rotate_prompts=True,
        )

    model = pi0_config.Pi0Config(pi05=True, action_horizon=2, discrete_state_input=False)
    data = factory(prompt_from_task=False).create(tmp_path / "assets", model)
    repack = data.repack_transforms.inputs
    print([type(t).__name__ for t in repack], repack[0].structure)
    assert isinstance(repack[1], EpisodePrompt) and repack[0].structure["episode_index"] == "episode_index"
    assert "prompt" not in repack[0].structure
    with pytest.raises(ValueError, match="prompt_from_task"):
        factory(prompt_from_task=True).create(tmp_path / "assets", model)
