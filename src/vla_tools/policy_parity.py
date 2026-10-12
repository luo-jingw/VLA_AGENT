"""Do two checkpoints of the same model (e.g. JAX and its PyTorch conversion) give the same output?

`dump` runs one checkpoint on recorded observations with fixed, seeded noise and saves the outputs;
`compare` reads two dumps. Run each dump in its own process: JAX preallocates GPU memory and does not
give it back. The difference is reported in the policy's output units and in frames of motion; bf16
rounding alone leaves a small non-zero difference, an unmerged adapter leaves one the size of the
fine-tuning itself.

Runs in the openpi environment:
    python -m vla_tools.policy_parity dump --config-name <cfg> --checkpoint <dir> --dataset <ds> \
        --spec robot_spec.json --replay-spec replay_spec.json --episodes 3 4 --per-episode 10 --out jax.npz
    python -m vla_tools.policy_parity dump ... --checkpoint <dir>_pytorch --out torch.npz
    python -m vla_tools.policy_parity compare --a jax.npz --b torch.npz --spec robot_spec.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from openpi.policies import policy_config
from openpi.training import config as _config

from vla_tools.episodes import FrameReader
from vla_tools.episodes import episode_paths
from vla_tools.episodes import load_episode
from vla_tools.episodes import tasks
from vla_tools.replay_eval import ReplaySpec
from vla_tools.robot_spec import RobotSpec


def dump(args: argparse.Namespace) -> None:
    train_config = _config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(train_config, str(args.checkpoint))
    spec, replay = RobotSpec.load(args.spec), ReplaySpec.load(args.replay_spec)
    reader, paths, prompts = FrameReader(args.dataset), dict(episode_paths(args.dataset)), tasks(args.dataset)
    horizon, action_dim = train_config.model.action_horizon, train_config.model.action_dim
    outputs, keys = [], []
    for ep in args.episodes:
        e = load_episode(ep, paths[ep], spec)
        table = pd.read_parquet(paths[ep], columns=[*replay.state.values(), "task_index"])
        for t in np.linspace(0, len(e.actions) - 1, args.per_episode).astype(int):
            obs = {key: reader.frame(ep, column, int(t)) for key, column in replay.images.items()}
            obs.update(
                {key: np.asarray(table[column].iloc[t], dtype=np.float32) for key, column in replay.state.items()}
            )
            obs[replay.prompt_key] = prompts[int(table["task_index"].iloc[0])]
            noise = np.random.default_rng(args.seed + 1000 * ep + int(t)).standard_normal((horizon, action_dim))
            outputs.append(np.asarray(policy.infer(obs, noise=noise.astype(np.float32))["actions"], dtype=np.float64))
            keys.append((ep, int(t)))
    np.savez(args.out, actions=np.stack(outputs), keys=np.array(keys), checkpoint=str(args.checkpoint))
    print(f"{args.checkpoint}: {len(keys)} outputs of shape {outputs[0].shape} -> {args.out}")


def compare(args: argparse.Namespace) -> None:
    a, b = np.load(args.a), np.load(args.b)
    if not np.array_equal(a["keys"], b["keys"]):
        raise SystemExit("the two dumps cover different frames")
    spec = RobotSpec.load(args.spec)
    joints = list(spec.joint_dims)
    diff = np.abs(a["actions"][..., : spec.action_dim] - b["actions"][..., : spec.action_dim])
    scale = np.abs(np.diff(a["actions"][..., joints], axis=1)).max(axis=-1)
    step = float(np.median(scale)) if scale.size else float("nan")
    joint_diff = diff[..., joints].max(axis=(1, 2))
    print(f"{a['checkpoint']} vs {b['checkpoint']}: {len(diff)} outputs")
    print(f"  joints   max |diff| median {np.median(joint_diff):.2e}, max {joint_diff.max():.2e} rad "
          f"(median step between output points {step:.2e})")  # fmt: skip
    for dim in spec.command_dims:
        print(f"  command {dim}: {int((diff[..., dim] > 0.5).sum())} of {diff[..., dim].size} values disagree")
    per_dim = diff.max(axis=(0, 1))
    print("  per-dim max: " + " ".join(f"{d}:{v:.1e}" for d, v in enumerate(per_dim)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--config-name", required=True)
    d.add_argument("--checkpoint", type=Path, required=True)
    d.add_argument("--dataset", type=Path, required=True)
    d.add_argument("--spec", type=Path, required=True)
    d.add_argument("--replay-spec", type=Path, required=True)
    d.add_argument("--episodes", nargs="+", type=int, required=True)
    d.add_argument("--per-episode", type=int, default=10)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--a", type=Path, required=True)
    c.add_argument("--b", type=Path, required=True)
    c.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "dump":
        dump(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
