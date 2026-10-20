"""Run openpi's norm-stats computation, training and serving on a TrainConfig object instead of a registered name.

A config built from LeRobotArcDataConfig cannot be registered in openpi's config.py at import time (the
two modules would import each other), so these functions take the config itself. They call openpi's
own scripts, so `<openpi>/scripts` must be on PYTHONPATH.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import compute_norm_stats as openpi_norm_stats
import numpy as np
import tqdm
import train as openpi_train
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import normalize
from openpi.training import config as _config

from vla_openpi.arc_transforms import ArcOutputs
from vla_openpi.per_point_stats import per_point
from vla_tools.arc_interp import ArcInterpolator


def norm_stats(config: _config.TrainConfig, max_frames: int | None = None, per_point_actions: bool = False) -> Path:
    """openpi's compute_norm_stats.main for a config object. Returns the directory written.

    `per_point_actions`: the action statistics are kept per point of the chunk, (H, D) arrays, instead of pooled
    over its H points, (D,); openpi's Normalize and Unnormalize broadcast either shape over the (H, D) actions.
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    loader, batches = openpi_norm_stats.create_torch_dataloader(
        data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
    )
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    chunk_shape: tuple[int, ...] = ()
    for batch in tqdm.tqdm(loader, total=batches, desc="norm stats"):
        stats["state"].update(np.asarray(batch["state"]))
        actions = np.asarray(batch["actions"])
        chunk_shape = actions.shape[1:]
        stats["actions"].update(actions.reshape(len(actions), -1) if per_point_actions else actions)
    result = {key: running.get_statistics() for key, running in stats.items()}
    if per_point_actions:
        result["actions"] = per_point(result["actions"], chunk_shape)
    output = config.assets_dirs / data_config.repo_id
    normalize.save(output, result)
    return output


def train(config: _config.TrainConfig) -> None:
    openpi_train.main(config)


def serving_config(config: _config.TrainConfig, checkpoint: Path, rate: float = 1.0) -> _config.TrainConfig:
    """The training config with its outputs replaced by per-frame interpolation from the checkpoint's report,
    executed at `rate` times the recording's average pace."""
    data_config = config.data.create(config.assets_dirs, config.model)
    report = Path(checkpoint) / "assets" / str(data_config.asset_id) / "arc_resample.json"
    interp = ArcInterpolator.from_report(report, rate)
    return dataclasses.replace(config, data=dataclasses.replace(config.data, outputs=ArcOutputs(interp)))


def serve_points(config: _config.TrainConfig, checkpoint: Path, port: int) -> None:
    """Serve with the training config's outputs: each request is answered with the (H, D) absolute points, for a
    client that interpolates them itself (vla_libero.evaluate)."""
    policy = policy_config.create_trained_policy(config, str(checkpoint))
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=port, metadata=policy.metadata
    ).serve_forever()


def serve(config: _config.TrainConfig, checkpoint: Path, port: int, rate: float = 1.0) -> None:
    policy = policy_config.create_trained_policy(serving_config(config, checkpoint, rate), str(checkpoint))
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=port, metadata=policy.metadata
    ).serve_forever()
