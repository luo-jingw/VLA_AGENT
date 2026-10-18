"""Chunk policies for the closed loop: a served openpi policy that answers with points, and the still policy
that answers with the observed pose (checks the loop without a model)."""

from __future__ import annotations

import numpy as np
from openpi_client import websocket_client_policy

from vla_libero.interfaces import Observation


class WebsocketChunkPolicy:
    """A policy served by `vla_openpi.launch.serve_points`; the keys are the training config's repacked keys."""

    def __init__(
        self,
        host: str,
        port: int,
        image_key: str = "observation/image",
        wrist_key: str = "observation/wrist_image",
        state_key: str = "observation/state",
    ) -> None:
        self._client = websocket_client_policy.WebsocketClientPolicy(host, port)
        self._keys = (image_key, wrist_key, state_key)

    def infer(self, observation: Observation, prompt: str) -> np.ndarray:
        image, wrist, state = self._keys
        out = self._client.infer({image: observation.agentview, wrist: observation.wrist,
                                  state: observation.state.astype(np.float32), "prompt": prompt})  # fmt: skip
        return np.asarray(out["actions"], dtype=np.float64)


class StillPolicy:
    """Every point is the observed pose: the robot holds still."""

    def __init__(self, action_horizon: int) -> None:
        self._horizon = action_horizon

    def infer(self, observation: Observation, prompt: str) -> np.ndarray:
        return np.repeat(observation.state[None, :].astype(np.float64), self._horizon, axis=0)
