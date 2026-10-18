"""A LIBERO task with the Panda driven by absolute joint targets through robosuite's joint position controller,
or by LIBERO's recorded 7-dim actions through the recorded OSC_POSE controller.

Joint control: each control frame sets the controller's goal to the target exactly (the action is the target
minus the measured joints over an output range of ±π rad), and the controller tracks it for the frame's
physics steps with torques M (kp e − kd q̇) plus gravity compensation, kd = 2 √kp (critically damped), clipped
to the actuators' torque limits. The command 0 / 1 maps to LIBERO's gripper action −1 (open) / +1 (close).

The environment arguments are those of LIBERO's ControlEnv (and of its recordings): Panda, agentview and
robot0_eye_in_hand cameras, 20 Hz control, offscreen rendering. Images are rotated 180 degrees, the
orientation of the converted dataset and of openpi's LIBERO examples. Run with MUJOCO_GL=egl and one
thread per process (OMP_NUM_THREADS=1); several threads per process slow a step about 20 times on a shared
host.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import robosuite
from libero.libero.envs import bddl_utils
from libero.libero.envs.bddl_base_domain import TASK_MAPPING

from vla_libero.interfaces import Observation

CAMERAS = ("agentview", "robot0_eye_in_hand")
CONTROL_FREQ = 20
JOINT_RANGE = np.pi  # rad: largest goal change per frame the joint action can express


class LiberoJointEnv:
    def __init__(
        self, bddl_file: Path, controller: Literal["joint", "osc"], kp: float = 150.0, camera_size: int = 128
    ) -> None:
        if controller == "joint":
            config = robosuite.load_controller_config(default_controller="JOINT_POSITION")
            config.update(kp=kp, damping_ratio=1.0, input_max=1.0, input_min=-1.0,
                          output_max=JOINT_RANGE, output_min=-JOINT_RANGE, interpolation=None)  # fmt: skip
        elif controller == "osc":
            config = robosuite.load_controller_config(default_controller="OSC_POSE")
        else:
            raise ValueError(f"controller must be joint or osc, got {controller!r}")
        problem = bddl_utils.get_problem_info(str(bddl_file))
        self.controller = controller
        self.kp = kp
        self.language = problem["language_instruction"]
        self._env = TASK_MAPPING[problem["problem_name"]](
            str(bddl_file),
            robots=["Panda"],
            controller_configs=config,
            gripper_types="default",
            initialization_noise=None,
            use_camera_obs=True,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera="frontview",
            render_collision_mesh=False,
            render_visual_mesh=True,
            render_gpu_device_id=-1,
            control_freq=CONTROL_FREQ,
            horizon=100_000,
            ignore_done=True,
            hard_reset=True,
            camera_names=list(CAMERAS),
            camera_heights=camera_size,
            camera_widths=camera_size,
            camera_depths=False,
            camera_segmentations=None,
            renderer="mujoco",
            renderer_config=None,
        )
        self._object = self._env.obj_of_interest[0]
        self._command = 0.0
        self._env.reset()

    def reset(self, sim_state: np.ndarray) -> Observation:
        self._env.reset()
        self._env.sim.set_state_from_flattened(np.asarray(sim_state, dtype=np.float64))
        self._env.sim.forward()
        self._env._check_success()
        self._env._post_process()
        self._env._update_observables(force=True)
        self._command = 0.0
        return self._observation(self._env._get_observations())

    def joints(self) -> np.ndarray:
        return np.array(self._env.sim.data.qpos[self._env.robots[0]._ref_joint_pos_indexes], dtype=np.float64)

    def gripper(self) -> np.ndarray:
        """(2,) finger joint positions."""
        return np.array(self._env.sim.data.qpos[self._env.robots[0]._ref_gripper_joint_pos_indexes], dtype=np.float64)

    def step(self, target: np.ndarray) -> Observation:
        """Joint control: (8,) absolute joint targets + command. OSC control: LIBERO's (7,) action."""
        target = np.asarray(target, dtype=np.float64)
        if self.controller == "joint":
            self._command = 1.0 if target[7] > 0.5 else 0.0
            delta = np.clip((target[:7] - self.joints()) / JOINT_RANGE, -1.0, 1.0)
            action = np.concatenate([delta, [2.0 * self._command - 1.0]])
        else:
            self._command = 1.0 if target[6] > 0.0 else 0.0
            action = target
        obs, _, _, _ = self._env.step(action)
        return self._observation(obs)

    def success(self) -> bool:
        return bool(self._env._check_success())

    def object_height(self) -> float:
        return float(self._env.sim.data.body_xpos[self._env.obj_body_id[self._object]][2])

    def close(self) -> None:
        self._env.close()

    def _observation(self, obs: dict) -> Observation:
        return Observation(
            agentview=np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
            wrist=np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
            state=np.concatenate([self.joints(), [self._command]]),
        )
