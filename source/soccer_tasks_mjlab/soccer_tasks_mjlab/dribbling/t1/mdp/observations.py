"""Dribbling-specific observation terms for T1 on mjlab.

These supplement the standard observations (base_lin_vel, base_ang_vel,
projected_gravity, joint_pos_rel, joint_vel_rel, last_action) which come
from mjlab.envs.mdp.observations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_ROBOT_CFG = SceneEntityCfg("robot")
_BALL_CFG = SceneEntityCfg("ball")


def ball_position_in_robot_root_frame(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg = _ROBOT_CFG,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
) -> torch.Tensor:
    """Ball position relative to the robot root frame (3D)."""
    robot: Entity = env.scene[robot_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]

    rel_pos_w = ball.data.root_link_pos_w - robot.data.root_link_pos_w
    return quat_apply_inverse(robot.data.root_link_quat_w, rel_pos_w)


def ball_velocity_in_robot_root_frame(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg = _ROBOT_CFG,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
) -> torch.Tensor:
    """Ball velocity relative to the robot root frame (3D)."""
    robot: Entity = env.scene[robot_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]

    rel_vel_w = ball.data.root_link_lin_vel_w - robot.data.root_link_lin_vel_w
    return quat_apply_inverse(robot.data.root_link_quat_w, rel_vel_w)


def gait_phase_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Sinusoidal gait phase observation (2D: sin, cos).

    Uses a fixed cycle time of 0.8s to provide the policy with a clock
    signal for generating alternating gait patterns.
    """
    cycle_time = 0.8
    phase = (env.episode_length_buf * env.step_dt) % cycle_time / cycle_time
    return torch.cat(
        [
            torch.sin(2 * torch.pi * phase).unsqueeze(1),
            torch.cos(2 * torch.pi * phase).unsqueeze(1),
        ],
        dim=1,
    )
