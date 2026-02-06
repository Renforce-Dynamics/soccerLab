from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply_inverse, yaw_quat


def _get_ball_and_robot(
    env: ManagerBasedRLEnv,
    ball_asset_name: str = "ball",
    robot_asset_name: str = "robot",
) -> tuple[RigidObject, Articulation]:
    ball: RigidObject = env.scene[ball_asset_name]
    robot: Articulation = env.scene[robot_asset_name]
    return ball, robot


def ball_pos_rel(
    env: ManagerBasedRLEnv,
    ball_asset_name: str = "ball",
    robot_asset_name: str = "robot",
) -> torch.Tensor:
    """Ball position relative to robot in robot yaw frame.

    Returns shape [num_envs, 3].
    """
    ball, robot = _get_ball_and_robot(env, ball_asset_name, robot_asset_name)

    rel_pos_w = ball.data.root_pos_w - robot.data.root_pos_w
    rel_pos_yaw = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), rel_pos_w)
    return rel_pos_yaw


def ball_vel_rel(
    env: ManagerBasedRLEnv,
    ball_asset_name: str = "ball",
    robot_asset_name: str = "robot",
) -> torch.Tensor:
    """Ball linear velocity relative to robot in robot yaw frame.

    Returns shape [num_envs, 3].
    """
    ball, robot = _get_ball_and_robot(env, ball_asset_name, robot_asset_name)

    rel_vel_w = ball.data.root_lin_vel_w - robot.data.root_lin_vel_w
    rel_vel_yaw = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), rel_vel_w)
    return rel_vel_yaw


def ball_vel_w(env: ManagerBasedRLEnv, ball_asset_name: str = "ball") -> torch.Tensor:
    """Ball linear velocity in world frame.

    Returns shape [num_envs, 3].
    """
    ball: RigidObject = env.scene[ball_asset_name]
    return ball.data.root_lin_vel_w


def ball_robot_distance(
    env: ManagerBasedRLEnv,
    ball_asset_name: str = "ball",
    robot_asset_name: str = "robot",
) -> torch.Tensor:
    """Planar (xy) distance between robot and ball.

    Returns shape [num_envs, 1].
    """
    rel = ball_pos_rel(env, ball_asset_name=ball_asset_name, robot_asset_name=robot_asset_name)
    dist_xy = torch.linalg.vector_norm(rel[:, :2], dim=1, keepdim=True)
    return dist_xy


def ball_robot_direction(
    env: ManagerBasedRLEnv,
    ball_asset_name: str = "ball",
    robot_asset_name: str = "robot",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Unit direction from robot to ball in robot yaw frame (xy only).

    Returns shape [num_envs, 2].
    """
    rel = ball_pos_rel(env, ball_asset_name=ball_asset_name, robot_asset_name=robot_asset_name)[:, :2]
    norm = torch.linalg.vector_norm(rel, dim=1, keepdim=True).clamp(min=eps)
    return rel / norm
