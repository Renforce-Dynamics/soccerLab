"""Observation terms for G1 kicking task (mjlab backend).

Ported from G1_kicking/kick_task/mdp/observations.py with IsaacLab->mjlab API mapping:
  - root_pos_w -> root_link_pos_w
  - root_quat_w -> root_link_quat_w
  - root_lin_vel_w -> root_link_lin_vel_w
  - body_pos_w -> body_link_pos_w
  - body_quat_w -> body_link_quat_w
"""

from __future__ import annotations

import torch
from typing import cast

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .commands import KickMotionCommand


def _ball_robot(env: ManagerBasedRlEnv, ball_asset_name: str = "ball", robot_asset_name: str = "robot"):
    ball = env.scene[ball_asset_name]
    robot = env.scene[robot_asset_name]
    return ball, robot


def _quat_apply_inverse(quat, vec):
    from mjlab.utils.lab_api.math import quat_apply_inverse as _qai, yaw_quat as _yq
    return _qai(quat, vec)


def _yaw_quat(quat):
    from mjlab.utils.lab_api.math import yaw_quat as _yq
    return _yq(quat)


def ball_pos_rel(env: ManagerBasedRlEnv, ball_asset_name: str = "ball", robot_asset_name: str = "robot") -> torch.Tensor:
    ball, robot = _ball_robot(env, ball_asset_name, robot_asset_name)
    rel_pos_w = ball.data.root_link_pos_w - robot.data.root_link_pos_w
    return _quat_apply_inverse(_yaw_quat(robot.data.root_link_quat_w), rel_pos_w)


def ball_vel_rel(env: ManagerBasedRlEnv, ball_asset_name: str = "ball", robot_asset_name: str = "robot") -> torch.Tensor:
    ball, robot = _ball_robot(env, ball_asset_name, robot_asset_name)
    rel_vel_w = ball.data.root_link_lin_vel_w - robot.data.root_link_lin_vel_w
    return _quat_apply_inverse(_yaw_quat(robot.data.root_link_quat_w), rel_vel_w)


def goal_dir_rel(env: ManagerBasedRlEnv, command_name: str = "ball_target_velocity", robot_asset_name: str = "robot") -> torch.Tensor:
    robot = env.scene[robot_asset_name]
    goal_vec_w = env.command_manager.get_command(command_name)[:, :3]
    return _quat_apply_inverse(_yaw_quat(robot.data.root_link_quat_w), goal_vec_w)


def ball_to_goal_dir_rel(
    env: ManagerBasedRlEnv,
    command_name: str = "ball_target_velocity",
    robot_asset_name: str = "robot",
    eps: float = 1e-6,
) -> torch.Tensor:
    robot = env.scene[robot_asset_name]
    goal_vec_w = env.command_manager.get_command(command_name)[:, :2]
    goal_dir_w = goal_vec_w / torch.linalg.vector_norm(goal_vec_w, dim=1, keepdim=True).clamp(min=eps)
    goal_dir_w_3d = torch.zeros((env.num_envs, 3), device=goal_dir_w.device, dtype=goal_dir_w.dtype)
    goal_dir_w_3d[:, :2] = goal_dir_w
    return _quat_apply_inverse(_yaw_quat(robot.data.root_link_quat_w), goal_dir_w_3d)


def kick_phase(env: ManagerBasedRlEnv, command_name: str = "kick_motion") -> torch.Tensor:
    command = cast(KickMotionCommand, env.command_manager.get_term(command_name))
    return command.phase.unsqueeze(-1)


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str = "kick_motion") -> torch.Tensor:
    command = cast(KickMotionCommand, env.command_manager.get_term(command_name))
    anchor_rel_w = command.anchor_pos_w - command.robot_anchor_pos_w
    return _quat_apply_inverse(_yaw_quat(command.robot_anchor_quat_w), anchor_rel_w)
