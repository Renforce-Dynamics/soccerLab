from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def track_ball_vel_xy_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Exponential reward for tracking commanded ball linear velocity in xy (world frame).

    Reward per env: exp(-||v_cmd_xy - v_ball_xy||^2 / std^2)
    Returns shape [num_envs].
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    vel_ball_xy = ball.data.root_lin_vel_w[:, :2]

    vel_cmd_xy = env.command_manager.get_command(command_name)[:, :2]

    sq_err = torch.sum(torch.square(vel_cmd_xy - vel_ball_xy), dim=1)
    return torch.exp(-sq_err / (std**2))


def track_ball_vel_direction(
    env: ManagerBasedRLEnv,
    command_name: str,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    eps: float = 1e-6,
) -> torch.Tensor:
    """Cosine-similarity reward between commanded xy direction and actual ball xy direction.

    Returns shape [num_envs].
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    v = ball.data.root_lin_vel_w[:, :2]
    v_cmd = env.command_manager.get_command(command_name)[:, :2]

    v_norm = torch.linalg.vector_norm(v, dim=1, keepdim=True).clamp(min=eps)
    v_cmd_norm = torch.linalg.vector_norm(v_cmd, dim=1, keepdim=True).clamp(min=eps)

    v_unit = v / v_norm
    v_cmd_unit = v_cmd / v_cmd_norm

    return torch.sum(v_unit * v_cmd_unit, dim=1).clamp(min=-1.0, max=1.0)


def ball_velocity_magnitude(
    env: ManagerBasedRLEnv,
    command_name: str,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reward for matching the speed magnitude in xy.

    Returns 1 - | |v_cmd| - |v| | / (|v_cmd| + eps), clipped to [0, 1].
    Shape [num_envs].
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    v = ball.data.root_lin_vel_w[:, :2]
    v_cmd = env.command_manager.get_command(command_name)[:, :2]

    spd = torch.linalg.vector_norm(v, dim=1)
    spd_cmd = torch.linalg.vector_norm(v_cmd, dim=1)

    rel_err = torch.abs(spd_cmd - spd) / (spd_cmd + eps)
    return (1.0 - rel_err).clamp(min=0.0, max=1.0)


def command_rate_l2(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """L2 penalty on command change-rate.

    This is a stub interface for hierarchical shooting. Different command generators store
    history differently; once you decide the command term type, you can compute:

        rate = cmd_t - cmd_{t-1}

    For now this returns zeros to keep training runnable until you wire it up.

    Returns shape [num_envs].
    """
    cmd = env.command_manager.get_command(command_name)
    return torch.zeros(cmd.shape[0], device=cmd.device)


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint energy usage.

    Computes an (unnormalized) mechanical power proxy:

        \sum_i |\dot{q}_i| * |\tau_i|

    Returns shape [num_envs].
    """

    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids

    qvel = asset.data.joint_vel[:, joint_ids]
    # Prefer applied torque when available; fall back to computed torque.
    if hasattr(asset.data, "applied_torque") and asset.data.applied_torque is not None:
        torque = asset.data.applied_torque[:, joint_ids]
    elif hasattr(asset.data, "computed_torque") and asset.data.computed_torque is not None:
        torque = asset.data.computed_torque[:, joint_ids]
    else:
        torque = torch.zeros_like(qvel)

    return torch.sum(torch.abs(qvel) * torch.abs(torque), dim=-1)
