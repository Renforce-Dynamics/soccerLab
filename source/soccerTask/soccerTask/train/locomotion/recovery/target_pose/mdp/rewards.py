from __future__ import annotations

import torch
from torch import Tensor

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from .observations import post_reset_flag


def target_body_pos_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str = None,
    std: float = 0.3,
) -> Tensor:
    """Exponential reward on body position error w.r.t. a fixed target pose from command.
    
    Only active after reset phase (masked by post_reset_flag).
    """
    robot: Articulation = env.scene["robot"]
    command = env.command_manager.get_term(command_name)

    # Use all environments in the batch. Reward manager does not pass env_ids here.
    root_state = robot.data.root_state_w
    root_pos = root_state[..., :3]
    root_quat = root_state[..., 3:7]
    yaw = yaw_quat(root_quat)

    curr_pos_w = robot.data.body_pos_w[:, command.body_indices]
    target_pos_w = command.body_pos_w

    curr_rel = curr_pos_w - root_pos.unsqueeze(1)
    target_rel = target_pos_w - root_pos.unsqueeze(1)

    num_bodies = curr_rel.shape[1]
    yaw_expanded = yaw.unsqueeze(1).expand(-1, num_bodies, -1)
    curr_yaw = quat_apply_inverse(yaw_expanded, curr_rel)
    target_yaw = quat_apply_inverse(yaw_expanded, target_rel)

    diff = curr_yaw - target_yaw
    sq_err = (diff**2).sum(dim=-1).mean(dim=-1)

    denom = 2.0 * std * std
    reward = (-sq_err / denom).exp()
    
    # Mask with post_reset_flag
    post_reset = post_reset_flag(env).to(dtype=reward.dtype)
    return reward * post_reset.squeeze(-1)


def target_body_height_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str | None = None,
    std: float = 0.1,
) -> Tensor:
    """Exponential reward on body height error w.r.t. a fixed target pose from command.

    This only tracks the vertical (z) component of the selected bodies, ignoring x/y.
    Only active after reset phase (masked by post_reset_flag).
    """
    robot: Articulation = env.scene["robot"]
    command = env.command_manager.get_term(command_name)

    # Current and target body positions in world frame.
    curr_pos_w = robot.data.body_pos_w[:, command.body_indices]   # [N, B, 3]
    target_pos_w = command.body_pos_w                             # [N, B, 3]

    # Extract z-coordinates and compute mean squared height error per env.
    curr_z = curr_pos_w[..., 2]
    target_z = target_pos_w[..., 2]
    diff_z = curr_z - target_z
    sq_err = (diff_z**2).mean(dim=-1)  # [N]

    denom = 2.0 * std * std
    reward = (-sq_err / denom).exp()
    
    # Mask with post_reset_flag
    post_reset = post_reset_flag(env).to(dtype=reward.dtype)
    return reward * post_reset.squeeze(-1)


def target_body_ori_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str = None,
    std: float = 0.4,
) -> Tensor:
    """Exponential reward on body orientation error w.r.t. a fixed target pose from command.
    
    Only active after reset phase (masked by post_reset_flag).
    """
    robot: Articulation = env.scene["robot"]
    command = env.command_manager.get_term(command_name)

    # Use all environments in the batch. Reward manager does not pass env_ids here.
    root_state = robot.data.root_state_w
    root_quat = root_state[..., 3:7]
    yaw = yaw_quat(root_quat)

    curr_quat_w = robot.data.body_quat_w[:, command.body_indices]
    target_quat_w = command.body_quat_w

    curr_quat = curr_quat_w / curr_quat_w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    target_quat = target_quat_w / target_quat_w.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # express orientations in yaw-aligned frame
    yaw_inv = yaw.conj()  # inverse for unit quaternions
    num_bodies = curr_quat.shape[1]
    yaw_inv_expanded = yaw_inv.unsqueeze(1).expand(-1, num_bodies, -1)
    curr_quat_yaw = quat_apply_inverse(yaw_inv_expanded, curr_quat)
    target_quat_yaw = quat_apply_inverse(yaw_inv_expanded, target_quat)

    dot = (curr_quat_yaw * target_quat_yaw).sum(dim=-1).abs().clamp(max=1.0)
    ang_err = 2.0 * dot.arccos()
    sq_err = (ang_err**2).mean(dim=-1)

    denom = 2.0 * std * std
    reward = (-sq_err / denom).exp()
    
    # Mask with post_reset_flag
    post_reset = post_reset_flag(env).to(dtype=reward.dtype)
    return reward * post_reset.squeeze(-1)


def target_body_pos_delta(
    env: ManagerBasedRLEnv,
    command_name: str = None,
) -> Tensor:
    """Flattened body position difference w.r.t. target pose from command (for critic observations)."""
    robot: Articulation = env.scene["robot"]
    command = env.command_manager.get_term(command_name)

    # Use all environments in the batch. Reward/observation manager does not pass env_ids here.
    root_state = robot.data.root_state_w
    root_pos = root_state[..., :3]
    root_quat = root_state[..., 3:7]
    yaw = yaw_quat(root_quat)

    curr_pos_w = robot.data.body_pos_w[:, command.body_indices]
    target_pos_w = command.body_pos_w

    curr_rel = curr_pos_w - root_pos.unsqueeze(1)
    target_rel = target_pos_w - root_pos.unsqueeze(1)

    num_bodies = curr_rel.shape[1]
    yaw_expanded = yaw.unsqueeze(1).expand(-1, num_bodies, -1)
    curr_yaw = quat_apply_inverse(yaw_expanded, curr_rel)
    target_yaw = quat_apply_inverse(yaw_expanded, target_rel)

    diff = curr_yaw - target_yaw
    return diff.reshape(diff.shape[0], -1)


# Import standard reward functions and create masked versions
from isaaclab.envs.mdp.rewards import (
    action_l2 as _action_l2,
    joint_torques_l2 as _joint_torques_l2,
    flat_orientation_l2 as _flat_orientation_l2,
)

from isaaclab_tasks.manager_based.locomotion.velocity.mdp import feet_slide as _feet_slide

def base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> Tensor:
    """Reward for tracking the target base height with an exponential kernel.
    
    Only active after reset phase (masked by post_reset_flag).
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    height_error = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward = torch.exp(-height_error / std**2)
    
    # Mask with post_reset_flag
    post_reset = post_reset_flag(env).to(dtype=reward.dtype)
    return reward * post_reset.squeeze(-1)


def flat_orientation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Trunk"]),
) -> Tensor:
    """Penalty for non-flat base orientation using L2 norm.
    
    Only active after reset phase (masked by post_reset_flag).
    """
    reward = _flat_orientation_l2(env, asset_cfg=asset_cfg)
    
    # Mask with post_reset_flag
    post_reset = post_reset_flag(env).to(dtype=reward.dtype)
    return reward * post_reset.squeeze(-1)


def action_l2(env: ManagerBasedRLEnv) -> Tensor:
    """Penalty for action magnitude using L2 norm.
    
    Only active after reset phase (masked by post_reset_flag).
    """
    reward = _action_l2(env)
    
    # Mask with post_reset_flag
    post_reset = post_reset_flag(env).to(dtype=reward.dtype)
    return reward * post_reset.squeeze(-1)


def joint_torques_l2(env: ManagerBasedRLEnv) -> Tensor:
    """Penalty for joint torques using L2 norm.
    
    Only active after reset phase (masked by post_reset_flag).
    """
    reward = _joint_torques_l2(env)
    
    # Mask with post_reset_flag
    post_reset = post_reset_flag(env).to(dtype=reward.dtype)
    return reward * post_reset.squeeze(-1)


def root_acc_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> Tensor:
    """Penalty for root acceleration using L2 norm.
    
    Computes root acceleration by tracking velocity changes.
    Only active after reset phase (masked by post_reset_flag).
    """
    # Extract the robot asset
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get current root velocities (both linear and angular)
    current_lin_vel = robot.data.root_lin_vel_w  # [num_envs, 3]
    current_ang_vel = robot.data.root_ang_vel_w  # [num_envs, 3]
    
    # Concatenate to form 6D velocity vector
    current_root_vel = torch.cat([current_lin_vel, current_ang_vel], dim=-1)  # [num_envs, 6]
    
    # Get previous root velocity from buffer (initialize if not exists)
    if not hasattr(env, "_root_acc_prev_vel"):
        env._root_acc_prev_vel = torch.zeros_like(current_root_vel)
        env._root_acc_first_call = True
    
    if env._root_acc_first_call:
        # First call: initialize previous velocity and return zeros
        env._root_acc_prev_vel.copy_(current_root_vel)
        env._root_acc_first_call = False
        reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    else:
        # Compute acceleration as velocity difference over timestep
        root_acc = (current_root_vel - env._root_acc_prev_vel) / env.step_dt
        
        # Update velocity history for next call
        env._root_acc_prev_vel.copy_(current_root_vel)
        
        # Compute L2 penalty on accelerations (sum of squared accelerations)
        reward = torch.sum(torch.square(root_acc), dim=-1)
    
    # Mask with post_reset_flag
    post_reset = post_reset_flag(env).to(dtype=reward.dtype)
    return reward * post_reset.squeeze(-1)


def feet_slide(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*foot_link.*"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*foot_link.*"),
) -> Tensor:
    """Penalty for feet sliding when in contact.
    
    Only active after reset phase (masked by post_reset_flag).
    """
    reward = _feet_slide(env, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg)
    
    # Mask with post_reset_flag
    post_reset = post_reset_flag(env).to(dtype=reward.dtype)
    return reward * post_reset.squeeze(-1)

