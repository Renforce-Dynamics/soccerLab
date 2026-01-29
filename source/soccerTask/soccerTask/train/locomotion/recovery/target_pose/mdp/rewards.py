from __future__ import annotations
from tkinter import NO

from torch import Tensor

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply_inverse, yaw_quat


def target_body_pos_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str = None,
    std: float = 0.3,
) -> Tensor:
    """Exponential reward on body position error w.r.t. a fixed target pose from command."""
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
    return (-sq_err / denom).exp()


def target_body_ori_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str = None,
    std: float = 0.4,
) -> Tensor:
    """Exponential reward on body orientation error w.r.t. a fixed target pose from command."""
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
    return (-sq_err / denom).exp()


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

