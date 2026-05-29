"""Dribbling reward terms for T1 on mjlab.

Ported from BackupDribbling's IsaacLab reward functions, adapted for T1
robot joint naming on the mjlab backend.

T1 joint names (capitalized, no _joint suffix):
  Left_Hip_Pitch, Left_Hip_Roll, Left_Hip_Yaw, Left_Knee_Pitch,
  Left_Ankle_Pitch, Left_Ankle_Roll, Right_Hip_*, AAHead_yaw, Head_pitch

Naming convention:
- Functions suffixed with `_reward` return positive values (positive weights).
- Functions suffixed with `_penalty` or `_cost` return positive values (negative weights).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
    euler_xyz_from_quat,
    quat_apply,
    quat_apply_inverse,
    wrap_to_pi,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_ROBOT_CFG = SceneEntityCfg("robot")
_BALL_CFG = SceneEntityCfg("ball")


# ==============================================================================
# Internal helpers
# ==============================================================================


def _get_phase(env: ManagerBasedRlEnv, cycle_time: float = 0.8) -> torch.Tensor:
    return (env.episode_length_buf * env.step_dt) % cycle_time / cycle_time


def _get_gait_phase(
    env: ManagerBasedRlEnv, cycle_time: float = 0.8, double_stand_phase: float = 0.5
) -> torch.Tensor:
    """Returns a stance mask [N, 2] for left/right feet."""
    phase = _get_phase(env, cycle_time)
    sin_pos = torch.sin(2 * torch.pi * phase)
    stance_mask = torch.zeros((env.num_envs, 2), device=env.device)
    stance_mask[:, 0] = sin_pos >= 0  # left foot stance
    stance_mask[:, 1] = sin_pos < 0   # right foot stance
    stance_mask[torch.abs(sin_pos) < double_stand_phase] = 1  # double support
    return stance_mask


# ==============================================================================
# Physics-agnostic rewards (trivial port)
# ==============================================================================


def ball_distance_exp(
    env: ManagerBasedRlEnv,
    std: float,
    robot_cfg: SceneEntityCfg = _ROBOT_CFG,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
) -> torch.Tensor:
    """Reward for being close to the ball (Laplacian kernel on XY distance)."""
    robot: Entity = env.scene[robot_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]
    dist = torch.norm(
        robot.data.root_link_pos_w[:, :2] - ball.data.root_link_pos_w[:, :2], dim=1
    )
    return torch.exp(-dist / std)


def base_height_reward(
    env: ManagerBasedRlEnv,
    target_height: float = 0.60,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    """Reward for maintaining target base height (Gaussian kernel).

    T1 default base height is ~0.60m (lower than G1's 0.75m).
    """
    asset: Entity = env.scene[asset_cfg.name]
    base_height = asset.data.root_link_pos_w[:, 2]
    return torch.exp(-torch.square(base_height - target_height) * 20.0)


def track_ball_velocity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
    target_speed: float = 1.0,
) -> torch.Tensor:
    """Reward for robot velocity projected onto the robot-to-ball direction."""
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]

    target_vec = ball.data.root_link_pos_w - robot.data.root_link_pos_w
    target_vec[:, 2] = 0.0
    target_dir = torch.nn.functional.normalize(target_vec, dim=1)

    robot_vel = robot.data.root_link_lin_vel_w.clone()
    robot_vel[:, 2] = 0.0

    vel_proj = torch.sum(robot_vel * target_dir, dim=1)
    error = target_speed - vel_proj
    return torch.exp(-torch.square(error) / 0.5)


def tracking_ball_view(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg = _ROBOT_CFG,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
    fov_h: float = 0.5,
    fov_v: float = 0.7,
    cam_offset: tuple[float, float, float] = (0.05, 0.0, 0.15),
) -> torch.Tensor:
    """Reward for keeping the ball within a simulated camera field of view."""
    robot: Entity = env.scene[robot_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]

    rel_pos_w = ball.data.root_link_pos_w - robot.data.root_link_pos_w
    rel_pos_b = quat_apply_inverse(robot.data.root_link_quat_w, rel_pos_w)

    offset_t = torch.tensor(cam_offset, device=env.device)
    ball_vis_vec = torch.nn.functional.normalize(rel_pos_b + offset_t, dim=1)

    in_view_h = torch.abs(ball_vis_vec[:, 1]) < fov_h
    in_view_v = torch.abs(ball_vis_vec[:, 2]) < fov_v
    in_front = ball_vis_vec[:, 0] > 0

    return (in_view_h & in_view_v & in_front).float()


def tracking_ball_target_vel_reward_fixed(
    env: ManagerBasedRlEnv,
    target_vel_x: float = 1.0,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
) -> torch.Tensor:
    """Reward for ball moving at target velocity in world X direction."""
    ball: Entity = env.scene[ball_cfg.name]
    ball_vel = ball.data.root_link_lin_vel_w[:, :2]

    target = torch.zeros_like(ball_vel)
    target[:, 0] = target_vel_x

    lin_vel_error = torch.norm(target - ball_vel, dim=-1)
    return torch.exp(-lin_vel_error / 2.0)


def tracking_ang_vel_reward(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg = _ROBOT_CFG,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
) -> torch.Tensor:
    """Reward for turning towards the ball (yaw angular velocity tracking)."""
    robot: Entity = env.scene[robot_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]

    ball_pos = ball.data.root_link_pos_w[:, :2]
    robot_pos = robot.data.root_link_pos_w[:, :2]
    ball_vec = ball_pos - robot_pos

    target_yaw = torch.atan2(ball_vec[:, 1], ball_vec[:, 0])

    base_euler = torch.stack(euler_xyz_from_quat(robot.data.root_link_quat_w), dim=-1)
    current_yaw = base_euler[:, 2]

    ang_error = wrap_to_pi(target_yaw - current_yaw)
    des_ang_vel = torch.clamp(5.0 * ang_error, -0.8, 0.8)

    current_ang_vel = robot.data.root_link_ang_vel_w[:, 2]

    ang_vel_error = torch.square(des_ang_vel - current_ang_vel)
    return torch.exp(-ang_vel_error * 5.0)


def orientation_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    """Reward for maintaining upright orientation."""
    asset: Entity = env.scene[asset_cfg.name]
    base_euler = torch.stack(euler_xyz_from_quat(asset.data.root_link_quat_w), dim=-1)
    projected_gravity = asset.data.projected_gravity_b

    quat_mismatch = torch.exp(-torch.sum(torch.abs(base_euler[:, :2]), dim=1) * 10)
    orientation = torch.exp(-torch.norm(projected_gravity[:, :2], dim=1) * 20)
    return (quat_mismatch + orientation) / 2.0


def dof_vel_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    """Penalty for joint velocities (sum of squares)."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel), dim=1)


def robot_forward_velocity_reward(
    env: ManagerBasedRlEnv,
    target_vel: float = 1.0,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    """Reward for robot forward velocity in world X."""
    asset: Entity = env.scene[asset_cfg.name]
    vel_x = asset.data.root_link_lin_vel_w[:, 0]
    return torch.exp(-torch.square(vel_x - target_vel))


# ==============================================================================
# Rewards needing mjlab API adaptation (with T1 joint names)
# ==============================================================================


def joint_pos_reward_stage1(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
    target_joint_pos_scale: float = 1.15,
    ref_pos_dir: list[float] | None = None,
    cycle_time: float = 0.8,
    double_stand_phase: float = 0.5,
) -> torch.Tensor:
    """Reference motion tracking reward for walking gait.

    For T1, maps to: Left/Right_Hip_Pitch, Left/Right_Knee_Pitch,
    Left/Right_Ankle_Pitch.
    """
    if ref_pos_dir is None:
        ref_pos_dir = [1, -1, 1, -1, 1, -1]

    asset: Entity = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    default_dof_pos = asset.data.default_joint_pos

    phase = _get_phase(env, cycle_time)
    sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
    sin_pos_l = sin_pos.clone()
    sin_pos_r = sin_pos.clone()

    _, joint_names = asset.find_joints((".*",))

    def get_idx(pattern: str) -> int:
        for i, name in enumerate(joint_names):
            if pattern in name:
                return i
        return -1

    # T1 joint names (capitalized)
    l_hip_pitch = get_idx("Left_Hip_Pitch")
    l_knee = get_idx("Left_Knee_Pitch")
    l_ankle_pitch = get_idx("Left_Ankle_Pitch")
    r_hip_pitch = get_idx("Right_Hip_Pitch")
    r_knee = get_idx("Right_Knee_Pitch")
    r_ankle_pitch = get_idx("Right_Ankle_Pitch")

    scale_1 = target_joint_pos_scale
    scale_2 = 2 * target_joint_pos_scale

    sin_pos_l[sin_pos_l > 0] = 0
    sin_pos_r[sin_pos_r < 0] = 0

    offsets = torch.zeros_like(joint_pos)
    if l_hip_pitch != -1:
        offsets[:, l_hip_pitch] = ref_pos_dir[0] * sin_pos_l.squeeze() * scale_1
    if l_knee != -1:
        offsets[:, l_knee] = ref_pos_dir[1] * sin_pos_l.squeeze() * scale_2
    if l_ankle_pitch != -1:
        offsets[:, l_ankle_pitch] = ref_pos_dir[2] * sin_pos_l.squeeze() * scale_1

    if r_hip_pitch != -1:
        offsets[:, r_hip_pitch] = ref_pos_dir[3] * sin_pos_r.squeeze() * scale_1
    if r_knee != -1:
        offsets[:, r_knee] = ref_pos_dir[4] * sin_pos_r.squeeze() * scale_2
    if r_ankle_pitch != -1:
        offsets[:, r_ankle_pitch] = ref_pos_dir[5] * sin_pos_r.squeeze() * scale_1

    mask = torch.abs(sin_pos).squeeze() < double_stand_phase
    offsets[mask] = 0.0

    target_pos = default_dof_pos + offsets
    diff = joint_pos - target_pos

    dist = torch.norm(diff, dim=1)
    return torch.exp(-2 * dist) - 0.2 * torch.clamp(dist, 0, 0.5)


def feet_orientation_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    """Reward for keeping feet flat.

    Requires asset_cfg.body_names = ("left_foot_link", "right_foot_link").
    """
    asset: Entity = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    assert body_ids is not None and len(body_ids) == 2

    feet_quat = asset.data.body_link_quat_w[:, body_ids]

    foot_ori_l = torch.stack(euler_xyz_from_quat(feet_quat[:, 0]), dim=-1)
    foot_ori_r = torch.stack(euler_xyz_from_quat(feet_quat[:, 1]), dim=-1)

    base_euler = torch.stack(euler_xyz_from_quat(asset.data.root_link_quat_w), dim=-1)
    target_ori = torch.zeros_like(base_euler)
    target_ori[:, 2] = base_euler[:, 2]

    diff_l = foot_ori_l - target_ori
    diff_r = foot_ori_r - target_ori

    return torch.exp(-torch.norm(diff_r, dim=1) - torch.norm(diff_l, dim=1))


def feet_distance_reward(
    env: ManagerBasedRlEnv,
    min_dist: float = 0.20,
    max_dist: float = 0.45,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    """Reward for maintaining lateral feet distance within a desired range."""
    asset: Entity = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    assert body_ids is not None and len(body_ids) == 2

    feet_pos = asset.data.body_link_pos_w[:, body_ids]
    foot_dist = torch.norm(feet_pos[:, 0, :2] - feet_pos[:, 1, :2], dim=1)

    d_min = torch.clamp(foot_dist - min_dist, -0.5, 0.0)
    d_max = torch.clamp(foot_dist - max_dist, 0.0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


def feet_contact_forces_cost(
    env: ManagerBasedRlEnv,
    max_contact_force: float = 300.0,
    sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
    """Penalty for excessive foot contact forces above a threshold."""
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.force is not None
    forces = sensor.data.force
    force_mag = torch.norm(forces, dim=-1)
    return torch.sum((force_mag - max_contact_force).clamp(0, 400), dim=1)


def feet_clearance_reward(
    env: ManagerBasedRlEnv,
    target_feet_height: float = 0.092,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    """Reward for swing foot height near target during swing phase.

    Requires asset_cfg.site_names = ("left_foot", "right_foot").
    """
    asset: Entity = env.scene[asset_cfg.name]
    assert asset_cfg.site_ids is not None
    feet_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2] - 0.05

    stance_mask = _get_gait_phase(env)
    swing_mask = 1.0 - stance_mask

    error = torch.abs(feet_z - target_feet_height)
    rew = (error < 0.01).float()

    return torch.sum(rew * swing_mask, dim=1)


def feet_air_time_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    threshold: float = 18.0,
) -> torch.Tensor:
    """Reward for feet being in the air (not in contact)."""
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.force is not None
    forces = sensor.data.force
    force_mag = torch.norm(forces, dim=-1)
    in_contact = force_mag > threshold
    return torch.sum((~in_contact).float(), dim=1)


def feet_stride_reward(
    env: ManagerBasedRlEnv,
    min_stride: float = 0.28,
    max_stride: float = 0.80,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    """Reward for stride length along the body forward direction."""
    asset: Entity = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    assert body_ids is not None and len(body_ids) >= 2
    body_ids = body_ids[:2]

    feet_pos_w = asset.data.body_link_pos_w[:, body_ids]

    fwd_local = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3)
    base_fwd_w = quat_apply(asset.data.root_link_quat_w, fwd_local)
    fwd_xy = base_fwd_w[:, :2]
    fwd_xy = fwd_xy / (torch.norm(fwd_xy, dim=1, keepdim=True) + 1e-8)

    diff_xy = feet_pos_w[:, 0, :2] - feet_pos_w[:, 1, :2]
    stride = torch.abs(torch.sum(diff_xy * fwd_xy, dim=1))

    rew = (stride - min_stride) / (max_stride - min_stride + 1e-6)
    return torch.clamp(rew, 0.0, 1.0)


def head_position_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    """Reward for keeping head joints near default position.

    T1 has head DOFs (AAHead_yaw, Head_pitch). This reward penalizes
    deviation from default to keep the head stable while dribbling.
    """
    asset: Entity = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    default_pos = asset.data.default_joint_pos

    _, joint_names = asset.find_joints((".*",))

    head_indices = []
    for i, name in enumerate(joint_names):
        if "Head" in name:
            head_indices.append(i)

    if not head_indices:
        return torch.zeros(env.num_envs, device=env.device)

    idx = torch.tensor(head_indices, device=env.device)
    head_diff = joint_pos[:, idx] - default_pos[:, idx]
    return torch.exp(-torch.sum(torch.square(head_diff), dim=1) * 10.0)


# ==============================================================================
# Penalty rewards
# ==============================================================================


def self_collision_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str = "self_collision",
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalty for self-collisions."""
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force_history is not None:
        force_mag = torch.norm(data.force_history, dim=-1)
        hit = (force_mag > force_threshold).any(dim=1)
        return hit.sum(dim=-1).float()
    assert data.found is not None
    return data.found.squeeze(-1).float()


def torques_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    """Penalty for applied torques (sum of squares)."""
    asset: Entity = env.scene[asset_cfg.name]
    if hasattr(asset.data, "applied_torque"):
        torques = asset.data.applied_torque
    elif hasattr(asset.data, "qfrc_actuator"):
        torques = asset.data.qfrc_actuator
    else:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.sum(torch.square(torques), dim=1)


def dof_acc_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    """Penalty for joint accelerations (sum of squares)."""
    asset: Entity = env.scene[asset_cfg.name]
    if hasattr(asset.data, "joint_acc"):
        return torch.sum(torch.square(asset.data.joint_acc), dim=1)
    elif hasattr(asset.data, "qacc"):
        return torch.sum(torch.square(asset.data.qacc), dim=1)
    return torch.zeros(env.num_envs, device=env.device)
