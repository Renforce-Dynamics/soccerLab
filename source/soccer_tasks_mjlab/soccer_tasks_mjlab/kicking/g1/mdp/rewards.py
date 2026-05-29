"""Reward terms for G1 kicking task (mjlab backend).

Ported from G1_kicking/kick_task/mdp/rewards.py with:
  - body_pos_w -> body_link_pos_w
  - body_quat_w -> body_link_quat_w
  - body_lin_vel_w -> body_link_lin_vel_w
  - body_ang_vel_w -> body_link_ang_vel_w
  - root_pos_w -> root_link_pos_w
  - root_quat_w -> root_link_quat_w
  - root_lin_vel_w -> root_link_lin_vel_w
  - root_ang_vel_w -> root_link_ang_vel_w
  - ManagerBasedRLEnv -> ManagerBasedRlEnv

Includes tiptoe + post-kick fixes per kicking_mjlab_port.md:
  - reward_support_foot_flat_contact_early (NEW)
  - reward_post_kick_upright_early (NEW)
  - Strengthened penalty_right_ankle_pitch_staged, penalty_post_kick_crouch
"""

from __future__ import annotations

import torch
from typing import cast

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    yaw_quat,
)

from .commands import KickMotionCommand


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_assets(env: ManagerBasedRlEnv, ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"), robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    ball = env.scene[ball_cfg.name]
    robot = env.scene[robot_cfg.name]
    return ball, robot


def _get_kick_foot_state(env: ManagerBasedRlEnv, kicking_foot_body_name: str, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot = env.scene[robot_cfg.name]
    foot_id = robot.find_bodies(kicking_foot_body_name, preserve_order=True)[0][0]
    return robot.data.body_link_pos_w[:, foot_id], robot.data.body_link_lin_vel_w[:, foot_id], foot_id


def _init_kick_buffers(env: ManagerBasedRlEnv):
    latched = getattr(env, "_kick_contact_latched", None)
    if latched is None or latched.shape[0] != env.num_envs:
        env._kick_contact_latched = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    contact_step = getattr(env, "_kick_contact_step", None)
    if contact_step is None or contact_step.shape[0] != env.num_envs:
        env._kick_contact_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
    start_root = getattr(env, "_kick_start_root_pos", None)
    if start_root is None or start_root.shape[0] != env.num_envs:
        env._kick_start_root_pos = torch.zeros((env.num_envs, 3), device=env.device)
    ball_vel_at_contact = getattr(env, "_kick_ball_vel_at_contact", None)
    if ball_vel_at_contact is None or ball_vel_at_contact.shape[0] != env.num_envs:
        env._kick_ball_vel_at_contact = torch.zeros((env.num_envs, 2), device=env.device)


def _kick_contact_latched(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env._kick_contact_latched


def _kick_contact_step(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env._kick_contact_step


def _kick_start_root_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env._kick_start_root_pos


def _kick_ball_vel_at_contact(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env._kick_ball_vel_at_contact


def _post_kick_gate(env: ManagerBasedRlEnv, delay_steps: int = 6, window_steps: int = 40) -> torch.Tensor:
    _init_kick_buffers(env)
    contact_latched = _kick_contact_latched(env)
    contact_step = _kick_contact_step(env)
    valid_contact = contact_step >= 0
    elapsed = env.episode_length_buf.long() - torch.where(valid_contact, contact_step, env.episode_length_buf.long())
    active = contact_latched & (elapsed >= delay_steps) & (elapsed <= window_steps)
    if window_steps > delay_steps:
        t = (elapsed.float() - float(delay_steps)) / float(max(window_steps - delay_steps, 1))
        decay = (1.0 - t).clamp(0.0, 1.0)
    else:
        decay = torch.ones_like(elapsed, dtype=torch.float32)
    return active.float() * decay


def _update_kick_state(
    env: ManagerBasedRlEnv,
    kicking_foot_body_name: str = "right_ankle_roll_link",
    contact_distance: float = 0.13,
    min_approach_speed: float = 0.15,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    _init_kick_buffers(env)
    ball, robot = _get_assets(env, ball_cfg, robot_cfg)
    foot_pos_w, foot_vel_w, _ = _get_kick_foot_state(env, kicking_foot_body_name, robot_cfg)

    just_reset = env.episode_length_buf == 0
    latched = _kick_contact_latched(env)
    contact_step = _kick_contact_step(env)
    start_root = _kick_start_root_pos(env)
    ball_vel_at_contact = _kick_ball_vel_at_contact(env)

    latched = torch.where(just_reset, torch.zeros_like(latched), latched)
    contact_step = torch.where(just_reset, torch.full_like(contact_step, -1), contact_step)
    start_root[just_reset] = robot.data.root_link_pos_w[just_reset]
    ball_vel_at_contact[just_reset] = 0.0

    foot_to_ball = ball.data.root_link_pos_w - foot_pos_w
    dist = torch.linalg.vector_norm(foot_to_ball, dim=1)
    foot_to_ball_dir = foot_to_ball / dist.unsqueeze(-1).clamp(min=1e-6)
    approach_speed = torch.sum(foot_vel_w * foot_to_ball_dir, dim=1)

    new_contact = (dist <= contact_distance) & (approach_speed >= min_approach_speed)
    first_contact = new_contact & (~latched)

    latched = latched | new_contact
    contact_step[first_contact] = env.episode_length_buf[first_contact].long()
    if first_contact.any():
        ball_vel_at_contact[first_contact] = ball.data.root_link_lin_vel_w[first_contact, :2]

    env._kick_contact_latched = latched
    env._kick_contact_step = contact_step
    env._kick_start_root_pos = start_root
    env._kick_ball_vel_at_contact = ball_vel_at_contact


# ---------------------------------------------------------------------------
# Gate functions
# ---------------------------------------------------------------------------

def kick_distance_gate(
    env: ManagerBasedRlEnv,
    kicking_foot_body_name: str = "right_ankle_roll_link",
    near_distance: float = 0.14,
    far_distance: float = 0.42,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    ball, _ = _get_assets(env, ball_cfg, robot_cfg)
    foot_pos_w, _, _ = _get_kick_foot_state(env, kicking_foot_body_name, robot_cfg)
    dist = torch.linalg.vector_norm(ball.data.root_link_pos_w[:, :2] - foot_pos_w[:, :2], dim=1)
    return ((far_distance - dist) / max(far_distance - near_distance, 1e-6)).clamp(0.0, 1.0)


def kick_phase_gate(
    env: ManagerBasedRlEnv,
    command_name: str = "kick_motion",
    strike_phase_window: tuple[float, float] = (0.35, 0.62),
) -> torch.Tensor:
    command = cast(KickMotionCommand, env.command_manager.get_term(command_name))
    phase = command.phase
    start, end = strike_phase_window
    inside = ((phase >= start) & (phase <= end)).float()
    pre_ramp = ((phase - (start - 0.08)) / 0.08).clamp(0.0, 1.0)
    post_ramp = (((end + 0.08) - phase) / 0.08).clamp(0.0, 1.0)
    return torch.maximum(inside, torch.minimum(pre_ramp, post_ramp))


def kick_leg_velocity_gate(
    env: ManagerBasedRlEnv,
    target_mode: str = "ball",
    kicking_foot_body_name: str = "right_ankle_roll_link",
    command_name: str = "ball_target_velocity",
    v_low: float = 0.0,
    v_high: float = 0.8,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    ball, _ = _get_assets(env, ball_cfg, robot_cfg)
    foot_pos_w, foot_vel_w, _ = _get_kick_foot_state(env, kicking_foot_body_name, robot_cfg)
    if target_mode == "goal":
        goal_vec = env.command_manager.get_command(command_name)[:, :2]
        target_dir = goal_vec / torch.linalg.vector_norm(goal_vec, dim=1, keepdim=True).clamp(min=1e-6)
    else:
        ball_vec = ball.data.root_link_pos_w[:, :2] - foot_pos_w[:, :2]
        target_dir = ball_vec / torch.linalg.vector_norm(ball_vec, dim=1, keepdim=True).clamp(min=1e-6)
    projected_speed = torch.sum(foot_vel_w[:, :2] * target_dir, dim=1)
    positive_speed = torch.clamp(projected_speed - v_low, min=0.0)
    gate = (positive_speed / max(v_high - v_low, 1e-6)).clamp(0.0, 1.0)
    return gate * gate


def kick_contact_gate(
    env: ManagerBasedRlEnv,
    kicking_foot_body_name: str = "right_ankle_roll_link",
    contact_distance: float = 0.13,
    min_approach_speed: float = 0.15,
    hold_steps: int = 6,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    _update_kick_state(env, kicking_foot_body_name, contact_distance, min_approach_speed, ball_cfg, robot_cfg)
    contact_latched = _kick_contact_latched(env)
    contact_step = _kick_contact_step(env)
    in_contact = contact_latched.float()
    valid_contact = contact_step >= 0
    elapsed = env.episode_length_buf.long() - torch.where(valid_contact, contact_step, env.episode_length_buf.long())
    decay = (1.0 - elapsed.float() / max(hold_steps, 1)).clamp(0.0, 1.0)
    return in_contact * decay


# ---------------------------------------------------------------------------
# Core kick rewards
# ---------------------------------------------------------------------------

def reward_first_clean_contact_bonus(
    env: ManagerBasedRlEnv,
    kicking_foot_body_name: str = "right_ankle_roll_link",
    contact_distance: float = 0.13,
    min_approach_speed: float = 0.15,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    supporting_foot_body_name: str = "left_ankle_roll_link",
    min_bad_body_distance: float = 0.16,
) -> torch.Tensor:
    _update_kick_state(env, kicking_foot_body_name, contact_distance, min_approach_speed, ball_cfg, robot_cfg)
    contact_step = _kick_contact_step(env)
    on_first_contact_step = (contact_step >= 0) & (contact_step == env.episode_length_buf.long())
    bad = penalty_bad_ball_contact(env, kicking_foot_body_name=kicking_foot_body_name, supporting_foot_body_name=supporting_foot_body_name, min_distance=min_bad_body_distance, ball_cfg=ball_cfg, robot_cfg=robot_cfg)
    clean_mask = (bad < 0.5).float()
    return on_first_contact_step.float() * clean_mask


def reward_approach_ball(
    env: ManagerBasedRlEnv,
    target_distance: float = 0.24,
    std: float = 0.09,
    kick_zone_distance: float = 0.17,
    kicking_foot_body_name: str = "right_ankle_roll_link",
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    _update_kick_state(env, kicking_foot_body_name=kicking_foot_body_name, ball_cfg=ball_cfg, robot_cfg=robot_cfg)
    ball, _ = _get_assets(env, ball_cfg, robot_cfg)
    foot_pos_w, _, _ = _get_kick_foot_state(env, kicking_foot_body_name, robot_cfg)
    dist_xy = torch.linalg.vector_norm(ball.data.root_link_pos_w[:, :2] - foot_pos_w[:, :2], dim=1)
    reward = torch.exp(-torch.square((dist_xy - target_distance) / std))
    pre_kick = (~_kick_contact_latched(env)) & (dist_xy > kick_zone_distance)
    return reward * pre_kick.float()


def reward_kick_leg_swing(
    env: ManagerBasedRlEnv,
    target_mode: str = "ball",
    kicking_foot_body_name: str = "right_ankle_roll_link",
    command_name: str = "ball_target_velocity",
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    g_dist = kick_distance_gate(env, kicking_foot_body_name, ball_cfg=ball_cfg, robot_cfg=robot_cfg)
    g_phase = kick_phase_gate(env)
    g_leg = kick_leg_velocity_gate(env, target_mode=target_mode, kicking_foot_body_name=kicking_foot_body_name, command_name=command_name, ball_cfg=ball_cfg, robot_cfg=robot_cfg)
    return g_dist * g_phase * g_leg


def reward_kick_foot_contact_ball(
    env: ManagerBasedRlEnv,
    kicking_foot_body_name: str = "right_ankle_roll_link",
    contact_distance: float = 0.13,
    min_approach_speed: float = 0.15,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    _update_kick_state(env, kicking_foot_body_name, contact_distance, min_approach_speed, ball_cfg, robot_cfg)
    return _kick_contact_latched(env).float()


def reward_ball_speed(
    env: ManagerBasedRlEnv,
    max_speed: float = 8.0,
    min_speed: float = 0.2,
    kicking_foot_body_name: str = "right_ankle_roll_link",
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    ball, _ = _get_assets(env, ball_cfg, robot_cfg)
    g_contact = kick_contact_gate(env, kicking_foot_body_name=kicking_foot_body_name, ball_cfg=ball_cfg, robot_cfg=robot_cfg)
    ball_speed = torch.linalg.vector_norm(ball.data.root_link_lin_vel_w[:, :2], dim=1)
    speed_score = ((ball_speed - min_speed) / max(max_speed - min_speed, 1e-6)).clamp(0.0, 1.0)
    return g_contact * speed_score


def reward_ball_goal_direction(
    env: ManagerBasedRlEnv,
    command_name: str = "ball_target_velocity",
    angle_threshold_deg: float = 25.0,
    min_ball_speed: float = 0.3,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    ball = env.scene[ball_cfg.name]
    ball_vel = ball.data.root_link_lin_vel_w[:, :2]
    ball_speed = torch.linalg.vector_norm(ball_vel, dim=1)
    goal_vec = env.command_manager.get_command(command_name)[:, :2]
    goal_dir = goal_vec / torch.linalg.vector_norm(goal_vec, dim=1, keepdim=True).clamp(min=1e-6)
    ball_dir = ball_vel / ball_speed.unsqueeze(-1).clamp(min=1e-6)
    cos_sim = torch.sum(ball_dir * goal_dir, dim=1).clamp(-1.0, 1.0)
    cos_thr = torch.cos(torch.tensor(angle_threshold_deg * torch.pi / 180.0, device=cos_sim.device, dtype=cos_sim.dtype))
    score = ((cos_sim - cos_thr) / max(1.0 - float(cos_thr), 1e-6)).clamp(0.0, 1.0)
    return score * (ball_speed >= min_ball_speed).float()


def reward_ball_goal_speed(
    env: ManagerBasedRlEnv,
    command_name: str = "ball_target_velocity",
    max_speed: float = 8.0,
    min_speed: float = 0.2,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    ball = env.scene[ball_cfg.name]
    goal_vec = env.command_manager.get_command(command_name)[:, :2]
    goal_dir = goal_vec / torch.linalg.vector_norm(goal_vec, dim=1, keepdim=True).clamp(min=1e-6)
    goal_speed = torch.sum(ball.data.root_link_lin_vel_w[:, :2] * goal_dir, dim=1)
    return ((goal_speed - min_speed) / max(max_speed - min_speed, 1e-6)).clamp(0.0, 1.0)


def reward_ball_impulse(
    env: ManagerBasedRlEnv,
    window_steps: int = 8,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    _init_kick_buffers(env)
    ball, _ = _get_assets(env, ball_cfg)
    contact_latched = _kick_contact_latched(env)
    contact_step = _kick_contact_step(env)
    valid_step = contact_step >= 0
    elapsed = env.episode_length_buf.long() - torch.where(valid_step, contact_step, env.episode_length_buf.long())
    in_window = contact_latched & (elapsed >= 0) & (elapsed <= window_steps)
    ball_vel_now = ball.data.root_link_lin_vel_w[:, :2]
    ball_vel_pre = _kick_ball_vel_at_contact(env)
    speed_now = torch.linalg.vector_norm(ball_vel_now, dim=1)
    speed_pre = torch.linalg.vector_norm(ball_vel_pre, dim=1)
    delta_speed = (speed_now - speed_pre).clamp(min=0.0)
    return delta_speed * in_window.float()


# ---------------------------------------------------------------------------
# Post-kick recovery rewards
# ---------------------------------------------------------------------------

def reward_post_kick_upright(
    env: ManagerBasedRlEnv,
    k_theta: float = 2.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    g_b = robot.data.projected_gravity_b
    theta_tilt = torch.linalg.vector_norm(g_b[:, :2], dim=1)
    g_post = _post_kick_gate(env)
    return g_post * torch.exp(-k_theta * theta_tilt**2)


def reward_post_kick_base_height(
    env: ManagerBasedRlEnv,
    h_nom: float = 0.90,
    k_h: float = 45.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    h = robot.data.root_link_pos_w[:, 2]
    g_post = _post_kick_gate(env)
    return g_post * torch.exp(-k_h * (h - h_nom) ** 2)


def reward_kick_leg_retract(
    env: ManagerBasedRlEnv,
    kicking_foot_body_name: str = "right_ankle_roll_link",
    x_max: float = 0.10,
    k: float = 80.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    foot_id = robot.find_bodies(kicking_foot_body_name, preserve_order=True)[0][0]
    root_quat_w = robot.data.root_link_quat_w
    root_pos_w = robot.data.root_link_pos_w
    foot_pos_w = robot.data.body_link_pos_w[:, foot_id]
    foot_rel_w = foot_pos_w - root_pos_w
    foot_rel_b = quat_apply_inverse(root_quat_w, foot_rel_w)
    over = (foot_rel_b[:, 0] - x_max).clamp(min=0.0)
    g_post = _post_kick_gate(env)
    return g_post * torch.exp(-k * over**2)


def reward_post_kick_recontact(
    env: ManagerBasedRlEnv,
    kicking_foot_body_name: str = "right_ankle_roll_link",
    max_foot_height: float = 0.12,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    foot_id = robot.find_bodies(kicking_foot_body_name, preserve_order=True)[0][0]
    foot_z = robot.data.body_link_pos_w[:, foot_id, 2]
    g_post = _post_kick_gate(env)
    recontact = (foot_z <= max_foot_height).float()
    return g_post * recontact


def penalty_leg_spread(
    env: ManagerBasedRlEnv,
    kicking_foot_body_name: str = "right_ankle_roll_link",
    supporting_foot_body_name: str = "left_ankle_roll_link",
    max_dx: float = 0.30,
    max_dy: float = 0.22,
    wx: float = 1.2,
    wy: float = 1.8,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    kick_id = robot.find_bodies(kicking_foot_body_name, preserve_order=True)[0][0]
    support_id = robot.find_bodies(supporting_foot_body_name, preserve_order=True)[0][0]
    root_quat_w = robot.data.root_link_quat_w
    root_pos_w = robot.data.root_link_pos_w
    kick_pos_w = robot.data.body_link_pos_w[:, kick_id]
    support_pos_w = robot.data.body_link_pos_w[:, support_id]
    kick_rel_b = quat_apply_inverse(root_quat_w, kick_pos_w - root_pos_w)
    support_rel_b = quat_apply_inverse(root_quat_w, support_pos_w - root_pos_w)
    dx = torch.abs(kick_rel_b[:, 0] - support_rel_b[:, 0])
    dy = torch.abs(kick_rel_b[:, 1] - support_rel_b[:, 1])
    g_post = _post_kick_gate(env)
    over_x = (dx - max_dx).clamp(min=0.0)
    over_y = (dy - max_dy).clamp(min=0.0)
    excess = wx * over_x**2 + wy * over_y**2
    return g_post * excess


def reward_post_kick_velocity_damping(
    env: ManagerBasedRlEnv,
    k_v: float = 1.5,
    k_w: float = 1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    v_xy = robot.data.root_link_lin_vel_w[:, :2]
    w_z = robot.data.root_link_ang_vel_w[:, 2]
    speed_xy = torch.linalg.vector_norm(v_xy, dim=1)
    g_post = _post_kick_gate(env)
    return g_post * torch.exp(-k_v * speed_xy**2 - k_w * w_z**2)


def reward_post_kick_joint_nominal(
    env: ManagerBasedRlEnv,
    joint_names: list[str] | None = None,
    k_q: float = 4.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    if joint_names is None:
        joint_names = [
            "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
            "right_hip_pitch_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint",
        ]
    index_cache_name = "_post_kick_nominal_joint_indices"
    nominal_cache_name = "_post_kick_nominal_joint_pos"
    joint_indices = getattr(env, index_cache_name, None)
    nominal_q = getattr(env, nominal_cache_name, None)
    if joint_indices is None or nominal_q is None:
        indices = [robot.joint_names.index(name) for name in joint_names if name in robot.joint_names]
        if len(indices) == 0:
            return torch.zeros(env.num_envs, device=env.device)
        joint_indices = torch.tensor(indices, device=env.device, dtype=torch.long)
        nominal_full = robot.data.default_joint_pos[0]
        nominal_q = nominal_full[joint_indices].clone().detach()
        setattr(env, index_cache_name, joint_indices)
        setattr(env, nominal_cache_name, nominal_q)
    q = robot.data.joint_pos[:, joint_indices]
    q_nom = nominal_q.unsqueeze(0).expand_as(q)
    err = torch.linalg.vector_norm(q - q_nom, dim=1)
    g_post = _post_kick_gate(env)
    return g_post * torch.exp(-k_q * err**2)


def penalty_post_joint_limit_stronger(
    env: ManagerBasedRlEnv,
    base_scale: float = 1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    from mjlab.envs.mdp import rewards as base_rewards
    base_penalty = base_rewards.joint_pos_limits(env, asset_cfg=robot_cfg)
    g_post = _post_kick_gate(env)
    return base_scale * g_post * base_penalty


def penalty_post_kick_crouch(
    env: ManagerBasedRlEnv,
    h_stand: float = 0.90,
    k_h: float = 10.0,
    knee_bounds: tuple[float, float] = (0.2, 1.4),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    h = robot.data.root_link_pos_w[:, 2]
    knee_names = ["right_knee_joint", "left_knee_joint"]
    knee_indices = [robot.joint_names.index(n) for n in knee_names if n in robot.joint_names]
    if len(knee_indices) > 0:
        joints = robot.data.joint_pos[:, knee_indices]
        knee_lo, knee_hi = knee_bounds
        too_crouched = (joints > knee_hi).float()
        too_straight = (joints < knee_lo).float()
        knee_penalty = torch.mean(too_crouched + too_straight, dim=1)
    else:
        knee_penalty = torch.zeros(env.num_envs, device=env.device)
    delay_steps = 12
    window_steps = 90
    _init_kick_buffers(env)
    contact_latched = _kick_contact_latched(env)
    contact_step = _kick_contact_step(env)
    valid_contact = contact_step >= 0
    elapsed = env.episode_length_buf.long() - torch.where(valid_contact, contact_step, env.episode_length_buf.long())
    late_post = contact_latched & (elapsed >= delay_steps) & (elapsed <= window_steps)
    height_term = (h_stand - h).clamp(min=0.0)
    penalty = k_h * height_term + knee_penalty
    return late_post.float() * penalty


def reward_post_kick_stand_height(
    env: ManagerBasedRlEnv,
    h_target: float = 0.90,
    h_min: float = 0.55,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    h = robot.data.root_link_pos_w[:, 2]
    delay_steps = 12
    window_steps = 90
    _init_kick_buffers(env)
    contact_latched = _kick_contact_latched(env)
    contact_step = _kick_contact_step(env)
    valid_contact = contact_step >= 0
    elapsed = env.episode_length_buf.long() - torch.where(valid_contact, contact_step, env.episode_length_buf.long())
    late_post = contact_latched & (elapsed >= delay_steps) & (elapsed <= window_steps)
    denom = max(h_target - h_min, 1e-6)
    score = ((h - h_min) / denom).clamp(0.0, 1.0)
    return late_post.float() * score


def reward_post_kick_stable_stand(
    env: ManagerBasedRlEnv,
    h_stand: float = 0.88,
    k_h: float = 25.0,
    k_theta: float = 2.0,
    v_xy_scale: float = 4.0,
    w_z_scale: float = 4.0,
    min_lateral: float = 0.12,
    max_lateral: float = 0.30,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    delay_steps = 14
    window_steps = 120
    _init_kick_buffers(env)
    contact_latched = _kick_contact_latched(env)
    contact_step = _kick_contact_step(env)
    valid_contact = contact_step >= 0
    elapsed = env.episode_length_buf.long() - torch.where(valid_contact, contact_step, env.episode_length_buf.long())
    late_post = contact_latched & (elapsed >= delay_steps) & (elapsed <= window_steps)
    if not late_post.any():
        return torch.zeros(env.num_envs, device=env.device)
    g_b = robot.data.projected_gravity_b
    theta_tilt = torch.linalg.vector_norm(g_b[:, :2], dim=1)
    upright_score = torch.exp(-k_theta * theta_tilt**2)
    h = robot.data.root_link_pos_w[:, 2]
    height_score = torch.exp(-k_h * (h - h_stand) ** 2)
    v_xy = robot.data.root_link_lin_vel_w[:, :2]
    w_z = robot.data.root_link_ang_vel_w[:, 2]
    speed_xy = torch.linalg.vector_norm(v_xy, dim=1)
    vel_score = torch.exp(-v_xy_scale * speed_xy**2 - w_z_scale * w_z**2)
    kick_id = robot.find_bodies("right_ankle_roll_link", preserve_order=True)[0][0]
    support_id = robot.find_bodies("left_ankle_roll_link", preserve_order=True)[0][0]
    root_quat_w = robot.data.root_link_quat_w
    root_pos_w = robot.data.root_link_pos_w
    kick_pos_w = robot.data.body_link_pos_w[:, kick_id]
    support_pos_w = robot.data.body_link_pos_w[:, support_id]
    kick_rel_b = quat_apply_inverse(root_quat_w, kick_pos_w - root_pos_w)
    support_rel_b = quat_apply_inverse(root_quat_w, support_pos_w - root_pos_w)
    spread = torch.abs(kick_rel_b[:, 1] - support_rel_b[:, 1])
    spread_low = (spread - min_lateral).clamp(min=0.0)
    spread_high = (max_lateral - spread).clamp(min=0.0)
    in_band = (spread_low > 0.0) & (spread_high > 0.0)
    spread_score = torch.where(
        in_band, torch.ones_like(spread),
        torch.exp(-5.0 * (spread_low.clamp(min=0.0) + (-spread_high).clamp(min=0.0)) ** 2),
    )
    score = upright_score * height_score * vel_score * spread_score
    return late_post.float() * score


# ---------------------------------------------------------------------------
# Penalties
# ---------------------------------------------------------------------------

def penalty_excess_travel(
    env: ManagerBasedRlEnv,
    free_distance: float = 0.14,
    limit_distance: float = 0.70,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    _init_kick_buffers(env)
    robot = env.scene[robot_cfg.name]
    travel = torch.linalg.vector_norm(robot.data.root_link_pos_w[:, :2] - _kick_start_root_pos(env)[:, :2], dim=1)
    return ((travel - free_distance) / max(limit_distance - free_distance, 1e-6)).clamp(0.0, 1.0)


def penalty_bad_ball_contact(
    env: ManagerBasedRlEnv,
    kicking_foot_body_name: str = "right_ankle_roll_link",
    supporting_foot_body_name: str = "left_ankle_roll_link",
    min_distance: float = 0.16,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    ball, robot = _get_assets(env, ball_cfg, robot_cfg)
    keep_names = {kicking_foot_body_name, supporting_foot_body_name}
    bad_ids = [i for i, name in enumerate(robot.body_names) if name not in keep_names]
    if len(bad_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    body_pos = robot.data.body_link_pos_w[:, bad_ids, :2]
    ball_pos = ball.data.root_link_pos_w[:, None, :2]
    nearest = torch.linalg.vector_norm(body_pos - ball_pos, dim=2).min(dim=1).values
    return (nearest <= min_distance).float()


# ---------------------------------------------------------------------------
# Motion tracking rewards
# ---------------------------------------------------------------------------

def tracking_phase_weight_gate(
    env: ManagerBasedRlEnv,
    command_name: str = "kick_motion",
    pre_weight: float = 0.7,
    strike_weight: float = 1.0,
    post_weight: float = 0.12,
) -> torch.Tensor:
    command = cast(KickMotionCommand, env.command_manager.get_term(command_name))
    phase = command.phase
    strike_start, strike_end = command.cfg.strike_phase_window
    phase_gate = torch.full_like(phase, pre_weight)
    in_strike = (phase >= strike_start) & (phase <= strike_end)
    in_post = phase >= command.cfg.recover_phase_start
    phase_gate[in_strike] = strike_weight
    phase_gate[in_post] = post_weight
    return phase_gate


def tracking_anchor_pos_gated(
    env: ManagerBasedRlEnv, command_name: str = "kick_motion", std: float = 0.25,
    pre_weight: float = 0.7, strike_weight: float = 1.0, post_weight: float = 0.12,
) -> torch.Tensor:
    command = cast(KickMotionCommand, env.command_manager.get_term(command_name))
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    base = torch.exp(-error / std**2)
    return base * tracking_phase_weight_gate(env, command_name, pre_weight, strike_weight, post_weight)


def tracking_anchor_ori_gated(
    env: ManagerBasedRlEnv, command_name: str = "kick_motion", std: float = 0.35,
    pre_weight: float = 0.7, strike_weight: float = 1.0, post_weight: float = 0.12,
) -> torch.Tensor:
    command = cast(KickMotionCommand, env.command_manager.get_term(command_name))
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    base = torch.exp(-error / std**2)
    return base * tracking_phase_weight_gate(env, command_name, pre_weight, strike_weight, post_weight)


def tracking_body_pos_gated(
    env: ManagerBasedRlEnv, command_name: str = "kick_motion", std: float = 0.25,
    body_names: list[str] | None = None,
    pre_weight: float = 0.7, strike_weight: float = 1.0, post_weight: float = 0.12,
) -> torch.Tensor:
    command = cast(KickMotionCommand, env.command_manager.get_term(command_name))
    if body_names is None:
        body_indexes = list(range(len(command.cfg.body_names)))
    else:
        body_indexes = [i for i, name in enumerate(command.cfg.body_names) if name in body_names]
    if len(body_indexes) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    err = torch.sum(torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1)
    base = torch.exp(-err.mean(dim=-1) / std**2)
    return base * tracking_phase_weight_gate(env, command_name, pre_weight, strike_weight, post_weight)


def tracking_body_ori_gated(
    env: ManagerBasedRlEnv, command_name: str = "kick_motion", std: float = 0.35,
    body_names: list[str] | None = None,
    pre_weight: float = 0.7, strike_weight: float = 1.0, post_weight: float = 0.12,
) -> torch.Tensor:
    command = cast(KickMotionCommand, env.command_manager.get_term(command_name))
    if body_names is None:
        body_indexes = list(range(len(command.cfg.body_names)))
    else:
        body_indexes = [i for i, name in enumerate(command.cfg.body_names) if name in body_names]
    if len(body_indexes) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    err = quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes]) ** 2
    base = torch.exp(-err.mean(dim=-1) / std**2)
    return base * tracking_phase_weight_gate(env, command_name, pre_weight, strike_weight, post_weight)


def tracking_body_vel_gated(
    env: ManagerBasedRlEnv, command_name: str = "kick_motion",
    std_lin: float = 1.2, std_ang: float = 2.8,
    body_names: list[str] | None = None,
    pre_weight: float = 0.7, strike_weight: float = 1.0, post_weight: float = 0.12,
) -> torch.Tensor:
    command = cast(KickMotionCommand, env.command_manager.get_term(command_name))
    if body_names is None:
        body_indexes = list(range(len(command.cfg.body_names)))
    else:
        body_indexes = [i for i, name in enumerate(command.cfg.body_names) if name in body_names]
    if len(body_indexes) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    err_lin = torch.sum(torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1)
    err_ang = torch.sum(torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1)
    lin = torch.exp(-err_lin.mean(dim=-1) / std_lin**2)
    ang = torch.exp(-err_ang.mean(dim=-1) / std_ang**2)
    return 0.5 * (lin + ang) * tracking_phase_weight_gate(env, command_name, pre_weight, strike_weight, post_weight)


# ---------------------------------------------------------------------------
# Right support foot anti-toe terms
# ---------------------------------------------------------------------------

def _right_support_ankle_pitch(env: ManagerBasedRlEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot = env.scene[robot_cfg.name]
    name = "right_ankle_pitch_joint"
    if name not in robot.joint_names:
        return None, None
    idx = robot.joint_names.index(name)
    return robot.data.joint_pos[:, idx], idx


def _right_support_ankle_roll(env: ManagerBasedRlEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot = env.scene[robot_cfg.name]
    name = "right_ankle_roll_joint"
    if name not in robot.joint_names:
        return None, None
    idx = robot.joint_names.index(name)
    return robot.data.joint_pos[:, idx], idx


def _right_knee_joint_pos(env: ManagerBasedRlEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot = env.scene[robot_cfg.name]
    name = "right_knee_joint"
    if name not in robot.joint_names:
        return None
    idx = robot.joint_names.index(name)
    return robot.data.joint_pos[:, idx]


def _right_support_enable_gate(env: ManagerBasedRlEnv, fz_min: float = 5.0) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def _late_post_kick_gate(env: ManagerBasedRlEnv, delay_steps: int = 12, window_steps: int = 90) -> torch.Tensor:
    _init_kick_buffers(env)
    contact_latched = _kick_contact_latched(env)
    contact_step = _kick_contact_step(env)
    valid_contact = contact_step >= 0
    elapsed = env.episode_length_buf.long() - torch.where(valid_contact, contact_step, env.episode_length_buf.long())
    return (contact_latched & (elapsed >= delay_steps) & (elapsed <= window_steps)).float()


def _support_geometry_gate(
    env: ManagerBasedRlEnv, command_name: str = "kick_motion", recover_phase_start: float = 0.30,
) -> torch.Tensor:
    g_post = _post_kick_gate(env)
    try:
        command = cast(KickMotionCommand, env.command_manager.get_term(command_name))
        phase_gate = (command.phase >= recover_phase_start).float()
    except Exception:
        phase_gate = torch.zeros(env.num_envs, device=env.device)
    return torch.maximum(g_post, phase_gate)


def _right_support_foot_toe_heel_heights(
    env: ManagerBasedRlEnv,
    foot_body_name: str = "right_ankle_roll_link",
    toe_offset_x: float = 0.11,
    heel_offset_x: float = -0.09,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene[robot_cfg.name]
    foot_id = robot.find_bodies(foot_body_name, preserve_order=True)[0][0]
    foot_pos_w = robot.data.body_link_pos_w[:, foot_id]
    foot_quat_w = robot.data.body_link_quat_w[:, foot_id]
    toe_off = torch.zeros_like(foot_pos_w)
    heel_off = torch.zeros_like(foot_pos_w)
    toe_off[:, 0] = toe_offset_x
    heel_off[:, 0] = heel_offset_x
    toe_pos_w = foot_pos_w + quat_apply(foot_quat_w, toe_off)
    heel_pos_w = foot_pos_w + quat_apply(foot_quat_w, heel_off)
    return toe_pos_w[:, 2], heel_pos_w[:, 2]


def penalty_right_ankle_pitch_staged(
    env: ManagerBasedRlEnv,
    th1_deg: float = 6.0,
    th2_deg: float = 15.0,
    th3_deg: float = 25.0,
    w1: float = 0.02,
    w2: float = 0.06,
    w3: float = 0.15,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    pitch, _ = _right_support_ankle_pitch(env, robot_cfg)
    if pitch is None:
        return torch.zeros(env.num_envs, device=env.device)
    deg = pitch * (180.0 / torch.pi)
    p1 = (deg - th1_deg).clamp(min=0.0) ** 2
    p2 = (deg - th2_deg).clamp(min=0.0) ** 2
    p3 = (deg - th3_deg).clamp(min=0.0) ** 2
    pen = w1 * p1 + w2 * p2 + w3 * p3
    gate = _right_support_enable_gate(env, fz_min=3.0)
    return gate * pen


def penalty_right_toe_only_contact(
    env: ManagerBasedRlEnv,
    margin: float = 0.010,
    toe_only_weight: float = 1.0,
    center_tol: float = 0.004,
    foot_body_name: str = "right_ankle_roll_link",
    toe_offset_x: float = 0.11,
    heel_offset_x: float = -0.09,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    foot_id = robot.find_bodies(foot_body_name, preserve_order=True)[0][0]
    toe_h, heel_h = _right_support_foot_toe_heel_heights(env, foot_body_name=foot_body_name, toe_offset_x=toe_offset_x, heel_offset_x=heel_offset_x, robot_cfg=robot_cfg)
    center_h = robot.data.body_link_pos_w[:, foot_id, 2]
    geo_pen = (heel_h - toe_h - margin).clamp(min=0.0) ** 2
    toe_lower_center = (center_h - toe_h - center_tol).clamp(min=0.0)
    heel_lower_center = (center_h - heel_h - center_tol).clamp(min=0.0)
    toe_only = (toe_lower_center - heel_lower_center).clamp(min=0.0)
    gate = _right_support_enable_gate(env, fz_min=3.0)
    return gate * (geo_pen + toe_only_weight * toe_only)


def penalty_right_toe_dominant_force(
    env: ManagerBasedRlEnv,
    ratio_th: float = 1.5,
    center_tol: float = 0.004,
    foot_body_name: str = "right_ankle_roll_link",
    toe_offset_x: float = 0.11,
    heel_offset_x: float = -0.09,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    foot_id = robot.find_bodies(foot_body_name, preserve_order=True)[0][0]
    toe_h, heel_h = _right_support_foot_toe_heel_heights(env, foot_body_name=foot_body_name, toe_offset_x=toe_offset_x, heel_offset_x=heel_offset_x, robot_cfg=robot_cfg)
    center_h = robot.data.body_link_pos_w[:, foot_id, 2]
    toe_load_proxy = (center_h - toe_h - center_tol).clamp(min=0.0)
    heel_load_proxy = (center_h - heel_h - center_tol).clamp(min=0.0)
    ratio = toe_load_proxy / (heel_load_proxy + 1.0e-6)
    pen = (ratio - ratio_th).clamp(min=0.0) ** 2
    gate = _right_support_enable_gate(env, fz_min=3.0)
    return gate * pen


def reward_right_foot_parallel(
    env: ManagerBasedRlEnv,
    foot_body_name: str = "right_ankle_roll_link",
    k: float = 10.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    foot_id = robot.find_bodies(foot_body_name, preserve_order=True)[0][0]
    foot_quat_w = robot.data.body_link_quat_w[:, foot_id]
    z_axis = torch.zeros((foot_quat_w.shape[0], 3), device=env.device, dtype=foot_quat_w.dtype)
    z_axis[:, 2] = 1.0
    z_w = quat_apply(foot_quat_w, z_axis)
    cos_up = z_w[:, 2].clamp(-1.0, 1.0)
    gate = _right_support_enable_gate(env, fz_min=3.0)
    return gate * torch.exp(-k * (1.0 - cos_up))


def penalty_support_toe_stance(
    env: ManagerBasedRlEnv, ankle_pitch_threshold: float = 0.35,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    g_post = _post_kick_gate(env)
    pitch, _ = _right_support_ankle_pitch(env, robot_cfg)
    if pitch is None:
        return torch.zeros(env.num_envs, device=env.device)
    excess = (pitch - ankle_pitch_threshold).clamp(min=0.0)
    return g_post * excess


def reward_support_foot_flat_contact(
    env: ManagerBasedRlEnv, neutral_pitch: float = 0.0, k: float = 8.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    g_post = _post_kick_gate(env)
    pitch, _ = _right_support_ankle_pitch(env, robot_cfg)
    if pitch is None:
        return torch.zeros(env.num_envs, device=env.device)
    return g_post * torch.exp(-k * (pitch - neutral_pitch) ** 2)


def reward_support_foot_stability(
    env: ManagerBasedRlEnv, support_foot_body_name: str = "right_ankle_roll_link",
    k_ang: float = 2.0, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    body_id = robot.find_bodies(support_foot_body_name, preserve_order=True)[0][0]
    ang_vel = robot.data.body_link_ang_vel_w[:, body_id]
    g_post = _support_geometry_gate(env)
    return g_post * torch.exp(-k_ang * torch.sum(ang_vel**2, dim=1))


def penalty_support_knee_drop(
    env: ManagerBasedRlEnv, knee_flex_threshold: float = 1.25,
    delay_steps: int = 18, window_steps: int = 100,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    knee_pos = _right_knee_joint_pos(env, robot_cfg)
    if knee_pos is None:
        return torch.zeros(env.num_envs, device=env.device)
    late = _late_post_kick_gate(env, delay_steps=delay_steps, window_steps=window_steps)
    excess = (knee_pos - knee_flex_threshold).clamp(min=0.0)
    return late * excess


def _yaw_from_quat_w(quat_w: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_w[:, 0], quat_w[:, 1], quat_w[:, 2], quat_w[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return (x + torch.pi) % (2.0 * torch.pi) - torch.pi


def reward_support_foot_parallel(
    env: ManagerBasedRlEnv, foot_body_name: str = "right_ankle_roll_link", k: float = 10.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    foot_id = robot.find_bodies(foot_body_name, preserve_order=True)[0][0]
    foot_quat_w = robot.data.body_link_quat_w[:, foot_id]
    z_axis = torch.zeros((foot_quat_w.shape[0], 3), device=env.device, dtype=foot_quat_w.dtype)
    z_axis[:, 2] = 1.0
    z_w = quat_apply(foot_quat_w, z_axis)
    cos_up = z_w[:, 2].clamp(-1.0, 1.0)
    g_post = _support_geometry_gate(env)
    return g_post * torch.exp(-k * (1.0 - cos_up))


def reward_support_foot_parallel_rp(
    env: ManagerBasedRlEnv, k_roll: float = 10.0, k_pitch: float = 14.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    roll, _ = _right_support_ankle_roll(env, robot_cfg)
    pitch, _ = _right_support_ankle_pitch(env, robot_cfg)
    if roll is None or pitch is None:
        return torch.zeros(env.num_envs, device=env.device)
    g = _support_geometry_gate(env)
    return g * torch.exp(-k_roll * roll**2 - k_pitch * pitch**2)


def penalty_support_foot_toe_scrape(
    env: ManagerBasedRlEnv,
    foot_body_name: str = "right_ankle_roll_link",
    margin: float = 0.01, toe_only_weight: float = 0.8,
    toe_offset_x: float = 0.11, heel_offset_x: float = -0.09, center_tol: float = 0.004,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    foot_id = robot.find_bodies(foot_body_name, preserve_order=True)[0][0]
    toe_h, heel_h = _right_support_foot_toe_heel_heights(env, foot_body_name=foot_body_name, toe_offset_x=toe_offset_x, heel_offset_x=heel_offset_x, robot_cfg=robot_cfg)
    center_h = robot.data.body_link_pos_w[:, foot_id, 2]
    diff = (heel_h - toe_h - margin).clamp(min=0.0)
    geo_pen = diff**2
    toe_lower_center = (center_h - toe_h - center_tol).clamp(min=0.0)
    heel_lower_center = (center_h - heel_h - center_tol).clamp(min=0.0)
    toe_only = (toe_lower_center - heel_lower_center).clamp(min=0.0)
    g = _support_geometry_gate(env)
    return g * (geo_pen + toe_only_weight * toe_only)


def reward_support_foot_yaw(
    env: ManagerBasedRlEnv, foot_body_name: str = "right_ankle_roll_link", k: float = 12.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    foot_id = robot.find_bodies(foot_body_name, preserve_order=True)[0][0]
    foot_quat_w = robot.data.body_link_quat_w[:, foot_id]
    base_quat_w = robot.data.root_link_quat_w
    foot_yaw = _yaw_from_quat_w(foot_quat_w)
    base_yaw = _yaw_from_quat_w(base_quat_w)
    err = _wrap_to_pi(foot_yaw - base_yaw)
    g_post = _post_kick_gate(env)
    return g_post * torch.exp(-k * err**2)


def penalty_support_foot_stumble(
    env: ManagerBasedRlEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    foot_body_name: str = "right_ankle_roll_link",
    mu: float = 0.35, fz_min: float = 8.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # Contact sensor may not be available in mjlab; degrade gracefully
    return torch.zeros(env.num_envs, device=env.device)


# ---------------------------------------------------------------------------
# NEW: Tiptoe + post-kick fix terms (per kicking_mjlab_port.md)
# ---------------------------------------------------------------------------

def reward_support_foot_flat_contact_early(
    env: ManagerBasedRlEnv,
    foot_body_name: str = "right_ankle_roll_link",
    toe_offset_x: float = 0.11,
    heel_offset_x: float = -0.09,
    ankle_pitch_threshold_deg: float = 5.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalizes heel-toe height diff + ankle pitch exceeding threshold.

    Active from Bootstrap/Stage A (not just B/C). Catches tiptoe posture
    during approach and strike phases.
    """
    toe_h, heel_h = _right_support_foot_toe_heel_heights(
        env, foot_body_name=foot_body_name,
        toe_offset_x=toe_offset_x, heel_offset_x=heel_offset_x, robot_cfg=robot_cfg,
    )
    pitch, _ = _right_support_ankle_pitch(env, robot_cfg)
    if pitch is None:
        return torch.zeros(env.num_envs, device=env.device)

    # Heel-toe height differential penalty
    height_diff = (heel_h - toe_h).clamp(min=0.0)

    # Ankle pitch penalty (positive pitch = toe down)
    threshold_rad = ankle_pitch_threshold_deg * torch.pi / 180.0
    pitch_excess = (pitch - threshold_rad).clamp(min=0.0)

    # Combined penalty magnitude (use negative weight in curriculum)
    return height_diff + pitch_excess


def reward_post_kick_upright_early(
    env: ManagerBasedRlEnv,
    k_theta: float = 2.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Lightweight upright reward active even in Stage A.

    Prevents policy from learning extreme tilt as acceptable.
    Uses same projected_gravity_b tilt metric as reward_post_kick_upright
    but with a simpler gate (any post-kick phase, not just delayed window).
    """
    robot = env.scene[robot_cfg.name]
    g_b = robot.data.projected_gravity_b
    theta_tilt = torch.linalg.vector_norm(g_b[:, :2], dim=1)

    # Simple post-kick gate: active after any kick contact
    _init_kick_buffers(env)
    contact_latched = _kick_contact_latched(env)
    contact_step = _kick_contact_step(env)
    valid_contact = contact_step >= 0
    elapsed = env.episode_length_buf.long() - torch.where(valid_contact, contact_step, env.episode_length_buf.long())
    # Active after a short delay (4 steps) to not interfere with follow-through
    active = contact_latched & (elapsed >= 4)

    return active.float() * torch.exp(-k_theta * theta_tilt**2)


# ---------------------------------------------------------------------------
# Metric / diagnostic terms (weight=0)
# ---------------------------------------------------------------------------

def metric_gate_dist(env: ManagerBasedRlEnv) -> torch.Tensor:
    return kick_distance_gate(env)

def metric_gate_phase(env: ManagerBasedRlEnv) -> torch.Tensor:
    return kick_phase_gate(env)

def metric_gate_leg(env: ManagerBasedRlEnv) -> torch.Tensor:
    return kick_leg_velocity_gate(env)

def metric_gate_contact(env: ManagerBasedRlEnv) -> torch.Tensor:
    return kick_contact_gate(env)

def metric_curriculum_stage(env: ManagerBasedRlEnv) -> torch.Tensor:
    stage = getattr(env, "_kick_curr_stage", "A")
    v = float({"A": 0.0, "B": 1.0, "C": 2.0}.get(stage, 0.0))
    return torch.full((env.num_envs,), v, device=env.device, dtype=torch.float32)

def metric_curriculum_steps_in_stage(env: ManagerBasedRlEnv) -> torch.Tensor:
    step = int(env.common_step_counter)
    enter = int(getattr(env, "_kick_curr_stage_enter_step", step))
    return torch.full((env.num_envs,), float(max(0, step - enter)), device=env.device, dtype=torch.float32)

def metric_curriculum_promotion_fired(env: ManagerBasedRlEnv) -> torch.Tensor:
    v = float(getattr(env, "_kick_promotion_fired", 0.0))
    return torch.full((env.num_envs,), v, device=env.device, dtype=torch.float32)

def metric_curriculum_demotion_fired(env: ManagerBasedRlEnv) -> torch.Tensor:
    v = float(getattr(env, "_kick_demotion_fired", 0.0))
    return torch.full((env.num_envs,), v, device=env.device, dtype=torch.float32)

def metric_curriculum_recovery_quality(env: ManagerBasedRlEnv) -> torch.Tensor:
    v = float(getattr(env, "_kick_recovery_quality", 0.0))
    return torch.full((env.num_envs,), v, device=env.device, dtype=torch.float32)
