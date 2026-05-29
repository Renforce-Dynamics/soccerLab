"""Termination terms for G1 kicking task (mjlab backend).

Ported from G1_kicking/kick_task/mdp/terminations.py with:
  - ManagerBasedRLEnv -> ManagerBasedRlEnv
  - root_pos_w -> root_link_pos_w
  - root_lin_vel_w -> root_link_lin_vel_w
  - Uses mjlab.envs.mdp.terminations as base
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import terminations as base_terms
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .rewards import kick_contact_gate


def _post_kick_elapsed(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kick_contact_gate(env)
    contact_latched = getattr(env, "_kick_contact_latched", None)
    contact_step = getattr(env, "_kick_contact_step", None)
    if contact_latched is None or contact_step is None:
        zero_bool = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        return zero_bool, torch.zeros(env.num_envs, device=env.device, dtype=torch.long), zero_bool

    valid_step = contact_step >= 0
    elapsed = env.episode_length_buf.long() - torch.where(
        valid_step, contact_step, env.episode_length_buf.long()
    )
    return contact_latched, elapsed, valid_step


def early_terminate_after_kick(
    env: ManagerBasedRlEnv,
    eval_window_steps: int = 16,
    min_ball_speed: float = 0.6,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    kick_contact_gate(env)
    contact_latched = getattr(env, "_kick_contact_latched", None)
    contact_step = getattr(env, "_kick_contact_step", None)
    if contact_latched is None or contact_step is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    ball = env.scene[ball_cfg.name]
    valid_step = contact_step >= 0
    elapsed = env.episode_length_buf.long() - torch.where(valid_step, contact_step, env.episode_length_buf.long())
    window_ready = elapsed >= eval_window_steps

    ball_speed = torch.linalg.vector_norm(ball.data.root_link_lin_vel_w[:, :2], dim=1)
    evaluated = contact_latched & window_ready
    return evaluated & (ball_speed >= min_ball_speed)


def bad_ball_stuck(
    env: ManagerBasedRlEnv,
    min_eval_steps: int = 12,
    max_ball_speed: float = 0.12,
    max_ball_distance: float = 0.20,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    kick_contact_gate(env)
    contact_latched = getattr(env, "_kick_contact_latched", None)
    contact_step = getattr(env, "_kick_contact_step", None)
    if contact_latched is None or contact_step is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    ball = env.scene[ball_cfg.name]
    robot = env.scene[robot_cfg.name]

    valid_step = contact_step >= 0
    elapsed = env.episode_length_buf.long() - torch.where(valid_step, contact_step, env.episode_length_buf.long())
    after_contact = contact_latched & (elapsed >= min_eval_steps)

    ball_speed = torch.linalg.vector_norm(ball.data.root_link_lin_vel_w[:, :2], dim=1)
    dist = torch.linalg.vector_norm(ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2], dim=1)
    stuck = (ball_speed <= max_ball_speed) & (dist <= max_ball_distance)
    return after_contact & stuck


def kick_root_height_below_minimum_with_window(
    env: ManagerBasedRlEnv,
    minimum_height: float = 0.22,
    grace_window_s: float = 3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    stage = getattr(env, "_kick_curr_stage", "A")
    if stage == "A":
        return base_terms.root_height_below_minimum(env, minimum_height=minimum_height, asset_cfg=asset_cfg)

    contact_latched, elapsed, valid_step = _post_kick_elapsed(env)
    has_contact = contact_latched & valid_step
    if not has_contact.any():
        return base_terms.root_height_below_minimum(env, minimum_height=minimum_height, asset_cfg=asset_cfg)

    step_dt = getattr(env, "step_dt", 0.02)
    grace_steps = int(grace_window_s / max(step_dt, 1e-6))
    in_grace = has_contact & (elapsed < grace_steps)

    base_term = base_terms.root_height_below_minimum(env, minimum_height=minimum_height, asset_cfg=asset_cfg)
    return base_term & ~in_grace


def kick_bad_orientation_with_window(
    env: ManagerBasedRlEnv,
    limit_angle: float = 1.0,
    grace_window_s: float = 3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    stage = getattr(env, "_kick_curr_stage", "A")
    if stage == "A":
        return base_terms.bad_orientation(env, limit_angle=limit_angle, asset_cfg=asset_cfg)

    contact_latched, elapsed, valid_step = _post_kick_elapsed(env)
    has_contact = contact_latched & valid_step
    if not has_contact.any():
        return base_terms.bad_orientation(env, limit_angle=limit_angle, asset_cfg=asset_cfg)

    step_dt = getattr(env, "step_dt", 0.02)
    grace_steps = int(grace_window_s / max(step_dt, 1e-6))
    in_grace = has_contact & (elapsed < grace_steps)

    base_term = base_terms.bad_orientation(env, limit_angle=limit_angle, asset_cfg=asset_cfg)
    return base_term & ~in_grace
