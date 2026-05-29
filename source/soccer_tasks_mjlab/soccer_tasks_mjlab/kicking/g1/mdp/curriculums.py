"""Curriculum functions for G1 kicking task (mjlab backend).

Ported from G1_kicking/kick_task/mdp/curriculums.py with tiptoe + post-kick fixes:
  - reward_support_foot_flat_contact_early active from Bootstrap/Stage A
  - reward_post_kick_upright_early active from Stage A (light)
  - penalty_right_ankle_pitch_staged strengthened in A: -2.0 -> -3.0
  - reward_support_foot_parallel_rp active in Stage A (weight 1.0)
  - penalty_post_kick_crouch: -0.5/A, -1.5/B1, -3.0/B2, -4.0/C
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch


_STAGE_A = "A"
_STAGE_B = "B"
_STAGE_C = "C"

_BOOTSTRAP_STAGE_A0 = "A0"
_BOOTSTRAP_STAGE_A1 = "A1"

_TRACKING_BODY_NAMES_STAGE_A = [
    "pelvis",
    "torso_link",
    "right_hip_roll_link",
    "right_hip_pitch_link",
    "right_knee_link",
]
_TRACKING_BODY_NAMES_STAGE_BC = [
    "pelvis",
    "torso_link",
    "right_hip_roll_link",
    "right_hip_pitch_link",
    "right_knee_link",
    "right_ankle_roll_link",
]


def _episode_avg(env, env_ids_t: torch.Tensor, term_name: str) -> torch.Tensor:
    return torch.mean(env.reward_manager._episode_sums[term_name][env_ids_t]) / env.max_episode_length_s


def _set_weight(env, term_name: str, value: float):
    term = env.reward_manager.get_term_cfg(term_name)
    term.weight = float(value)


def _set_term_params(env, term_name: str, **kwargs):
    term = env.reward_manager.get_term_cfg(term_name)
    if term.params is None:
        term.params = {}
    term.params.update(kwargs)


def _set_weight_if_exists(env, term_name: str, value: float):
    """Set reward weight if term exists; no-op otherwise."""
    try:
        _set_weight(env, term_name, value)
    except (ValueError, KeyError):
        return


def _apply_stage_weights(env, stage: str):
    """Apply reward weights for the given curriculum stage (A/B/C).

    Integrates tiptoe + post-kick fixes from the porting plan:
      - reward_support_foot_flat_contact_early: -2.5/A, -3.5/B, -4.5/C
      - reward_post_kick_upright_early: 0.3/A, 0.8/B, 1.2/C
      - penalty_right_ankle_pitch_staged strengthened: -3.0/A (was -2.0)
      - reward_support_foot_parallel_rp: 1.0/A (was 0.0)
      - penalty_post_kick_crouch: -0.5/A, -1.5/B1, -3.0/B2, -4.0/C
    """
    if stage == _STAGE_A:
        _set_weight(env, "reward_approach_ball", 0.5)
        _set_weight(env, "reward_kick_leg_swing", 6.0)
        _set_weight(env, "reward_kick_foot_contact_ball", 11.0)
        _set_weight(env, "reward_first_clean_contact_bonus", 4.5)
        _set_weight(env, "reward_ball_speed", 13.0)
        _set_weight(env, "reward_ball_goal_direction", 0.0)
        _set_weight(env, "reward_ball_goal_speed", 0.0)

        _set_weight(env, "tracking_anchor_pos", 0.30)
        _set_weight(env, "tracking_anchor_ori", 0.28)
        _set_weight(env, "tracking_body_pos", 0.32)
        _set_weight(env, "tracking_body_ori", 0.30)
        _set_weight(env, "tracking_body_vel", 0.22)
        _set_term_params(env, "tracking_body_pos", body_names=_TRACKING_BODY_NAMES_STAGE_A)
        _set_term_params(env, "tracking_body_ori", body_names=_TRACKING_BODY_NAMES_STAGE_A)
        _set_term_params(env, "tracking_body_vel", body_names=_TRACKING_BODY_NAMES_STAGE_A)

        _set_weight(env, "penalty_excess_travel", -0.08)
        _set_weight(env, "penalty_bad_ball_contact", -1.8)
        _set_weight(env, "action_rate", -2.0e-4)
        _set_weight(env, "joint_limit", -0.06)

        # Anti-toe constraints — STRENGTHENED per fix: -3.0 (was -2.0)
        _set_weight(env, "penalty_right_ankle_pitch_staged", -3.0)
        _set_term_params(env, "penalty_right_toe_only_contact", margin=0.010, toe_only_weight=0.8)
        _set_weight(env, "penalty_right_toe_only_contact", -2.5)
        _set_term_params(env, "penalty_right_toe_dominant_force", ratio_th=1.5)
        _set_weight(env, "penalty_right_toe_dominant_force", -1.5)
        _set_term_params(env, "reward_right_foot_parallel", k=10.0)
        _set_weight(env, "reward_right_foot_parallel", 1.5)

        # NEW tiptoe fix: reward_support_foot_flat_contact_early active in Stage A
        _set_weight(env, "reward_support_foot_flat_contact_early", -2.5)

        # NEW fix: reward_support_foot_parallel_rp active in Stage A (was 0.0)
        _set_term_params(env, "reward_support_foot_parallel_rp", k_roll=10.0, k_pitch=14.0)
        _set_weight(env, "reward_support_foot_parallel_rp", 1.0)

        # Post-kick: mostly off but NEW fixes add light shaping
        _set_weight(env, "reward_post_kick_upright", 0.0)
        _set_weight(env, "reward_post_kick_base_height", 0.0)
        _set_weight(env, "reward_kick_leg_retract", 0.0)
        _set_weight(env, "reward_post_kick_recontact", 0.0)
        _set_weight(env, "reward_post_kick_velocity_damping", 0.0)
        _set_weight(env, "reward_post_kick_joint_nominal", 0.0)
        _set_weight(env, "penalty_leg_spread", 0.0)
        _set_weight(env, "penalty_post_joint_limit_stronger", 0.0)
        _set_weight(env, "penalty_support_toe_stance", 0.0)
        _set_weight(env, "reward_support_foot_flat_contact", 0.0)
        _set_weight(env, "reward_support_foot_stability", 0.0)
        _set_weight(env, "penalty_support_knee_drop", 0.0)
        _set_weight(env, "reward_support_foot_parallel", 0.0)
        _set_weight(env, "penalty_support_foot_toe_scrape", 0.0)
        _set_weight(env, "reward_support_foot_yaw", 0.0)
        _set_weight(env, "penalty_support_foot_stumble", 0.0)

        # NEW post-kick fix: light upright reward in Stage A
        _set_weight(env, "reward_post_kick_upright_early", 0.3)
        # NEW post-kick fix: light crouch penalty in Stage A (was 0.0)
        _set_weight(env, "penalty_post_kick_crouch", -0.5)
        _set_weight(env, "reward_post_kick_stand_height", 0.0)
        _set_weight(env, "reward_post_kick_stable_stand", 0.0)

    elif stage == _STAGE_B:
        step = int(env.common_step_counter)
        enter = int(getattr(env, "_kick_curr_stage_enter_step", step))
        steps_in_stage = max(0, step - enter)
        b1_steps = 200_000

        _set_weight(env, "reward_approach_ball", 0.35)
        _set_weight(env, "reward_kick_leg_swing", 4.0)
        _set_weight(env, "reward_kick_foot_contact_ball", 8.5)
        _set_weight(env, "reward_first_clean_contact_bonus", 3.0)
        _set_weight(env, "reward_ball_speed", 7.5)
        _set_weight(env, "reward_ball_impulse", 6.0)

        _set_weight(env, "penalty_excess_travel", -0.30)
        _set_weight(env, "penalty_bad_ball_contact", -2.8)
        _set_weight(env, "action_rate", -6.0e-4)
        _set_weight(env, "joint_limit", -0.10)

        _set_weight(env, "penalty_support_toe_stance", 0.0)
        _set_weight(env, "reward_support_foot_flat_contact", 0.0)
        _set_term_params(env, "tracking_body_pos", body_names=_TRACKING_BODY_NAMES_STAGE_BC)
        _set_term_params(env, "tracking_body_ori", body_names=_TRACKING_BODY_NAMES_STAGE_BC)
        _set_term_params(env, "tracking_body_vel", body_names=_TRACKING_BODY_NAMES_STAGE_BC)

        # Anti-toe (Stage B middle strength)
        _set_weight(env, "penalty_right_ankle_pitch_staged", -3.0)
        _set_term_params(env, "penalty_right_toe_only_contact", margin=0.010, toe_only_weight=0.8)
        _set_weight(env, "penalty_right_toe_only_contact", -3.5)
        _set_term_params(env, "penalty_right_toe_dominant_force", ratio_th=1.5)
        _set_weight(env, "penalty_right_toe_dominant_force", -2.2)
        _set_term_params(env, "reward_right_foot_parallel", k=10.0)
        _set_weight(env, "reward_right_foot_parallel", 1.2)

        # NEW tiptoe fix: reward_support_foot_flat_contact_early in Stage B
        _set_weight(env, "reward_support_foot_flat_contact_early", -3.5)

        if steps_in_stage < b1_steps:
            # B1
            _set_weight(env, "tracking_anchor_pos", 0.25)
            _set_weight(env, "tracking_anchor_ori", 0.25)
            _set_weight(env, "tracking_body_pos", 0.35)
            _set_weight(env, "tracking_body_ori", 0.35)
            _set_weight(env, "tracking_body_vel", 0.20)

            _set_weight(env, "reward_ball_goal_direction", 0.2)
            _set_weight(env, "reward_ball_goal_speed", 0.2)

            _set_term_params(env, "reward_support_foot_parallel_rp", k_roll=10.0, k_pitch=14.0)
            _set_weight(env, "reward_support_foot_parallel_rp", 3.0)
            _set_weight(env, "reward_support_foot_parallel", 0.0)
            _set_weight(env, "reward_support_foot_yaw", 1.0)
            _set_weight(env, "reward_support_foot_stability", 2.0)
            _set_term_params(env, "penalty_support_foot_toe_scrape", margin=0.010, toe_only_weight=0.8)
            _set_weight(env, "penalty_support_foot_toe_scrape", -4.0)
            _set_term_params(env, "penalty_support_foot_stumble", mu=0.45, fz_min=8.0)
            _set_weight(env, "penalty_support_foot_stumble", -3.0)
            _set_weight(env, "penalty_support_knee_drop", -3.2)

            _set_term_params(env, "reward_kick_leg_retract", x_max=0.09, k=100.0)
            _set_weight(env, "reward_kick_leg_retract", 3.2)
            _set_term_params(env, "penalty_leg_spread", max_dx=0.30, max_dy=0.22, wx=1.2, wy=1.8)
            _set_weight(env, "penalty_leg_spread", -3.5)

            _set_weight(env, "reward_post_kick_upright", 1.2)
            _set_weight(env, "reward_post_kick_base_height", 1.0)
            _set_weight(env, "reward_post_kick_stand_height", 0.4)
            _set_weight(env, "reward_post_kick_stable_stand", 0.3)
            # NEW fix: penalty_post_kick_crouch B1 -1.5 (was -0.8)
            _set_weight(env, "penalty_post_kick_crouch", -1.5)
            _set_weight(env, "reward_post_kick_recontact", 2.2)
            _set_weight(env, "reward_post_kick_velocity_damping", 0.8)
            _set_weight(env, "reward_post_kick_joint_nominal", 0.8)
            _set_weight(env, "penalty_post_joint_limit_stronger", -0.18)

            # NEW post-kick fix: upright_early in B1
            _set_weight(env, "reward_post_kick_upright_early", 0.8)

        else:
            # B2
            _set_weight(env, "tracking_anchor_pos", 0.25)
            _set_weight(env, "tracking_anchor_ori", 0.25)
            _set_weight(env, "tracking_body_pos", 0.35)
            _set_weight(env, "tracking_body_ori", 0.35)
            _set_weight(env, "tracking_body_vel", 0.20)

            _set_weight(env, "reward_ball_goal_direction", 0.2)
            _set_weight(env, "reward_ball_goal_speed", 0.2)

            _set_term_params(env, "reward_support_foot_parallel_rp", k_roll=10.0, k_pitch=14.0)
            _set_weight(env, "reward_support_foot_parallel_rp", 3.0)
            _set_weight(env, "reward_support_foot_parallel", 0.0)
            _set_weight(env, "reward_support_foot_yaw", 1.1)
            _set_weight(env, "reward_support_foot_stability", 1.6)
            _set_term_params(env, "penalty_support_foot_toe_scrape", margin=0.010, toe_only_weight=0.8)
            _set_weight(env, "penalty_support_foot_toe_scrape", -4.0)
            _set_term_params(env, "penalty_support_foot_stumble", mu=0.45, fz_min=8.0)
            _set_weight(env, "penalty_support_foot_stumble", -3.0)
            _set_weight(env, "penalty_support_knee_drop", -3.0)

            _set_term_params(env, "reward_kick_leg_retract", x_max=0.09, k=100.0)
            _set_weight(env, "reward_kick_leg_retract", 3.2)
            _set_term_params(env, "penalty_leg_spread", max_dx=0.30, max_dy=0.22, wx=1.2, wy=1.8)
            _set_weight(env, "penalty_leg_spread", -3.5)

            _set_weight(env, "reward_post_kick_upright", 1.3)
            _set_weight(env, "reward_post_kick_base_height", 1.1)
            _set_weight(env, "reward_post_kick_stand_height", 0.5)
            _set_weight(env, "reward_post_kick_stable_stand", 0.4)
            # NEW fix: penalty_post_kick_crouch B2 -3.0 (was -2.4)
            _set_weight(env, "penalty_post_kick_crouch", -3.0)
            _set_weight(env, "reward_post_kick_recontact", 2.4)
            _set_weight(env, "reward_post_kick_velocity_damping", 1.2)
            _set_weight(env, "reward_post_kick_joint_nominal", 1.2)
            _set_weight(env, "penalty_post_joint_limit_stronger", -0.22)

            # NEW post-kick fix: upright_early in B2
            _set_weight(env, "reward_post_kick_upright_early", 0.8)

    # Stage C
    else:
        _set_weight(env, "reward_approach_ball", 0.30)
        _set_weight(env, "reward_kick_leg_swing", 3.5)
        _set_weight(env, "reward_kick_foot_contact_ball", 7.5)
        _set_weight(env, "reward_first_clean_contact_bonus", 2.5)
        _set_weight(env, "reward_ball_speed", 7.0)
        _set_weight(env, "reward_ball_impulse", 7.0)
        _set_weight(env, "reward_ball_goal_direction", 0.6)
        _set_weight(env, "reward_ball_goal_speed", 0.6)

        _set_weight(env, "tracking_anchor_pos", 0.35)
        _set_weight(env, "tracking_anchor_ori", 0.35)
        _set_weight(env, "tracking_body_pos", 0.45)
        _set_weight(env, "tracking_body_ori", 0.45)
        _set_weight(env, "tracking_body_vel", 0.25)
        _set_term_params(env, "tracking_body_pos", body_names=_TRACKING_BODY_NAMES_STAGE_BC)
        _set_term_params(env, "tracking_body_ori", body_names=_TRACKING_BODY_NAMES_STAGE_BC)
        _set_term_params(env, "tracking_body_vel", body_names=_TRACKING_BODY_NAMES_STAGE_BC)

        _set_weight(env, "penalty_excess_travel", -0.60)
        _set_weight(env, "penalty_bad_ball_contact", -3.2)
        _set_weight(env, "action_rate", -8.0e-4)
        _set_weight(env, "joint_limit", -0.13)

        # Anti-toe (Stage C strongest)
        _set_weight(env, "penalty_right_ankle_pitch_staged", -4.0)
        _set_term_params(env, "penalty_right_toe_only_contact", margin=0.008, toe_only_weight=1.0)
        _set_weight(env, "penalty_right_toe_only_contact", -4.5)
        _set_term_params(env, "penalty_right_toe_dominant_force", ratio_th=1.5)
        _set_weight(env, "penalty_right_toe_dominant_force", -3.0)
        _set_term_params(env, "reward_right_foot_parallel", k=10.0)
        _set_weight(env, "reward_right_foot_parallel", 1.0)

        # NEW tiptoe fix: reward_support_foot_flat_contact_early in Stage C
        _set_weight(env, "reward_support_foot_flat_contact_early", -4.5)

        _set_weight(env, "penalty_support_toe_stance", 0.0)
        _set_weight(env, "reward_support_foot_flat_contact", 0.0)
        _set_term_params(env, "reward_support_foot_parallel_rp", k_roll=12.0, k_pitch=18.0)
        _set_weight(env, "reward_support_foot_parallel_rp", 4.0)
        _set_weight(env, "reward_support_foot_parallel", 0.0)
        _set_weight(env, "reward_support_foot_yaw", 1.6)
        _set_weight(env, "reward_support_foot_stability", 2.0)
        _set_term_params(env, "penalty_support_foot_toe_scrape", margin=0.008, toe_only_weight=1.0)
        _set_weight(env, "penalty_support_foot_toe_scrape", -5.0)
        _set_term_params(env, "penalty_support_foot_stumble", mu=0.38, fz_min=5.0)
        _set_weight(env, "penalty_support_foot_stumble", -4.0)
        _set_weight(env, "penalty_support_knee_drop", -3.8)
        _set_weight(env, "reward_post_kick_upright", 2.2)
        _set_weight(env, "reward_post_kick_base_height", 1.8)
        _set_term_params(env, "reward_kick_leg_retract", x_max=0.07, k=120.0)
        _set_weight(env, "reward_kick_leg_retract", 3.8)
        _set_weight(env, "reward_post_kick_recontact", 2.4)
        _set_weight(env, "reward_post_kick_velocity_damping", 1.8)
        _set_weight(env, "reward_post_kick_joint_nominal", 1.8)
        _set_term_params(env, "penalty_leg_spread", max_dx=0.27, max_dy=0.20, wx=1.2, wy=2.0)
        _set_weight(env, "penalty_leg_spread", -4.5)
        _set_weight(env, "penalty_post_joint_limit_stronger", -0.26)
        # NEW fix: penalty_post_kick_crouch C -4.0 (was -3.0)
        _set_weight(env, "penalty_post_kick_crouch", -4.0)
        _set_weight(env, "reward_post_kick_stand_height", 1.2)
        _set_weight(env, "reward_post_kick_stable_stand", 1.0)

        # NEW post-kick fix: upright_early in C
        _set_weight(env, "reward_post_kick_upright_early", 1.2)


def _apply_bootstrap_stage_weights(env, stage: str):
    """Bootstrap A0 (stabilize) -> A1 (basic kick). Includes tiptoe fix terms."""
    if stage == _BOOTSTRAP_STAGE_A0:
        _set_weight_if_exists(env, "reward_approach_ball", 0.2)
        _set_weight_if_exists(env, "reward_kick_leg_swing", 1.0)
        _set_weight_if_exists(env, "reward_kick_foot_contact_ball", 1.0)
        _set_weight_if_exists(env, "reward_first_clean_contact_bonus", 0.5)
        _set_weight_if_exists(env, "reward_ball_speed", 1.0)
        _set_weight_if_exists(env, "reward_ball_impulse", 0.5)

        _set_weight_if_exists(env, "tracking_anchor_pos", 0.06)
        _set_weight_if_exists(env, "tracking_anchor_ori", 0.05)
        _set_weight_if_exists(env, "tracking_body_pos", 0.08)
        _set_weight_if_exists(env, "tracking_body_ori", 0.07)
        _set_weight_if_exists(env, "tracking_body_vel", 0.06)

        _set_weight_if_exists(env, "penalty_excess_travel", -0.02)
        _set_weight_if_exists(env, "penalty_bad_ball_contact", -0.8)
        _set_weight_if_exists(env, "action_rate", -5.0e-5)
        _set_weight_if_exists(env, "joint_limit", -0.03)

        # NEW tiptoe fix: active even in Bootstrap A0 (light)
        _set_weight_if_exists(env, "reward_support_foot_flat_contact_early", -1.5)

        # Keep all post-kick/anti-toe/support-foot terms off in A0
        for name in [
            "reward_ball_goal_direction",
            "reward_ball_goal_speed",
            "reward_post_kick_upright",
            "reward_post_kick_base_height",
            "reward_kick_leg_retract",
            "reward_post_kick_recontact",
            "reward_post_kick_velocity_damping",
            "reward_post_kick_joint_nominal",
            "penalty_leg_spread",
            "penalty_post_joint_limit_stronger",
            "penalty_post_kick_crouch",
            "reward_post_kick_stand_height",
            "reward_post_kick_stable_stand",
            "penalty_support_toe_stance",
            "reward_support_foot_flat_contact",
            "reward_support_foot_stability",
            "penalty_support_knee_drop",
            "reward_support_foot_parallel",
            "reward_support_foot_parallel_rp",
            "penalty_support_foot_toe_scrape",
            "reward_support_foot_yaw",
            "penalty_support_foot_stumble",
            "penalty_right_ankle_pitch_staged",
            "penalty_right_toe_only_contact",
            "penalty_right_toe_dominant_force",
            "reward_right_foot_parallel",
            "reward_post_kick_upright_early",
        ]:
            _set_weight_if_exists(env, name, 0.0)

    # Bootstrap A1: basic kick
    else:
        _set_weight_if_exists(env, "reward_approach_ball", 0.35)
        _set_weight_if_exists(env, "reward_kick_leg_swing", 4.0)
        _set_weight_if_exists(env, "reward_kick_foot_contact_ball", 6.0)
        _set_weight_if_exists(env, "reward_first_clean_contact_bonus", 3.0)
        _set_weight_if_exists(env, "reward_ball_speed", 8.0)
        _set_weight_if_exists(env, "reward_ball_impulse", 6.0)

        _set_weight_if_exists(env, "tracking_anchor_pos", 0.12)
        _set_weight_if_exists(env, "tracking_anchor_ori", 0.10)
        _set_weight_if_exists(env, "tracking_body_pos", 0.18)
        _set_weight_if_exists(env, "tracking_body_ori", 0.16)
        _set_weight_if_exists(env, "tracking_body_vel", 0.12)

        _set_weight_if_exists(env, "penalty_excess_travel", -0.02)
        _set_weight_if_exists(env, "penalty_bad_ball_contact", -1.0)
        _set_weight_if_exists(env, "action_rate", -5.0e-5)
        _set_weight_if_exists(env, "joint_limit", -0.03)

        # NEW tiptoe fix: active in Bootstrap A1 (same as Stage A weight)
        _set_weight_if_exists(env, "reward_support_foot_flat_contact_early", -2.5)

        for name in [
            "reward_ball_goal_direction",
            "reward_ball_goal_speed",
            "reward_post_kick_upright",
            "reward_post_kick_base_height",
            "reward_kick_leg_retract",
            "reward_post_kick_recontact",
            "reward_post_kick_velocity_damping",
            "reward_post_kick_joint_nominal",
            "penalty_leg_spread",
            "penalty_post_joint_limit_stronger",
            "penalty_post_kick_crouch",
            "reward_post_kick_stand_height",
            "reward_post_kick_stable_stand",
            "penalty_support_toe_stance",
            "reward_support_foot_flat_contact",
            "reward_support_foot_stability",
            "penalty_support_knee_drop",
            "reward_support_foot_parallel",
            "reward_support_foot_parallel_rp",
            "penalty_support_foot_toe_scrape",
            "reward_support_foot_yaw",
            "penalty_support_foot_stumble",
            "penalty_right_ankle_pitch_staged",
            "penalty_right_toe_only_contact",
            "penalty_right_toe_dominant_force",
            "reward_right_foot_parallel",
            "reward_post_kick_upright_early",
        ]:
            _set_weight_if_exists(env, name, 0.0)


def kick_skill_curriculum(
    env,
    env_ids: Sequence[int] | torch.Tensor,
    touch_metric_name: str = "reward_kick_foot_contact_ball",
    power_metric_name: str = "reward_ball_speed",
    direction_metric_name: str = "reward_ball_goal_direction",
    goal_speed_metric_name: str = "reward_ball_goal_speed",
    force_stage_b_step: int = 250_000,
    force_stage_c_step: int = 700_000,
    touch_threshold_b: float = 2.4,
    power_threshold_b: float = 1.8,
    touch_threshold_c: float = 2.8,
    direction_threshold_c: float = 1.4,
    goal_speed_threshold_c: float = 1.3,
    min_stage_b_steps: int = 100_000,
    min_stage_c_steps: int = 150_000,
    recovery_upright_threshold_c: float = 0.12,
    recovery_retract_threshold_c: float = 0.12,
    recovery_agg_threshold_c: float = 0.22,
    demote_b_to_a_touch: float = 1.0,
    demote_b_to_a_power: float = 0.6,
    demote_c_to_b_touch: float = 2.0,
    demote_c_to_b_power: float = 1.2,
    demote_c_to_b_direction: float = 0.8,
    demote_c_to_b_goal_speed: float = 0.7,
) -> float:
    """3-stage curriculum: A (kick learning) -> B (support foot + recovery) -> C (polish).

    Supports KICK_SKILL_FORCE_STAGE and KICK_SKILL_START_STAGE env vars.
    """
    # Pin stage via environment variable
    force_stage = os.environ.get("KICK_SKILL_FORCE_STAGE", "").strip().upper()
    if force_stage in (_STAGE_A, _STAGE_B, _STAGE_C):
        if not hasattr(env, "_kick_curr_stage") or env._kick_curr_stage != force_stage:
            env._kick_curr_stage = force_stage
            _apply_stage_weights(env, force_stage)
            env._kick_curr_stage_enter_step = int(env.common_step_counter)
        env._kick_promotion_fired = 0.0
        env._kick_demotion_fired = 0.0
        return float({"A": 0.0, "B": 1.0, "C": 2.0}[force_stage])

    if isinstance(env_ids, torch.Tensor):
        env_ids_t = env_ids.to(device=env.device)
    else:
        env_ids_t = torch.tensor(list(env_ids), device=env.device, dtype=torch.long)

    if not hasattr(env, "_kick_curr_stage"):
        start_stage = _STAGE_A
        if os.environ.get("KICK_SKILL_START_STAGE", "").strip().upper() == _STAGE_B:
            start_stage = _STAGE_B
            env._kick_b_start_mode = True
        else:
            env._kick_b_start_mode = False
        env._kick_curr_stage = start_stage
        env._kick_curr_stage_enter_step = int(env.common_step_counter)
        _apply_stage_weights(env, env._kick_curr_stage)
    if not hasattr(env, "_kick_promotion_fired"):
        env._kick_promotion_fired = 0.0
    if not hasattr(env, "_kick_demotion_fired"):
        env._kick_demotion_fired = 0.0
    if not hasattr(env, "_kick_recovery_quality"):
        env._kick_recovery_quality = 0.0

    if env.common_step_counter % env.max_episode_length != 0:
        return float({"A": 0.0, "B": 1.0, "C": 2.0}[env._kick_curr_stage])

    step = int(env.common_step_counter)
    steps_in_stage = step - getattr(env, "_kick_curr_stage_enter_step", step)
    stage = env._kick_curr_stage

    touch_avg = _episode_avg(env, env_ids_t, touch_metric_name)
    power_avg = _episode_avg(env, env_ids_t, power_metric_name)
    dir_avg = _episode_avg(env, env_ids_t, direction_metric_name)
    goal_speed_avg = _episode_avg(env, env_ids_t, goal_speed_metric_name)

    # Recovery metrics for B->C
    upright_avg = _episode_avg(env, env_ids_t, "reward_post_kick_upright")
    retract_avg = _episode_avg(env, env_ids_t, "reward_kick_leg_retract")
    recontact_avg = _episode_avg(env, env_ids_t, "reward_post_kick_recontact")
    vel_damp_avg = _episode_avg(env, env_ids_t, "reward_post_kick_velocity_damping")
    leg_spread_avg = _episode_avg(env, env_ids_t, "penalty_leg_spread")
    recovery_agg = upright_avg + retract_avg + recontact_avg + vel_damp_avg + leg_spread_avg
    env._kick_recovery_quality = float(recovery_agg) if isinstance(recovery_agg, torch.Tensor) else recovery_agg

    promotion_fired = False
    demotion_fired = False

    # A->B
    if stage == _STAGE_A:
        if (touch_avg >= touch_threshold_b and power_avg >= power_threshold_b) or (step >= force_stage_b_step):
            stage = _STAGE_B
            promotion_fired = True
    elif stage == _STAGE_B:
        # B->C
        kick_quality_ok = (
            touch_avg >= touch_threshold_c
            and power_avg >= power_threshold_b
        )
        stand_height_avg = _episode_avg(env, env_ids_t, "reward_post_kick_stand_height")
        stable_stand_avg = _episode_avg(env, env_ids_t, "reward_post_kick_stable_stand")
        crouch_penalty_avg = _episode_avg(env, env_ids_t, "penalty_post_kick_crouch")

        stand_ok = (
            upright_avg >= recovery_upright_threshold_c
            and retract_avg >= recovery_retract_threshold_c
            and stand_height_avg >= recovery_upright_threshold_c
            and stable_stand_avg >= recovery_retract_threshold_c
            and crouch_penalty_avg > -0.05
        ) or (recovery_agg >= recovery_agg_threshold_c)

        direction_ok = (
            dir_avg >= (0.8 * direction_threshold_c)
            and goal_speed_avg >= (0.8 * goal_speed_threshold_c)
        )

        if (kick_quality_ok and stand_ok and direction_ok) or (step >= force_stage_c_step):
            stage = _STAGE_C
            promotion_fired = True
        else:
            # B->A demotion (conservative)
            b_start_mode = getattr(env, "_kick_b_start_mode", False)
            if (not b_start_mode) and steps_in_stage >= min_stage_b_steps and (
                touch_avg < demote_b_to_a_touch or power_avg < demote_b_to_a_power
            ):
                stage = _STAGE_A
                demotion_fired = True
    else:
        # C->B demotion
        if steps_in_stage >= min_stage_c_steps and (
            touch_avg < demote_c_to_b_touch
            or power_avg < demote_c_to_b_power
            or dir_avg < demote_c_to_b_direction
            or goal_speed_avg < demote_c_to_b_goal_speed
        ):
            stage = _STAGE_B
            demotion_fired = True

    if stage != env._kick_curr_stage:
        env._kick_curr_stage = stage
        env._kick_curr_stage_enter_step = step
        _apply_stage_weights(env, stage)
        env._kick_promotion_fired = 1.0 if promotion_fired else 0.0
        env._kick_demotion_fired = 1.0 if demotion_fired else 0.0
    else:
        env._kick_promotion_fired = 0.0
        env._kick_demotion_fired = 0.0

    return float({"A": 0.0, "B": 1.0, "C": 2.0}[env._kick_curr_stage])


def kick_skill_bootstrap_curriculum(
    env,
    env_ids: Sequence[int] | torch.Tensor,
    min_episode_length_ratio_a1: float = 0.6,
    touch_threshold_a1: float = 0.8,
    power_threshold_a1: float = 0.4,
) -> float:
    """Two-stage bootstrap curriculum: A0 (stabilize) -> A1 (basic kick)."""
    if isinstance(env_ids, torch.Tensor):
        env_ids_t = env_ids.to(device=env.device)
    else:
        env_ids_t = torch.tensor(list(env_ids), device=env.device, dtype=torch.long)

    if not hasattr(env, "_kick_boot_stage"):
        env._kick_boot_stage = _BOOTSTRAP_STAGE_A0
        _apply_bootstrap_stage_weights(env, env._kick_boot_stage)

    stage = env._kick_boot_stage

    if env.common_step_counter % env.max_episode_length != 0:
        return float({_BOOTSTRAP_STAGE_A0: 0.0, _BOOTSTRAP_STAGE_A1: 1.0}[stage])

    # A0 -> A1
    if stage == _BOOTSTRAP_STAGE_A0:
        max_len = float(env.max_episode_length)
        ep_len = env.episode_length_buf[env_ids_t].float()
        ep_ratio = torch.clamp(ep_len / max_len, 0.0, 1.0)
        len_ratio_avg = float(ep_ratio.mean())

        touch_avg = _episode_avg(env, env_ids_t, "reward_kick_foot_contact_ball")
        power_avg = _episode_avg(env, env_ids_t, "reward_ball_speed")

        if (len_ratio_avg >= float(min_episode_length_ratio_a1)) and (
            touch_avg >= touch_threshold_a1 and power_avg >= power_threshold_a1
        ):
            stage = _BOOTSTRAP_STAGE_A1
            _apply_bootstrap_stage_weights(env, stage)

    env._kick_boot_stage = stage
    return float({_BOOTSTRAP_STAGE_A0: 0.0, _BOOTSTRAP_STAGE_A1: 1.0}[stage])
