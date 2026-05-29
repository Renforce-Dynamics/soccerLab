"""G1 kicking AMP environment configurations (mjlab backend).

Factory functions returning ManagerBasedRlEnvCfg for:
  - g1_kick_basic_env_cfg()       -> Stage A (kick learning)
  - g1_kick_bootstrap_env_cfg()   -> Bootstrap (A0->A1)
  - g1_kick_stage_b_env_cfg()     -> Stage B (support foot + recovery)
  - g1_kick_stage_c_env_cfg()     -> Stage C (polish + stability)

All variants share the same obs/action structure. Only curriculum,
event configs, and episode length differ. Reward weights are set by
the curriculum callbacks at runtime.
"""

from __future__ import annotations

import os
import pathlib

import mujoco

from mjlab.asset_zoo.robots import get_g1_robot_cfg, G1_ACTION_SCALE
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from beyondAMP.mjlab.obs_groups import amp_obs_basic_group

from soccer_tasks_mjlab.kicking.g1.mdp import (
    # observations
    ball_pos_rel,
    ball_vel_rel,
    goal_dir_rel,
    ball_to_goal_dir_rel,
    kick_phase,
    motion_anchor_pos_b,
    # rewards
    reward_approach_ball,
    reward_kick_leg_swing,
    reward_kick_foot_contact_ball,
    reward_first_clean_contact_bonus,
    reward_ball_speed,
    reward_ball_impulse,
    reward_ball_goal_direction,
    reward_ball_goal_speed,
    tracking_anchor_pos_gated,
    tracking_anchor_ori_gated,
    tracking_body_pos_gated,
    tracking_body_ori_gated,
    tracking_body_vel_gated,
    penalty_excess_travel,
    penalty_bad_ball_contact,
    penalty_right_ankle_pitch_staged,
    penalty_right_toe_only_contact,
    penalty_right_toe_dominant_force,
    reward_right_foot_parallel,
    reward_post_kick_upright,
    reward_post_kick_base_height,
    reward_kick_leg_retract,
    reward_post_kick_recontact,
    reward_post_kick_velocity_damping,
    reward_post_kick_joint_nominal,
    penalty_leg_spread,
    penalty_post_joint_limit_stronger,
    penalty_post_kick_crouch,
    reward_post_kick_stand_height,
    reward_post_kick_stable_stand,
    penalty_support_toe_stance,
    reward_support_foot_flat_contact,
    reward_support_foot_stability,
    penalty_support_knee_drop,
    reward_support_foot_parallel,
    reward_support_foot_parallel_rp,
    penalty_support_foot_toe_scrape,
    reward_support_foot_yaw,
    penalty_support_foot_stumble,
    reward_support_foot_flat_contact_early,
    reward_post_kick_upright_early,
    metric_gate_dist,
    metric_gate_phase,
    metric_gate_leg,
    metric_gate_contact,
    metric_curriculum_stage,
    metric_curriculum_steps_in_stage,
    metric_curriculum_promotion_fired,
    metric_curriculum_demotion_fired,
    metric_curriculum_recovery_quality,
    # terminations
    kick_root_height_below_minimum_with_window,
    kick_bad_orientation_with_window,
    bad_ball_stuck,
    # events
    reset_ball_state,
    reset_root_and_ball_right_front,
    # curriculums
    kick_skill_curriculum,
    kick_skill_bootstrap_curriculum,
    # actions / base mdp
    action_rate_l2,
    joint_pos_limits,
    time_out,
    reset_root_state_uniform,
    reset_joints_by_offset,
    JointPositionActionCfg,
    # commands
    BallCenterVelocityCommandCfg,
    KickMotionCommandCfg,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SOCCERLAB_ROOT = pathlib.Path(__file__).resolve().parents[5]  # soccerLab/
_BALL_MJCF = _SOCCERLAB_ROOT / "data" / "assets" / "ball" / "soccer_ball.xml"


def _resolve_kick_motion_default() -> str:
    default_rel = "data/datasets/g1_kick_skill/wo_cf_shoot_74_06.npz"
    for prefix in ("", str(_SOCCERLAB_ROOT)):
        path = os.path.join(prefix, default_rel) if prefix else default_rel
        if os.path.isfile(path):
            return path
    return default_rel


_KICK_MOTION_FILE = os.getenv("KICK_SKILL_MOTION_FILE", _resolve_kick_motion_default())


# ---------------------------------------------------------------------------
# Ball entity
# ---------------------------------------------------------------------------

def _get_ball_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_file(str(_BALL_MJCF)),
        init_state=EntityCfg.InitialStateCfg(
            pos=(1.0, 0.0, 0.115),
        ),
    )


# ---------------------------------------------------------------------------
# Tracking body names
# ---------------------------------------------------------------------------

_TRACKING_BODY_NAMES = [
    "pelvis",
    "torso_link",
    "right_hip_roll_link",
    "right_hip_pitch_link",
    "right_knee_link",
    "right_ankle_roll_link",
]

_KICK_BODY_NAMES = [
    "pelvis",
    "torso_link",
    "right_hip_roll_link",
    "right_hip_pitch_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "left_hip_roll_link",
    "left_hip_pitch_link",
    "left_knee_link",
    "left_ankle_roll_link",
]

_CRITICAL_BODY_NAMES = [
    "pelvis",
    "torso_link",
    "right_hip_roll_link",
    "right_hip_pitch_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
]


# ---------------------------------------------------------------------------
# Shared observation terms
# ---------------------------------------------------------------------------

def _make_obs_terms(enable_noise: bool = True) -> dict[str, ObservationTermCfg]:
    terms = {
        "base_lin_vel": ObservationTermCfg(
            func=envs_mdp.base_lin_vel,
            scale=0.8,
        ),
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.base_ang_vel,
            scale=0.2,
            noise=Unoise(n_min=-0.15, n_max=0.15) if enable_noise else None,
        ),
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity,
            noise=Unoise(n_min=-0.03, n_max=0.03) if enable_noise else None,
        ),
        "joint_pos_rel": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01) if enable_noise else None,
        ),
        "joint_vel_rel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            scale=0.05,
            noise=Unoise(n_min=-0.8, n_max=0.8) if enable_noise else None,
        ),
        "last_action": ObservationTermCfg(func=envs_mdp.last_action),
        "ball_pos_rel": ObservationTermCfg(func=ball_pos_rel),
        "ball_vel_rel": ObservationTermCfg(func=ball_vel_rel),
        "goal_dir_rel": ObservationTermCfg(
            func=goal_dir_rel,
            params={"command_name": "ball_target_velocity"},
        ),
        "ball_to_goal_dir_rel": ObservationTermCfg(
            func=ball_to_goal_dir_rel,
            params={"command_name": "ball_target_velocity"},
        ),
        "kick_phase": ObservationTermCfg(
            func=kick_phase,
            params={"command_name": "kick_motion"},
        ),
        "motion_anchor_pos_b": ObservationTermCfg(
            func=motion_anchor_pos_b,
            params={"command_name": "kick_motion"},
        ),
    }
    return terms


def _make_observations() -> dict[str, ObservationGroupCfg]:
    return {
        "actor": ObservationGroupCfg(
            terms=_make_obs_terms(enable_noise=True),
            concatenate_terms=True,
            enable_corruption=True,
        ),
        "critic": ObservationGroupCfg(
            terms=_make_obs_terms(enable_noise=False),
            concatenate_terms=True,
            enable_corruption=False,
            history_length=4,
        ),
    }


# ---------------------------------------------------------------------------
# Shared rewards dict
# ---------------------------------------------------------------------------

def _make_full_rewards() -> dict[str, RewardTermCfg]:
    """Full reward set used by basic/stageB/stageC variants.

    Initial weights are for Stage A; curriculum callbacks override them.
    """
    return {
        "reward_approach_ball": RewardTermCfg(
            func=reward_approach_ball, weight=0.6,
            params={"target_distance": 0.20, "kick_zone_distance": 0.16},
        ),
        "reward_kick_leg_swing": RewardTermCfg(
            func=reward_kick_leg_swing, weight=12.0,
            params={"target_mode": "ball"},
        ),
        "reward_kick_foot_contact_ball": RewardTermCfg(
            func=reward_kick_foot_contact_ball, weight=11.0,
            params={"min_approach_speed": 0.10, "contact_distance": 0.14},
        ),
        "reward_first_clean_contact_bonus": RewardTermCfg(
            func=reward_first_clean_contact_bonus, weight=8.0,
            params={"min_approach_speed": 0.18, "contact_distance": 0.14},
        ),
        "reward_ball_speed": RewardTermCfg(
            func=reward_ball_speed, weight=14.0,
            params={"min_speed": 0.08},
        ),
        "reward_ball_impulse": RewardTermCfg(
            func=reward_ball_impulse, weight=10.0,
            params={"window_steps": 10},
        ),
        "reward_ball_goal_direction": RewardTermCfg(
            func=reward_ball_goal_direction, weight=0.0,
            params={"angle_threshold_deg": 25.0},
        ),
        "reward_ball_goal_speed": RewardTermCfg(
            func=reward_ball_goal_speed, weight=0.0,
        ),
        # Tracking
        "tracking_anchor_pos": RewardTermCfg(func=tracking_anchor_pos_gated, weight=0.20),
        "tracking_anchor_ori": RewardTermCfg(func=tracking_anchor_ori_gated, weight=0.18),
        "tracking_body_pos": RewardTermCfg(
            func=tracking_body_pos_gated, weight=0.26,
            params={"body_names": _TRACKING_BODY_NAMES},
        ),
        "tracking_body_ori": RewardTermCfg(
            func=tracking_body_ori_gated, weight=0.22,
            params={"body_names": _TRACKING_BODY_NAMES},
        ),
        "tracking_body_vel": RewardTermCfg(
            func=tracking_body_vel_gated, weight=0.18,
            params={"body_names": _TRACKING_BODY_NAMES},
        ),
        # Regularization
        "penalty_excess_travel": RewardTermCfg(func=penalty_excess_travel, weight=-0.02),
        "penalty_bad_ball_contact": RewardTermCfg(func=penalty_bad_ball_contact, weight=-1.0),
        "action_rate": RewardTermCfg(func=action_rate_l2, weight=-1e-4),
        "joint_limit": RewardTermCfg(func=joint_pos_limits, weight=-0.04),
        # Anti-toe (curriculum-controlled)
        "penalty_right_ankle_pitch_staged": RewardTermCfg(func=penalty_right_ankle_pitch_staged, weight=0.0),
        "penalty_right_toe_only_contact": RewardTermCfg(func=penalty_right_toe_only_contact, weight=0.0),
        "penalty_right_toe_dominant_force": RewardTermCfg(func=penalty_right_toe_dominant_force, weight=0.0),
        "reward_right_foot_parallel": RewardTermCfg(func=reward_right_foot_parallel, weight=0.0),
        # NEW tiptoe fix: active from Bootstrap/Stage A
        "reward_support_foot_flat_contact_early": RewardTermCfg(func=reward_support_foot_flat_contact_early, weight=0.0),
        # Post-kick recovery (curriculum-controlled)
        "reward_post_kick_upright": RewardTermCfg(func=reward_post_kick_upright, weight=0.0),
        "reward_post_kick_base_height": RewardTermCfg(func=reward_post_kick_base_height, weight=0.0),
        "reward_kick_leg_retract": RewardTermCfg(func=reward_kick_leg_retract, weight=0.0),
        "reward_post_kick_recontact": RewardTermCfg(func=reward_post_kick_recontact, weight=0.0),
        "reward_post_kick_velocity_damping": RewardTermCfg(func=reward_post_kick_velocity_damping, weight=0.0),
        "reward_post_kick_joint_nominal": RewardTermCfg(func=reward_post_kick_joint_nominal, weight=0.0),
        "penalty_leg_spread": RewardTermCfg(func=penalty_leg_spread, weight=0.0),
        "penalty_post_joint_limit_stronger": RewardTermCfg(func=penalty_post_joint_limit_stronger, weight=0.0),
        "penalty_post_kick_crouch": RewardTermCfg(func=penalty_post_kick_crouch, weight=0.0),
        "reward_post_kick_stand_height": RewardTermCfg(func=reward_post_kick_stand_height, weight=0.0),
        "reward_post_kick_stable_stand": RewardTermCfg(func=reward_post_kick_stable_stand, weight=0.0),
        # NEW post-kick fix: light upright from Stage A
        "reward_post_kick_upright_early": RewardTermCfg(func=reward_post_kick_upright_early, weight=0.0),
        # Support foot (curriculum-controlled)
        "penalty_support_toe_stance": RewardTermCfg(func=penalty_support_toe_stance, weight=0.0),
        "reward_support_foot_flat_contact": RewardTermCfg(func=reward_support_foot_flat_contact, weight=0.0),
        "reward_support_foot_stability": RewardTermCfg(func=reward_support_foot_stability, weight=0.0),
        "penalty_support_knee_drop": RewardTermCfg(func=penalty_support_knee_drop, weight=0.0),
        "reward_support_foot_parallel": RewardTermCfg(func=reward_support_foot_parallel, weight=0.0),
        "reward_support_foot_parallel_rp": RewardTermCfg(func=reward_support_foot_parallel_rp, weight=0.0),
        "penalty_support_foot_toe_scrape": RewardTermCfg(func=penalty_support_foot_toe_scrape, weight=0.0),
        "reward_support_foot_yaw": RewardTermCfg(func=reward_support_foot_yaw, weight=0.0),
        "penalty_support_foot_stumble": RewardTermCfg(func=penalty_support_foot_stumble, weight=0.0),
        # Metrics (weight=0)
        "metric_gate_dist": RewardTermCfg(func=metric_gate_dist, weight=0.0),
        "metric_gate_phase": RewardTermCfg(func=metric_gate_phase, weight=0.0),
        "metric_gate_leg": RewardTermCfg(func=metric_gate_leg, weight=0.0),
        "metric_gate_contact": RewardTermCfg(func=metric_gate_contact, weight=0.0),
        "metric_curriculum_stage": RewardTermCfg(func=metric_curriculum_stage, weight=0.0),
        "metric_curriculum_steps_in_stage": RewardTermCfg(func=metric_curriculum_steps_in_stage, weight=0.0),
        "metric_curriculum_promotion_fired": RewardTermCfg(func=metric_curriculum_promotion_fired, weight=0.0),
        "metric_curriculum_demotion_fired": RewardTermCfg(func=metric_curriculum_demotion_fired, weight=0.0),
        "metric_curriculum_recovery_quality": RewardTermCfg(func=metric_curriculum_recovery_quality, weight=0.0),
    }


def _make_bootstrap_rewards() -> dict[str, RewardTermCfg]:
    """Reduced reward set for bootstrap variant."""
    return {
        "reward_approach_ball": RewardTermCfg(
            func=reward_approach_ball, weight=0.4,
            params={"target_distance": 0.20, "kick_zone_distance": 0.16},
        ),
        "reward_kick_leg_swing": RewardTermCfg(
            func=reward_kick_leg_swing, weight=4.0,
            params={"target_mode": "ball"},
        ),
        "reward_kick_foot_contact_ball": RewardTermCfg(
            func=reward_kick_foot_contact_ball, weight=6.0,
            params={"min_approach_speed": 0.10, "contact_distance": 0.14},
        ),
        "reward_first_clean_contact_bonus": RewardTermCfg(
            func=reward_first_clean_contact_bonus, weight=3.0,
            params={"min_approach_speed": 0.18, "contact_distance": 0.14},
        ),
        "reward_ball_speed": RewardTermCfg(
            func=reward_ball_speed, weight=8.0,
            params={"min_speed": 0.08},
        ),
        "reward_ball_impulse": RewardTermCfg(
            func=reward_ball_impulse, weight=6.0,
            params={"window_steps": 10},
        ),
        "reward_ball_goal_direction": RewardTermCfg(
            func=reward_ball_goal_direction, weight=0.0,
            params={"angle_threshold_deg": 25.0},
        ),
        "reward_ball_goal_speed": RewardTermCfg(func=reward_ball_goal_speed, weight=0.0),
        # Tracking (weak)
        "tracking_anchor_pos": RewardTermCfg(func=tracking_anchor_pos_gated, weight=0.12),
        "tracking_anchor_ori": RewardTermCfg(func=tracking_anchor_ori_gated, weight=0.10),
        "tracking_body_pos": RewardTermCfg(
            func=tracking_body_pos_gated, weight=0.18,
            params={"body_names": _TRACKING_BODY_NAMES},
        ),
        "tracking_body_ori": RewardTermCfg(
            func=tracking_body_ori_gated, weight=0.16,
            params={"body_names": _TRACKING_BODY_NAMES},
        ),
        "tracking_body_vel": RewardTermCfg(
            func=tracking_body_vel_gated, weight=0.12,
            params={"body_names": _TRACKING_BODY_NAMES},
        ),
        # Regularization
        "penalty_excess_travel": RewardTermCfg(func=penalty_excess_travel, weight=-0.02),
        "penalty_bad_ball_contact": RewardTermCfg(func=penalty_bad_ball_contact, weight=-1.0),
        "action_rate": RewardTermCfg(func=action_rate_l2, weight=-5.0e-5),
        "joint_limit": RewardTermCfg(func=joint_pos_limits, weight=-0.03),
        # NEW tiptoe fix: active from Bootstrap
        "reward_support_foot_flat_contact_early": RewardTermCfg(func=reward_support_foot_flat_contact_early, weight=0.0),
        # Post-kick (all off in bootstrap, controlled by bootstrap curriculum)
        "reward_post_kick_upright": RewardTermCfg(func=reward_post_kick_upright, weight=0.0),
        "reward_post_kick_base_height": RewardTermCfg(func=reward_post_kick_base_height, weight=0.0),
        "reward_kick_leg_retract": RewardTermCfg(func=reward_kick_leg_retract, weight=0.0),
        "reward_post_kick_recontact": RewardTermCfg(func=reward_post_kick_recontact, weight=0.0),
        "reward_post_kick_velocity_damping": RewardTermCfg(func=reward_post_kick_velocity_damping, weight=0.0),
        "reward_post_kick_joint_nominal": RewardTermCfg(func=reward_post_kick_joint_nominal, weight=0.0),
        "penalty_leg_spread": RewardTermCfg(func=penalty_leg_spread, weight=0.0),
        "penalty_post_joint_limit_stronger": RewardTermCfg(func=penalty_post_joint_limit_stronger, weight=0.0),
        "penalty_post_kick_crouch": RewardTermCfg(func=penalty_post_kick_crouch, weight=0.0),
        "reward_post_kick_stand_height": RewardTermCfg(func=reward_post_kick_stand_height, weight=0.0),
        "reward_post_kick_stable_stand": RewardTermCfg(func=reward_post_kick_stable_stand, weight=0.0),
        "reward_post_kick_upright_early": RewardTermCfg(func=reward_post_kick_upright_early, weight=0.0),
        # Metrics
        "metric_gate_dist": RewardTermCfg(func=metric_gate_dist, weight=0.0),
        "metric_gate_phase": RewardTermCfg(func=metric_gate_phase, weight=0.0),
        "metric_gate_leg": RewardTermCfg(func=metric_gate_leg, weight=0.0),
        "metric_gate_contact": RewardTermCfg(func=metric_gate_contact, weight=0.0),
        "metric_curriculum_stage": RewardTermCfg(func=metric_curriculum_stage, weight=0.0),
        "metric_curriculum_steps_in_stage": RewardTermCfg(func=metric_curriculum_steps_in_stage, weight=0.0),
        "metric_curriculum_promotion_fired": RewardTermCfg(func=metric_curriculum_promotion_fired, weight=0.0),
        "metric_curriculum_demotion_fired": RewardTermCfg(func=metric_curriculum_demotion_fired, weight=0.0),
        "metric_curriculum_recovery_quality": RewardTermCfg(func=metric_curriculum_recovery_quality, weight=0.0),
    }


# ---------------------------------------------------------------------------
# Shared command configs
# ---------------------------------------------------------------------------

def _make_kick_motion_cmd_cfg(*, bootstrap: bool = False) -> KickMotionCommandCfg:
    base_kwargs = dict(
        resampling_time_range=(1.0e9, 1.0e9),
        asset_name="robot",
        motion_file=_KICK_MOTION_FILE,
        anchor_body_name="torso_link",
        body_names=_KICK_BODY_NAMES,
        critical_body_names=_CRITICAL_BODY_NAMES,
        debug_vis=False,
        strike_phase_window=(0.35, 0.62),
        pre_kick_phase_end=0.34,
        recover_phase_start=0.72,
        kicking_foot_body_name="right_ankle_roll_link",
        supporting_foot_body_name="left_ankle_roll_link",
    )
    if bootstrap:
        base_kwargs.update(
            pose_range={"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.003, 0.003),
                         "roll": (-0.02, 0.02), "pitch": (-0.02, 0.02), "yaw": (-0.06, 0.06)},
            velocity_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.03, 0.03),
                            "roll": (-0.06, 0.06), "pitch": (-0.06, 0.06), "yaw": (-0.08, 0.08)},
            joint_position_range=(-0.06, 0.06),
            start_phase_range=(0.0, 0.30),
        )
    else:
        base_kwargs.update(
            pose_range={"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.005, 0.005),
                         "roll": (-0.03, 0.03), "pitch": (-0.03, 0.03), "yaw": (-0.08, 0.08)},
            velocity_range={"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
                            "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.12, 0.12)},
            joint_position_range=(-0.08, 0.08),
        )
    return KickMotionCommandCfg(**base_kwargs)


def _make_ball_vel_cmd_cfg() -> BallCenterVelocityCommandCfg:
    return BallCenterVelocityCommandCfg(
        resampling_time_range=(10.0, 10.0),
        asset_name="ball",
        goal_heading_range_deg=(-8.0, 8.0),
        goal_speed_range=(0.45, 0.95),
        goal_direction_xy=(1.0, 0.0),
    )


# ---------------------------------------------------------------------------
# Shared sim / scene
# ---------------------------------------------------------------------------

def _make_sim_cfg() -> SimulationCfg:
    return SimulationCfg(
        nconmax=70,
        njmax=500,
        mujoco=MujocoCfg(
            timestep=0.005,
            iterations=10,
            ls_iterations=20,
            ccd_iterations=500,
        ),
        contact_sensor_maxmatch=500,
    )


def _make_scene(num_envs: int = 4096) -> SceneCfg:
    return SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={
            "robot": get_g1_robot_cfg(),
            "ball": _get_ball_cfg(),
        },
        num_envs=num_envs,
        extent=3.0,
    )


# ====================================================================
# Factory: Basic (Stage A kick learning, auto-promotes to B/C)
# ====================================================================

def g1_kick_basic_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    obs = _make_observations()
    obs["amp"] = amp_obs_basic_group()

    events = {
        "reset_base": EventTermCfg(
            func=reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.22, 0.22)},
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.12, 0.12),
                "velocity_range": (-0.10, 0.10),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "reset_ball": EventTermCfg(
            func=reset_ball_state,
            mode="reset",
            params={
                "position_range": {"x": (0.12, 0.15), "y": (0.02, 0.04), "z": (0.00, 0.01)},
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            },
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
        "base_height": TerminationTermCfg(
            func=kick_root_height_below_minimum_with_window,
            params={"minimum_height": 0.22, "grace_window_s": 3.0},
        ),
        "bad_orientation": TerminationTermCfg(
            func=kick_bad_orientation_with_window,
            params={"limit_angle": 1.0, "grace_window_s": 3.0},
        ),
        "bad_ball_stuck": TerminationTermCfg(
            func=bad_ball_stuck,
            params={"min_eval_steps": 12, "max_ball_speed": 0.12, "max_ball_distance": 0.20},
        ),
    }

    curriculum = {
        "kick_skill": CurriculumTermCfg(func=kick_skill_curriculum),
    }

    num_envs = 64 if play else 4096

    if play:
        obs["actor"].enable_corruption = False

    cfg = ManagerBasedRlEnvCfg(
        scene=_make_scene(num_envs=num_envs),
        observations=obs,
        actions={"joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=G1_ACTION_SCALE,
            use_default_offset=True,
        )},
        commands={
            "ball_target_velocity": _make_ball_vel_cmd_cfg(),
            "kick_motion": _make_kick_motion_cmd_cfg(bootstrap=False),
        },
        events=events,
        rewards=_make_full_rewards(),
        terminations=terminations,
        curriculum=curriculum,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",
            distance=4.0,
            elevation=-10.0,
            azimuth=90.0,
        ),
        sim=_make_sim_cfg(),
        decimation=4,
        episode_length_s=5.0,
    )
    return cfg


# ====================================================================
# Factory: Bootstrap (A0 -> A1, from-scratch pre-training)
# ====================================================================

def g1_kick_bootstrap_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    obs = _make_observations()
    obs["amp"] = amp_obs_basic_group()

    events = {
        "reset_base": EventTermCfg(
            func=reset_root_and_ball_right_front,
            mode="reset",
            params={
                "root_pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "yaw": (-0.12, 0.12)},
                "root_velocity_range": {},
                "ball_relative_position_range": {"x": (0.52, 0.55), "y": (0.25, 0.30), "z": (0.00, 0.01)},
                "ball_velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.05, 0.05),
                "velocity_range": (-0.05, 0.05),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
        "base_height": TerminationTermCfg(
            func=kick_root_height_below_minimum_with_window,
            params={"minimum_height": 0.20, "grace_window_s": 3.5},
        ),
        "bad_orientation": TerminationTermCfg(
            func=kick_bad_orientation_with_window,
            params={"limit_angle": 1.2, "grace_window_s": 3.5},
        ),
        "bad_ball_stuck": TerminationTermCfg(
            func=bad_ball_stuck,
            params={"min_eval_steps": 12, "max_ball_speed": 0.12, "max_ball_distance": 0.20},
        ),
    }

    curriculum = {
        "kick_skill_bootstrap": CurriculumTermCfg(func=kick_skill_bootstrap_curriculum),
    }

    num_envs = 64 if play else 4096

    if play:
        obs["actor"].enable_corruption = False

    cfg = ManagerBasedRlEnvCfg(
        scene=_make_scene(num_envs=num_envs),
        observations=obs,
        actions={"joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=G1_ACTION_SCALE,
            use_default_offset=True,
        )},
        commands={
            "ball_target_velocity": _make_ball_vel_cmd_cfg(),
            "kick_motion": _make_kick_motion_cmd_cfg(bootstrap=True),
        },
        events=events,
        rewards=_make_bootstrap_rewards(),
        terminations=terminations,
        curriculum=curriculum,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",
            distance=4.0,
            elevation=-10.0,
            azimuth=90.0,
        ),
        sim=_make_sim_cfg(),
        decimation=4,
        episode_length_s=4.0,
    )
    return cfg


# ====================================================================
# Factory: Stage B (support foot + recovery, starts at B via env var)
# ====================================================================

def g1_kick_stage_b_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Same as basic but starts at Stage B (KICK_SKILL_START_STAGE=B)."""
    cfg = g1_kick_basic_env_cfg(play=play)
    # The curriculum callback reads KICK_SKILL_START_STAGE to pick B
    # We set it here so task registration carries the intent.
    os.environ.setdefault("KICK_SKILL_START_STAGE", "B")
    return cfg


# ====================================================================
# Factory: Stage C (polish + stability, force-locked to C)
# ====================================================================

def g1_kick_stage_c_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Same as basic but force-locked to Stage C."""
    cfg = g1_kick_basic_env_cfg(play=play)
    os.environ.setdefault("KICK_SKILL_FORCE_STAGE", "C")
    return cfg
