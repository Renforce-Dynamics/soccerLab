"""Dribbling environment configuration for T1 on mjlab.

Ported from BackupDribbling's IsaacLab config. The task is pure PPO
(no AMP) -- the robot learns to walk towards and dribble a soccer ball.

Key T1 differences from G1:
- 23 actuated DOFs (14 used for dribbling: 12 lower body + 2 head)
- Lower base height (~0.60m vs G1's ~0.75m)
- Head DOFs present (AAHead_yaw, Head_pitch) -> head_pos reward kept
- T1 foot bodies: left_foot_link, right_foot_link
- T1 joint names: capitalized, no _joint suffix
"""

from __future__ import annotations

import math
import pathlib

import mujoco

from mjlab.actuator.builtin_actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from soccer_tasks_mjlab.dribbling.t1.mdp import observations as dribbling_obs
from soccer_tasks_mjlab.dribbling.t1.mdp import rewards as dribbling_rew
from mjlab.tasks.velocity import mdp as vel_mdp


# ==============================================================================
# T1 robot configuration
# ==============================================================================

_T1_MJCF = pathlib.Path(__file__).resolve().parents[4] / ".." / ".." / "data" / "assets" / "t1" / "t1_23dof.xml"
# Resolve to absolute for robustness
_SOCCERLAB_ROOT = pathlib.Path(__file__).resolve().parents[5]  # soccerLab/
_T1_MJCF = _SOCCERLAB_ROOT / "data" / "assets" / "t1" / "t1_23dof.xml"

# T1 body/site names for rewards
T1_FOOT_BODIES: tuple[str, ...] = (
    "left_foot_link",
    "right_foot_link",
)
T1_FOOT_SITES: tuple[str, ...] = ("left_foot", "right_foot")

# Action scale: original T1 dribbling used 0.4 for all joints
T1_ACTION_SCALE = 0.4

# Actuator pattern for the 14 dribbling joints (lower body + head)
T1_DRIBBLING_ACTUATOR_PATTERN = (
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
    "AAHead_yaw",
    "Head_pitch",
)


def _get_t1_robot_cfg() -> EntityCfg:
    """Create T1 robot configuration for mjlab."""
    return EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_file(str(_T1_MJCF)),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(
                        "Left_Hip_Pitch",
                        "Left_Hip_Roll",
                        "Left_Hip_Yaw",
                        "Right_Hip_Pitch",
                        "Right_Hip_Roll",
                        "Right_Hip_Yaw",
                    ),
                    stiffness=250.0,
                    damping=10.0,
                    effort_limit=200.0,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(
                        "Left_Knee_Pitch",
                        "Right_Knee_Pitch",
                    ),
                    stiffness=250.0,
                    damping=10.0,
                    effort_limit=200.0,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(
                        "Left_Ankle_Pitch",
                        "Left_Ankle_Roll",
                        "Right_Ankle_Pitch",
                        "Right_Ankle_Roll",
                    ),
                    stiffness=100.0,
                    damping=2.0,
                    effort_limit=200.0,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(
                        "AAHead_yaw",
                        "Head_pitch",
                    ),
                    stiffness=20.0,
                    damping=2.0,
                    effort_limit=20.0,
                ),
                BuiltinPositionActuatorCfg(
                    target_names_expr=(
                        "Left_Shoulder_Pitch",
                        "Left_Shoulder_Roll",
                        "Left_Elbow_Pitch",
                        "Left_Elbow_Yaw",
                        "Right_Shoulder_Pitch",
                        "Right_Shoulder_Roll",
                        "Right_Elbow_Pitch",
                        "Right_Elbow_Yaw",
                        "Waist",
                    ),
                    stiffness=50.0,
                    damping=5.0,
                    effort_limit=50.0,
                ),
            ),
            soft_joint_pos_limit_factor=0.9,
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.65),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
    )


# ==============================================================================
# Ball configuration
# ==============================================================================

_BALL_MJCF = _SOCCERLAB_ROOT / "data" / "assets" / "ball" / "soccer_ball.xml"


def _get_ball_cfg() -> EntityCfg:
    """Create soccer ball configuration for mjlab."""
    return EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_file(str(_BALL_MJCF)),
        init_state=EntityCfg.InitialStateCfg(
            pos=(1.0, 0.0, 0.115),
        ),
    )


# ==============================================================================
# Environment configuration factory
# ==============================================================================


def make_dribbling_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create T1 dribbling task configuration."""

    ##
    # Sensors
    ##

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_foot_link|right_foot_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )

    ##
    # Observations (single "policy" group, matching original IsaacLab config)
    ##

    policy_terms = {
        "base_lin_vel": ObservationTermCfg(
            func=envs_mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=envs_mdp.last_action),
        "ball_pos": ObservationTermCfg(
            func=dribbling_obs.ball_position_in_robot_root_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ball_cfg": SceneEntityCfg("ball"),
            },
        ),
        "ball_vel": ObservationTermCfg(
            func=dribbling_obs.ball_velocity_in_robot_root_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ball_cfg": SceneEntityCfg("ball"),
            },
        ),
        "gait_phase": ObservationTermCfg(func=dribbling_obs.gait_phase_obs),
    }

    observations = {
        "actor": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=True,
        ),
        "critic": ObservationGroupCfg(
            terms={**policy_terms},
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    ##
    # Actions -- only control the 14 dribbling joints (lower body + head)
    ##

    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=T1_DRIBBLING_ACTUATOR_PATTERN,
            scale=T1_ACTION_SCALE,
            use_default_offset=True,
        ),
    }

    ##
    # Events
    ##

    events = {
        "reset_base": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.1, 0.1),
                    "y": (-0.1, 0.1),
                    "yaw": (-0.1, 0.1),
                },
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=envs_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "reset_ball": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("ball"),
                "pose_range": {
                    "x": (1.0, 2.0),
                    "y": (-0.5, 0.5),
                },
                "velocity_range": {},
            },
        ),
    }

    ##
    # Rewards (T1-specific values from original BackupDribbling config)
    ##

    robot_cfg = SceneEntityCfg("robot")
    ball_cfg = SceneEntityCfg("ball")
    foot_body_cfg = SceneEntityCfg("robot", body_names=T1_FOOT_BODIES)
    foot_site_cfg = SceneEntityCfg("robot", site_names=T1_FOOT_SITES)

    rewards = {
        # -- Task rewards (positive) --
        "ball_distance": RewardTermCfg(
            func=dribbling_rew.ball_distance_exp,
            weight=2.0,
            params={"std": 2.0, "robot_cfg": robot_cfg, "ball_cfg": ball_cfg},
        ),
        "track_ball_vel": RewardTermCfg(
            func=dribbling_rew.track_ball_velocity,
            weight=1.5,
            params={
                "asset_cfg": robot_cfg,
                "ball_cfg": ball_cfg,
                "target_speed": 1.0,
            },
        ),
        "tracking_ball_target_vel": RewardTermCfg(
            func=dribbling_rew.tracking_ball_target_vel_reward_fixed,
            weight=1.5,
            params={"target_vel_x": 1.0, "ball_cfg": ball_cfg},
        ),
        "tracking_ball_view": RewardTermCfg(
            func=dribbling_rew.tracking_ball_view,
            weight=0.5,
            params={"robot_cfg": robot_cfg, "ball_cfg": ball_cfg},
        ),
        "tracking_ang_vel": RewardTermCfg(
            func=dribbling_rew.tracking_ang_vel_reward,
            weight=0.2,
            params={"robot_cfg": robot_cfg, "ball_cfg": ball_cfg},
        ),
        # -- Locomotion rewards (positive) --
        "base_height": RewardTermCfg(
            func=dribbling_rew.base_height_reward,
            weight=5.0,
            params={"target_height": 0.60, "asset_cfg": robot_cfg},
            # T1 base height ~0.60m (original value, not G1's 0.75m)
        ),
        "orientation": RewardTermCfg(
            func=dribbling_rew.orientation_reward,
            weight=2.0,
            params={"asset_cfg": robot_cfg},
        ),
        "joint_pos": RewardTermCfg(
            func=dribbling_rew.joint_pos_reward_stage1,
            weight=1.8,
            params={
                "asset_cfg": robot_cfg,
                "target_joint_pos_scale": 1.15,
                "ref_pos_dir": [1, -1, 1, -1, 1, -1],
                "cycle_time": 0.8,
                "double_stand_phase": 0.5,
            },
        ),
        "feet_orientation": RewardTermCfg(
            func=dribbling_rew.feet_orientation_reward,
            weight=1.0,
            params={"asset_cfg": foot_body_cfg},
        ),
        "feet_distance": RewardTermCfg(
            func=dribbling_rew.feet_distance_reward,
            weight=1.5,
            params={"min_dist": 0.20, "max_dist": 0.45, "asset_cfg": foot_body_cfg},
        ),
        "feet_clearance": RewardTermCfg(
            func=dribbling_rew.feet_clearance_reward,
            weight=0.65,
            params={"target_feet_height": 0.092, "asset_cfg": foot_site_cfg},
        ),
        "feet_stride": RewardTermCfg(
            func=dribbling_rew.feet_stride_reward,
            weight=3.6,
            params={
                "min_stride": 0.28,
                "max_stride": 0.80,
                "asset_cfg": foot_body_cfg,
            },
        ),
        "forward_velocity": RewardTermCfg(
            func=dribbling_rew.robot_forward_velocity_reward,
            weight=1.4,
            params={"target_vel": 0.7, "asset_cfg": robot_cfg},
        ),
        "feet_air_time": RewardTermCfg(
            func=dribbling_rew.feet_air_time_reward,
            weight=2.3,
            params={
                "sensor_name": feet_ground_cfg.name,
                "threshold": 18.0,
            },
        ),
        # -- T1-specific: head position reward (T1 has head DOFs) --
        "head_pos": RewardTermCfg(
            func=dribbling_rew.head_position_reward,
            weight=1.0,
            params={"asset_cfg": robot_cfg},
        ),
        # -- Penalties (negative weights) --
        "feet_contact_forces": RewardTermCfg(
            func=dribbling_rew.feet_contact_forces_cost,
            weight=-0.0003,
            params={
                "max_contact_force": 300.0,
                "sensor_name": feet_ground_cfg.name,
            },
        ),
        "self_collisions": RewardTermCfg(
            func=dribbling_rew.self_collision_cost,
            weight=-2.0,
            params={
                "sensor_name": self_collision_cfg.name,
                "force_threshold": 1.0,
            },
        ),
        "torques": RewardTermCfg(
            func=dribbling_rew.torques_penalty,
            weight=-1e-5,
            params={"asset_cfg": robot_cfg},
        ),
        "dof_vel": RewardTermCfg(
            func=dribbling_rew.dof_vel_penalty,
            weight=-1e-5,
            params={"asset_cfg": robot_cfg},
        ),
        "dof_acc": RewardTermCfg(
            func=dribbling_rew.dof_acc_penalty,
            weight=-2.5e-7,
            params={"asset_cfg": robot_cfg},
        ),
    }

    ##
    # Terminations
    ##

    terminations = {
        "time_out": TerminationTermCfg(func=vel_mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=vel_mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
        ),
    }

    ##
    # Scene
    ##

    scene = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={
            "robot": _get_t1_robot_cfg(),
            "ball": _get_ball_cfg(),
        },
        sensors=(feet_ground_cfg, self_collision_cfg),
        num_envs=4096,
        extent=2.5,
    )

    ##
    # Play-mode overrides
    ##

    if play:
        observations["actor"].enable_corruption = False
        events.pop("push_robot", None)

    ##
    # Assemble
    ##

    return ManagerBasedRlEnvCfg(
        scene=scene,
        observations=observations,
        actions=actions,
        commands={},
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum={},
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="Trunk",
            distance=4.0,
            elevation=-10.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=45,
            njmax=300,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                ccd_iterations=50,
            ),
            contact_sensor_maxmatch=64,
        ),
        decimation=4,  # Control freq = 1/(0.005*4) = 50 Hz (matches original)
        episode_length_s=20.0,
    )
