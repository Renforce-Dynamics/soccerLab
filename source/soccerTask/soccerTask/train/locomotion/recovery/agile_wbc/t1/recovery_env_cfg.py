"""T1 robot-specific recovery environment configuration."""

import math

from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import RayCasterCfg, patterns

from robotlib.soccerLab import booster_t1
from ..recovery_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    CurriculumCfg,
    EventCfg,
    ObservationsCfg,
    RecoveryEnvCfg,
    RewardsCfg,
    SceneCfg,
    TerminationsCfg,
    ViewerCfg,
)
from .. import mdp
from ..recovery_env_cfg import REST_DURATION_S, from_scratch, with_curriculum  # noqa: F401
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass


@configclass
class T1SceneCfg(SceneCfg):
    """T1-specific scene configuration."""

    # robots
    robot = booster_t1.T1_DELAYED_DC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_measurement_sensor = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.0, 0.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=5.0,
    )


@configclass
class T1ActionsCfg(ActionsCfg):
    """T1-specific actions configuration."""

    lift = mdp.LiftActionCfg(
        asset_name="robot",
        link_to_lift="H2",  # Head
        stiffness_forces=5000.0,
        damping_forces=500.0,
        force_limit=300.0,
        height_sensor="height_measurement_sensor",
        target_height=booster_t1.DEFAULT_TRUNK_HEIGHT,
        start_lifting_time_s=3.0,
        lifting_duration_s=10.0,
    )


@configclass
class T1RewardsCfg(RewardsCfg):
    """T1-specific rewards configuration."""

    # Task:
    base_height_rough = RewTerm(
        func=mdp.base_height_exp,
        weight=2.0,
        params={
            "target_height": booster_t1.DEFAULT_TRUNK_HEIGHT,
            "std": 0.5,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
        },
    )
    base_height_medium = RewTerm(
        func=mdp.base_height_exp,
        weight=8.0,
        params={
            "target_height": booster_t1.DEFAULT_TRUNK_HEIGHT,
            "std": 0.25,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
        },
    )
    base_height_fine = RewTerm(
        func=mdp.base_height_exp,
        weight=16.0,
        params={
            "target_height": booster_t1.DEFAULT_TRUNK_HEIGHT,
            "std": 0.1,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
        },
    )

    joint_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_exp_if_standing,
        weight=0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "standing_height_threshold": booster_t1.DEFAULT_TRUNK_HEIGHT * 0.8,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
            "std": 0.1,
        },
    )

    not_moving = RewTerm(
        func=mdp.moving_if_standing,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "weight_lin": 1.0,
            "weight_ang": 1.0,
            "standing_height_threshold": booster_t1.DEFAULT_TRUNK_HEIGHT * 0.8,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
        },
    )

    equal_foot_force = RewTerm(
        func=mdp.equal_foot_force_if_standing,
        weight=2.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_link"),
            "asset_cfg": SceneEntityCfg("robot"),
            "standing_height_threshold": booster_t1.DEFAULT_TRUNK_HEIGHT * 0.8,
            "height_measurement_sensor": SceneEntityCfg("height_measurement_sensor"),
        },
    )

    illegal_contacts = RewTerm(
        func=mdp.illegal_contact,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=booster_t1.UNDESIRED_CONTACTS_LINKS),
            "threshold": 1.0,
        },
    )

    feet_distance = RewTerm(
        func=mdp.feet_distance_from_ref_if_standing,
        weight=-50.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=booster_t1.FEET_LINK_NAMES),
            "ref_distance": 0.2,  # 20cm lateral distance between feet
            "standing_height_threshold": booster_t1.DEFAULT_TRUNK_HEIGHT * 0.8,
        },
    )

    feet_yaw_mean = RewTerm(
        func=mdp.feet_yaw_mean_vs_base_if_standing,
        weight=-5.0,
        params={
            "feet_asset_cfg": SceneEntityCfg("robot", body_names=".*foot_link.*"),
            "base_body_cfg": SceneEntityCfg("robot", body_names="Waist"),
            "standing_height_threshold": booster_t1.DEFAULT_TRUNK_HEIGHT * 0.8,
        },
    )


@configclass
class T1TerminationsCfg(TerminationsCfg):
    """T1-specific terminations configuration."""

    standing = DoneTerm(
        func=mdp.standing,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "min_height": booster_t1.DEFAULT_TRUNK_HEIGHT * 0.8,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
            "duration_s": 5.0,
        },
    )


@configclass
class T1RecoveryEnvCfg(RecoveryEnvCfg):
    """T1-specific recovery environment configuration."""

    scene: T1SceneCfg = T1SceneCfg(num_envs=4096, env_spacing=2.5)
    actions: T1ActionsCfg = T1ActionsCfg()
    rewards: T1RewardsCfg = T1RewardsCfg()
    terminations: T1TerminationsCfg = T1TerminationsCfg()
