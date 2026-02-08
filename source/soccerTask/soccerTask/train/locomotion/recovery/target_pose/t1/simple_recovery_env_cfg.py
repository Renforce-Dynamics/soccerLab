"""T1-specific simple recovery environment configuration."""

from isaaclab.utils import configclass
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

from robotlib.soccerLab import booster_t1

import pathlib

from .. import mdp
from ..recovery_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    CurriculumCfg,
    EventCfg,
    ObservationsCfg,
    SimpleRecoveryEnvCfg,
    RewardsCfg,
    SceneCfg,
    TerminationsCfg,
)


@configclass
class T1SceneCfg(SceneCfg):
    """T1-specific scene configuration for the simple recovery task."""

    # robots
    robot = booster_t1.T1_DELAYED_DC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # height measurement sensor on the trunk
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
class T1RewardsCfg(RewardsCfg):
    """T1-specific rewards for the simple recovery task."""

    # stand-up objective: encourage target height around DEFAULT_TRUNK_HEIGHT
    base_height = RewTerm(
        func=mdp.base_height_exp,
        weight=10.0,
        params={
            "target_height": booster_t1.DEFAULT_TRUNK_HEIGHT,
            "std": 0.25,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
        },
    )


@configclass
class T1CommandsCfg(CommandsCfg):
    """T1-specific target pose command configuration."""

    target_pose = mdp.TargetPoseCommandCfg(
        asset_name="robot",
        initial_pose_path=str(pathlib.Path(__file__).resolve().parent / "initial_pose.json"),
        keybody_names=[
            r".*foot_link.*",
            r".*hand_link.*",
            r"Trunk",
            r"Waist",
        ],
    )


@configclass
class T1SimpleRecoveryEnvCfg(SimpleRecoveryEnvCfg):
    """Simple stand-up recovery env specialized for the T1 robot."""

    scene: T1SceneCfg = T1SceneCfg(num_envs=2048, env_spacing=2.5)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: T1CommandsCfg = T1CommandsCfg()

    rewards: T1RewardsCfg = T1RewardsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

