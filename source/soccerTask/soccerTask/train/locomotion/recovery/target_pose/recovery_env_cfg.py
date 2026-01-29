import math
import pathlib

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CommandTermCfg as CmdTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from ..agile_wbc.mdp.terrains import STAND_UP_ROUGH_TERRAIN_CFG


FILE_DIR = pathlib.Path(__file__).parent

REST_DURATION_S = 2.0


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Minimal scene configuration for stand-up recovery."""

    # ground terrain: reuse the rough terrain config but without curriculum logic here
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=STAND_UP_ROUGH_TERRAIN_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                f"TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # robot: to be specified by robot-specific configs (e.g. T1)
    # robot = ...

    # basic sensors (robot body path is still generic)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=1,
        track_air_time=False,
    )

    # height_measurement_sensor: to be specified by robot-specific configs
    height_measurement_sensor: RayCasterCfg = None

    # simple light for visualization
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class ObservationsCfg:
    """Simplified observations for policy and critic."""

    @configclass
    class Policy(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action, clip=(-12, 12))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.flatten_history_dim = True

    @configclass
    class Critic(ObsGroup):
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        base_height = ObsTerm(
            func=mdp.base_height_from_sensor,
            params={"sensor_cfg": SceneEntityCfg("height_measurement_sensor")},
        )
        target_body_pos_delta = ObsTerm(
            func=mdp.target_body_pos_delta,
            params={"command_name": "target_pose"},
        )

        def __post_init__(self):
            self.history_length = 3
            self.enable_corruption = True
            self.concatenate_terms = True
            self.flatten_history_dim = True

    policy: Policy = Policy()
    critic: Critic = Critic()


@configclass
class ActionsCfg:
    """Simple joint position control."""

    joint_pos = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        clip={".*": (-3.14, 3.14)},
        use_zero_offset=True,
        preserve_order=True,
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    target_pose = mdp.TargetPoseCommandCfg(
        asset_name="robot",
        initial_pose_path=None, # place holder
        keybody_names=None , # place holder
    )


@configclass
class RewardsCfg:
    """Minimal rewards for stand-up recovery."""

    # task terms
    target_body_pos = RewTerm(
        func=mdp.target_body_pos_error_exp,
        weight=1.0,
        params={
            "command_name": "target_pose",
            "std": 0.3,
        },
    )
    base_height = RewTerm(
        func=mdp.base_height_exp,
        weight=2.0,
        params={
            "target_height": 0.0,  # placeholder, override in robot-specific rewards if needed
            "std": 0.25,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
        },
    )
    orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["Trunk"])},
    )


    # Aux
    action_l2 = RewTerm(func=mdp.action_l2_if_actor_active, weight=-0.05, params={"rest_duration_s": REST_DURATION_S})
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)

    root_acc = RewTerm(
        func=mdp.root_acc_l2,  # type: ignore
        weight=-5e-2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot_link.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_link.*"),
        },
    )



@configclass
class TerminationsCfg:
    """Basic termination conditions."""

    # Only time-out termination: no explicit stand-up termination
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class EventCfg:
    """Reset events to initialize states (including fallen poses)."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform_some_standing,
        mode="reset",
        params={
            "standing_ratio": 0.2,
            "pose_range": {
                # start from a pose range close to the nominal initial state
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "yaw": (-math.radians(20), math.radians(20)),
                "roll": (-math.radians(10), math.radians(10)),
                "pitch": (-math.radians(10), math.radians(10)),
            },
            "velocity_range": {
                # smaller initial velocity range; curriculum will expand this
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.0, 0.0),
                "roll": (-1.0, 1.0),
                "pitch": (-1.0, 1.0),
                "yaw": (-1.0, 1.0),
            },
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*foot_link.*"]),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.0, 2.0),
            "velocity_range": (-1.0, 1.0),
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the simple stand-up MDP."""

    reset_pose_curriculum = CurrTerm(
        func=mdp.reset_pose_curriculum,
        params={
            # initial ranges (close to reset_base) as part of curriculum state
            "init_pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "yaw": (-math.radians(20), math.radians(20)),
                "roll": (-math.radians(10), math.radians(10)),
                "pitch": (-math.radians(10), math.radians(10)),
            },
            "init_velocity_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.0, 0.0),
                "roll": (-1.0, 1.0),
                "pitch": (-1.0, 1.0),
                "yaw": (-1.0, 1.0),
            },
            # target ranges the curriculum will move toward
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "yaw": (-math.pi, math.pi),
                "roll": (-math.radians(30), math.radians(30)),
                "pitch": (-math.radians(30), math.radians(30)),
            },
            "velocity_range": {
                "x": (-3.0, 3.0),
                "y": (-3.0, 3.0),
                "z": (0.0, 0.0),
                "roll": (-3.0, 3.0),
                "pitch": (-3.0, 3.0),
                "yaw": (-3.0, 3.0),
            },
            "alpha": 0.05,
            # use the main task reward as curriculum signal
            "reward_term_name": "target_body_pos",
        },
    )

@configclass
class ViewerCfg:
    """Simple viewer configuration."""

    eye: tuple[float, float, float] = (0.0, -6.0, 3.0)
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cam_prim_path: str = "/OmniverseKit_Persp"
    resolution: tuple[int, int] = (1280, 720)
    origin_type = "asset_root"
    asset_name: str = "robot"
    env_index: int = 0


@configclass
class SimpleRecoveryEnvCfg(ManagerBasedRLEnvCfg):
    """A simplified stand-up recovery environment configuration.

    Compared to ``RecoveryEnvCfg``, this version:
    - removes curriculum
    - removes external perturbation events
    - uses a smaller set of observations and rewards
    """

    scene: SceneCfg = SceneCfg(num_envs=2048, env_spacing=2.5)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    viewer: ViewerCfg = ViewerCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 10
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 500
        self.sim.render_interval = self.decimation

        # match physics material with terrain
        self.sim.physics_material = self.scene.terrain.physics_material

        # update sensors at simulation rate
        self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.height_measurement_sensor is not None:
            self.scene.height_measurement_sensor.update_period = self.sim.dt

        # 如果启用了地形 curriculum，则打开地形生成器的 curriculum 开关
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
