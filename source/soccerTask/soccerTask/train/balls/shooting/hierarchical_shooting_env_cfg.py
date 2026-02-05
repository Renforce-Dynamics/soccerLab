"""Hierarchical shooting environment configuration.

This version uses velocity commands as actions, assuming a pre-trained velocity policy
is available for low-level control.
"""

from isaaclab.utils import configclass
from ..ball_env_cfg import SoccerBallSceneCfg
from .shooting_env_cfg import (
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
    EventCfg,
    CurriculumCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise


@configclass
class HierarchicalCommandsCfg:
    """Command specifications for hierarchical shooting.
    
    Same as end-to-end version: command the target velocity of the ball.
    """

    ball_target_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="ball",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(0.0, 0.0)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-2.0, 2.0), lin_vel_y=(-2.0, 2.0), ang_vel_z=(0.0, 0.0)
        ),
    )


@configclass
class HierarchicalActionsCfg:
    """Action specifications for hierarchical shooting.
    
    Hierarchical version: Output velocity commands for the low-level policy.
    The low-level policy (pre-trained velocity task) will execute these commands.
    """

    # TODO: Implement VelocityActionCfg for hierarchical control
    # This should output velocity commands (lin_vel_x, lin_vel_y, ang_vel_z)
    # that are passed to the pre-trained velocity policy
    velocity_command = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(0.0, 0.0),  # Don't resample, use action instead
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-0.5, 0.5)
        ),
    )


@configclass
class HierarchicalObservationsCfg(ObservationsCfg):
    """Observation specifications for hierarchical shooting.
    
    Extends the base observations with current velocity command.
    """

    @configclass
    class HierarchicalPolicyCfg(ObsGroup):
        """Observations for hierarchical policy group."""

        # Robot state observations
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2), clip=(-100, 100)
        )
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), clip=(-100, 100)
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5), clip=(-100, 100)
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-12, 12))

        # Ball state observations (TODO: implement custom MDP functions)
        # ball_pos_rel = ObsTerm(func=mdp.ball_pos_rel, params={"ball_asset_name": "ball"})
        # ball_vel_rel = ObsTerm(func=mdp.ball_vel_rel, params={"ball_asset_name": "ball"})
        # ball_vel_w = ObsTerm(func=mdp.ball_vel_w, params={"ball_asset_name": "ball"})

        # Target command (ball target velocity)
        ball_target_velocity = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "ball_target_velocity"}
        )

        # Current velocity command (output of this policy, input to low-level policy)
        current_velocity_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "velocity_command"}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # Override policy observations
    policy: HierarchicalPolicyCfg = HierarchicalPolicyCfg()

    @configclass
    class HierarchicalCriticCfg(ObsGroup):
        """Observations for hierarchical critic group."""

        # Robot state
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        # Ball state (TODO: implement custom MDP functions)
        # ball_pos_rel = ObsTerm(func=mdp.ball_pos_rel, params={"ball_asset_name": "ball"})
        # ball_vel_rel = ObsTerm(func=mdp.ball_vel_rel, params={"ball_asset_name": "ball"})
        # ball_vel_w = ObsTerm(func=mdp.ball_vel_w, params={"ball_asset_name": "ball"})

        # Target command
        ball_target_velocity = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "ball_target_velocity"}
        )

        # Current velocity command
        current_velocity_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "velocity_command"}
        )

        def __post_init__(self):
            self.history_length = 5

    # Override critic observations
    critic: HierarchicalCriticCfg = HierarchicalCriticCfg()


@configclass
class HierarchicalRewardsCfg(RewardsCfg):
    """Reward terms for hierarchical shooting.
    
    Extends base rewards with velocity command smoothness penalty.
    """

    # Add velocity command smoothness penalty
    # TODO: Implement command_rate_l2 function in mdp/rewards
    # velocity_command_rate = RewTerm(
    #     func=mdp.command_rate_l2, weight=-0.1, params={"command_name": "velocity_command"}
    # )


@configclass
class HierarchicalShootingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the hierarchical shooting environment.
    
    This version uses velocity commands as actions, assuming a pre-trained velocity policy
    is available for low-level control.
    """

    # Scene settings
    scene: SoccerBallSceneCfg = SoccerBallSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: HierarchicalObservationsCfg = HierarchicalObservationsCfg()
    actions: HierarchicalActionsCfg = HierarchicalActionsCfg()
    commands: HierarchicalCommandsCfg = HierarchicalCommandsCfg()
    # MDP settings
    rewards: HierarchicalRewardsCfg = HierarchicalRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class HierarchicalShootingPlayEnvCfg(HierarchicalShootingEnvCfg):
    """Configuration for the hierarchical shooting environment in play mode."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.ball_target_velocity.ranges = self.commands.ball_target_velocity.limit_ranges
