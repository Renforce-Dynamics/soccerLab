from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg

from soccerLab.multi_player.soccer_scene_cfg import SoccerSceneCfg
from soccerLab.multi_player import \
    RewardsCfg, CommandsCfg, EventsCfg, ObservationsCfg, CurriculumsCfg, TerminationsCfg, ActionsCfg

from .soccer_3v3_cfg import Soccer3v3Cfg

@configclass
class MultiPlayerSoccerEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: SoccerSceneCfg = SoccerSceneCfg(num_envs=4, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumsCfg = CurriculumsCfg()

    soccer: Soccer3v3Cfg = Soccer3v3Cfg()

    def setup_multi_player(self):
        self.scene.setup_soccer_game(self.soccer)
        for team in self.soccer.teams():
            for name in team.get_player_entities("height_scanner"):
                target = getattr(self.scene, name, None)
                if target is not None:
                    target.update_period = self.decimation * self.sim.dt
            for name in team.get_player_entities("contact_forces"):
                target = getattr(self.scene, name, None)
                if target is not None:
                    target.update_period = self.sim.dt

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

        self.setup_multi_player()

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False