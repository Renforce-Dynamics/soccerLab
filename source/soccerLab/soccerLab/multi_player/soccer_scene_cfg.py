import math
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

import soccerLab.terrain as terrain_gen
from soccerLab.soccer_game_cfg import SoccerGameCfg

SOCCER_GROUND_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(16.0, 16.0),
    border_width=20.0,
    num_rows=2,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshWallTerrainCfg(
            rail_height_range=(1, 1),
            rail_thickness_range=(0.2, 0.2)
        ),
    },
)


@configclass
class SoccerSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=SOCCER_GROUND_CFG,  # None, ROUGH_TERRAINS_CFG
        max_init_terrain_level=SOCCER_GROUND_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    def setup_soccer_game(self, game_cfg: SoccerGameCfg):
        game_cfg.group_1_cfg
        for player in game_cfg.group_1_cfg.players:
            self.set_robot_entity(
                player.name,
                player.robot_cfg
            )

    def set_robot_entity(self, prim_name, robot_cfg:ArticulationCfg, scanner=None):
        # robots
        robot: ArticulationCfg = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/" + prim_name)
        if scanner is not None:
            # sensors
            height_scanner = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/" + f"{prim_name}/{scanner}",
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
                attach_yaw_only=True,
                pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
                debug_vis=False,
                mesh_prim_paths=["/World/ground"],
            )
        contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/" + f"{prim_name}/.*", history_length=3, track_air_time=True)

        setattr(self, f"robot_{prim_name}", robot)
        setattr(self, f"height_scanner_{prim_name}", height_scanner)
        setattr(self, f"contact_forces_{prim_name}", contact_forces)

