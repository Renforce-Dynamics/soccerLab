import math
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab.assets import RigidObjectCfg

import soccerLab.terrain as terrain_gen
from soccerLab.soccer_game_cfg import SoccerGameCfg
from robotlib import ROBOTLIB_ASSETLIB_DIR
from dataclasses import dataclass

@dataclass
class SoccerFieldConfig:
    field_length: float  # A
    field_width: float   # B
    goal_depth: float    # C
    goal_width: float    # D
    goal_area_length: float # E
    goal_area_width: float  # F
    penalty_area_length: float # G
    penalty_area_width: float  # H
    penalty_mark_dist: float   # I
    center_circle_dia: float   # J
    border_strip_width: float  # K
    corner_arc_radius: float   # L
    goal_height: float
    post_diameter: float
    line_width: float
    mark_size: float

# Field Presets
S_FIELD = SoccerFieldConfig(
    field_length=9.0, field_width=6.0,
    goal_depth=0.6, goal_width=1.8,  # Using avg depth
    goal_area_length=1.0, goal_area_width=3.0,
    penalty_area_length=2.0, penalty_area_width=4.0,
    penalty_mark_dist=1.5,
    center_circle_dia=1.5,
    border_strip_width=0.0, # S-Field often has min border
    corner_arc_radius=0.0,
    goal_height=1.1, # Approx for small goals
    post_diameter=0.1,
    line_width=0.05,
    mark_size=0.10
)

M_FIELD = SoccerFieldConfig(
    field_length=14.0, field_width=9.0,
    goal_depth=1.0, goal_width=2.4, # Using avg depth
    goal_area_length=1.0, goal_area_width=4.0,
    penalty_area_length=3.0, penalty_area_width=6.0,
    penalty_mark_dist=2.0,
    center_circle_dia=3.0,
    border_strip_width=1.0,
    corner_arc_radius=0.5,
    goal_height=1.8, # M-Field goal height
    post_diameter=0.1,
    line_width=0.05,
    mark_size=0.10
)

L_FIELD = SoccerFieldConfig(
    field_length=22.0, field_width=14.0,
    goal_depth=1.5, goal_width=3.0, # Using avg depth
    goal_area_length=1.0, goal_area_width=5.0,
    penalty_area_length=3.5, penalty_area_width=7.0,
    penalty_mark_dist=2.5,
    center_circle_dia=4.0,
    border_strip_width=1.0,
    corner_arc_radius=1.0,
    goal_height=2.0, # L-Field/MSL goal height
    post_diameter=0.12,
    line_width=0.12,
    mark_size=0.15
)

# Field preset lookup
FIELD_PRESETS = {
    "S": S_FIELD,
    "M": M_FIELD,
    "L": L_FIELD,
}

# Get active field from config
def get_active_field():
    import os
    import json
    config_str = os.environ.get("SOCCER_MATCH_CONFIG", "{}")
    try:
        config = json.loads(config_str) if config_str else {}
        preset = config.get("field", {}).get("preset", "M").upper()
        return FIELD_PRESETS.get(preset, M_FIELD)
    except:
        return M_FIELD

ACTIVE_FIELD = get_active_field()

SOCCER_GROUND_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(ACTIVE_FIELD.field_length + 2 * ACTIVE_FIELD.border_strip_width + 4.0, 
          ACTIVE_FIELD.field_width + 2 * ACTIVE_FIELD.border_strip_width + 4.0),
    border_width=0.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshSoccerTerrainCfg(
            size=(ACTIVE_FIELD.field_length + 2.0, ACTIVE_FIELD.field_width + 2.0), # Slightly larger for safety
            rail_height_range=(0.0, 0.0), # No walls
            rail_thickness_range=(0.0, 0.0),
            proportion=1.0,
            goal_width=ACTIVE_FIELD.goal_width,
            goal_depth=ACTIVE_FIELD.goal_depth
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
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 0.5, 0.0), # Green Grass
            metallic=0.0,
            roughness=0.8,
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
    
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ball",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=f"{ROBOTLIB_ASSETLIB_DIR}/third_party/olympics/urdf/soccer.urdf",
            fix_base=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.5,
                angular_damping=0.5,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.43,
            ),
            joint_drive=None
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.2),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    def setup_soccer_game(self, game_cfg: SoccerGameCfg):
        for team in game_cfg.teams():
            team.setup_soccer_team(self)

    def set_robot_marker(self, prim_name, color_cfg):
        prim_path = "{ENV_REGEX_NS}/" + f"{prim_name}"
        setattr(
            self, f"marker_{prim_name}", 
            RigidObjectCfg(
                prim_path=prim_path + "/team_marker",
                spawn=sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=color_cfg,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=False
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=True
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 1.0),
                ),
            )
        )

    def set_robot_entity(self, prim_name, robot_cfg:ArticulationCfg, scanner=None):
        # robots
        robot_cfg.prim_path = "{ENV_REGEX_NS}/" + prim_name
        robot: ArticulationCfg = robot_cfg
        setattr(self, f"robot_{prim_name}", robot)
        
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
            
            setattr(self, f"height_scanner_{prim_name}", height_scanner)
        contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/" + f"{prim_name}/.*", history_length=3, track_air_time=True)
        setattr(self, f"contact_forces_{prim_name}", contact_forces)

    def __post_init__(self):
        super().__post_init__()
        self._add_field_lines(ACTIVE_FIELD)
        self._add_goal_posts(ACTIVE_FIELD)

    def _add_field_lines(self, cfg: SoccerFieldConfig):
        """Adds visual field lines using thin cuboids."""
        line_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0))
        # Flattened path: spawn directly in env namespace with prefix to avoid missing parent prim error
        lines_prim_path_prefix = "{ENV_REGEX_NS}/line" 
        
        # Helper to create a line
        def create_line(name, size, pos, rot=(1.0, 0.0, 0.0, 0.0)):
            setattr(
                self, f"line_{name}",
                RigidObjectCfg(
                    prim_path=f"{lines_prim_path_prefix}_{name}",
                    spawn=sim_utils.CuboidCfg(
                        size=size,
                        visual_material=line_material,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
                )
            )

        lw = cfg.line_width
        L = cfg.field_length
        W = cfg.field_width
        z_offset = 0.005 # Raise slightly above ground

        # 1. Boundary Lines
        # Top (Side Line)
        create_line("side_top", (L + 2*lw, lw, 0.01), (0.0, W/2 + lw/2, z_offset))
        # Bottom (Side Line)
        create_line("side_btm", (L + 2*lw, lw, 0.01), (0.0, -(W/2 + lw/2), z_offset))
        # Left (Goal Line)
        create_line("goal_left", (lw, W, 0.01), (-(L/2 + lw/2), 0.0, z_offset))
        # Right (Goal Line)
        create_line("goal_right", (lw, W, 0.01), ((L/2 + lw/2), 0.0, z_offset))

        # 2. Center Line
        create_line("center_mid", (lw, W, 0.01), (0.0, 0.0, z_offset))
        
        # 3. Center Mark (Cross approximation)
        cl = 0.6 # length of cross mark
        create_line("center_cross_h", (cl, lw, 0.01), (0.0, 0.0, z_offset))
        create_line("center_cross_v", (lw, cl, 0.01), (0.0, 0.0, z_offset))

        # 4. Goal Area (Small Box)
        # Left
        gal = cfg.goal_area_length
        gaw = cfg.goal_area_width
        # Top segment
        create_line("ga_left_top", (gal, lw, 0.01), (-(L/2 - gal/2), gaw/2 + lw/2, z_offset))
        # Bottom segment
        create_line("ga_left_btm", (gal, lw, 0.01), (-(L/2 - gal/2), -(gaw/2 + lw/2), z_offset))
        # Front segment
        create_line("ga_left_front", (lw, gaw + 2*lw, 0.01), (-(L/2 - gal), 0.0, z_offset))

        # Right
        # Top segment
        create_line("ga_right_top", (gal, lw, 0.01), ((L/2 - gal/2), gaw/2 + lw/2, z_offset))
        # Bottom segment
        create_line("ga_right_btm", (gal, lw, 0.01), ((L/2 - gal/2), -(gaw/2 + lw/2), z_offset))
        # Front segment
        create_line("ga_right_front", (lw, gaw + 2*lw, 0.01), ((L/2 - gal), 0.0, z_offset))

        # 5. Penalty Area (Big Box)
        # Left
        pal = cfg.penalty_area_length
        paw = cfg.penalty_area_width
        create_line("pa_left_top", (pal, lw, 0.01), (-(L/2 - pal/2), paw/2 + lw/2, z_offset))
        create_line("pa_left_btm", (pal, lw, 0.01), (-(L/2 - pal/2), -(paw/2 + lw/2), z_offset))
        create_line("pa_left_front", (lw, paw + 2*lw, 0.01), (-(L/2 - pal), 0.0, z_offset))

        # Right
        # Right
        create_line("pa_right_top", (pal, lw, 0.01), ((L/2 - pal/2), paw/2 + lw/2, z_offset))
        create_line("pa_right_btm", (pal, lw, 0.01), ((L/2 - pal/2), -(paw/2 + lw/2), z_offset))
        create_line("pa_right_front", (lw, paw + 2*lw, 0.01), ((L/2 - pal), 0.0, z_offset))

    def _add_goal_posts(self, cfg: SoccerFieldConfig):
        """Adds visual goal posts using cylinders."""
        # Material: White for posts
        post_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9))
        
        posts_prim_path_prefix = "{ENV_REGEX_NS}/goal"
        
        radius = cfg.post_diameter / 2.0
        h = cfg.goal_height
        w = cfg.goal_width
        L = cfg.field_length
        z_offset = h / 2.0 
        
        # Helper for posts
        def create_post(name, height, pos, rot=(1.0, 0.0, 0.0, 0.0), is_crossbar=False):
            # If crossbar, rotate 90 deg around X (or Y depending on orientation)
            # Default cylinder is along Z
            if is_crossbar:
               # Rotate 90 deg around X axis: (0.7071068, 0.7071068, 0.0, 0.0) -> No, standard quaternion order (w, x, y, z)
               # We need horizontal cylinder along Y axis. 
               # Cylinder default is Z. Rotate 90 deg around X to align with Y.
               # rot = (0.7071068, 0.7071068, 0.0, 0.0)
               pass
            
            spawn_cfg = sim_utils.CylinderCfg(
                    radius=radius,
                    height=height,
                    visual_material=post_material,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True), # Collision enabled for goals!
                    mass_props=sim_utils.MassPropertiesCfg(mass=10.0), # Heavy/Static
                )
            
            setattr(
                self, f"goal_post_{name}",
                RigidObjectCfg(
                    prim_path=f"{posts_prim_path_prefix}_{name}",
                    spawn=spawn_cfg,
                    init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
                )
            )

        # Quaternion for rotating cylinder from Z to Y: 90 deg around X
        # cos(45) = 0.7071068, sin(45) = 0.7071068
        # q = (cos, sin*1, 0, 0) -> (0.7071068, 0.7071068, 0.0, 0.0)
        rot_z_to_y = (0.7071068, 0.7071068, 0.0, 0.0)

        # LEFT GOAL
        # Left Post (y-positive side)
        create_post("left_post_1", h, (-(L/2), w/2, z_offset))
        # Right Post (y-negative side)
        create_post("left_post_2", h, (-(L/2), -w/2, z_offset))
        # Crossbar
        create_post("left_crossbar", w, (-(L/2), 0.0, h), rot=rot_z_to_y)

        # RIGHT GOAL
        # Left Post (y-positive side)
        create_post("right_post_1", h, ((L/2), w/2, z_offset))
        # Right Post (y-negative side)
        create_post("right_post_2", h, ((L/2), -w/2, z_offset))
        # Crossbar
        create_post("right_crossbar", w, ((L/2), 0.0, h), rot=rot_z_to_y)

