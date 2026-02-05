import os
import math
import json
from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.envs import ViewerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg, ActionTermCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from soccerLab.soccer_scene_cfg import SoccerSceneCfg
from soccerLab.soccer_game_cfg import SoccerGameCfg, SoccerTeamCfg
from soccerTask.soccer.multi_player_soccer_env_cfg import MultiPlayerSoccerEnvCfg
from robotlib.beyondMimic.robots.g1 import G1_CYLINDER_CFG
from locomotion_rl_lab.locomotion import mdp
from robotlib.beyondMimic.robots.k1 import K1_CFG, K1_ACTION_SCALE

# Load Config
config_str = os.environ.get("SOCCER_MATCH_CONFIG", "{}")
MATCH_CONFIG = json.loads(config_str) if config_str else {}
FIELD_CONFIG = MATCH_CONFIG.get("field", {"length": 9.0, "width": 6.0})
TEAMS_CONFIG = MATCH_CONFIG.get("teams", {})

def get_robot_cfg(robot_type):
    if robot_type == "k1" or robot_type == "booster":
        return K1_CFG
    return G1_CYLINDER_CFG

# Maximum robots per team (preallocation)
MAX_ROBOTS_PER_TEAM = 7

def create_dynamic_game_cfg():
    def build_team(team_name, color, start_x_sign):
        team_cfg = TEAMS_CONFIG.get(team_name, {})
        active_count = team_cfg.get("count", 1)  # Number of active robots
        robot_type = team_cfg.get("robot_type", "g1")
        spawn_pos = team_cfg.get("spawn_positions", [])
        robot_cfg = get_robot_cfg(robot_type)
        
        players = []
        field_len = float(FIELD_CONFIG.get("length", 9.0))
        y_spacing = 1.0
        
        for i in range(MAX_ROBOTS_PER_TEAM):
            if i < active_count:
                # Active robot: use specified position or default
                if i < len(spawn_pos):
                    pos = spawn_pos[i]
                    if len(pos) < 2:
                        pos = [start_x_sign * (field_len / 4), i * y_spacing - (active_count - 1) * y_spacing / 2]
                    yaw = pos[2] if len(pos) >= 3 else None
                else:
                    # Default position
                    start_y = -((active_count - 1) * y_spacing) / 2
                    pos = [start_x_sign * (field_len / 4), start_y + i * y_spacing]
                    yaw = 0.0 if start_x_sign < 0 else math.pi  # Face center
                
                players.append(SoccerTeamCfg.PlayerCfg(
                    name=f"{team_name[0]}p{i}",
                    init_pos=(pos[0], pos[1]),
                    init_yaw=yaw
                ))
            else:
                # Inactive robot: place off-field
                players.append(SoccerTeamCfg.PlayerCfg(
                    name=f"{team_name[0]}p{i}",
                    init_pos=(100.0, 100.0 + i),  # Stagger slightly to avoid overlap
                    init_yaw=0.0
                ))
        
        return SoccerTeamCfg(team_name=team_name, team_color_cfg=sim_utils.PreviewSurfaceCfg(diffuse_color=color), players=players, robot_cfg=robot_cfg)

    @configclass
    class DynamicSoccerGameCfg(SoccerGameCfg):
        group_1_cfg: SoccerTeamCfg = build_team("red", (0.85, 0.2, 1.0), -1)
        group_2_cfg: SoccerTeamCfg = build_team("blue", (0.15, 0.45, 1.0), 1)
    return DynamicSoccerGameCfg()

# Helper to get all player asset names (always returns all 20)
def get_all_player_names():
    names = []
    for i in range(MAX_ROBOTS_PER_TEAM):
        names.append(f"robot_rp{i}")
    for i in range(MAX_ROBOTS_PER_TEAM):
        names.append(f"robot_bp{i}")
    return names

# Helper to get active player count per team
def get_active_counts():
    red_count = TEAMS_CONFIG.get("red", {}).get("count", 1)
    blue_count = TEAMS_CONFIG.get("blue", {}).get("count", 1)
    return {"red": red_count, "blue": blue_count}

PLAYER_ASSETS = get_all_player_names()
ACTIVE_COUNTS = get_active_counts()

@configclass
class ActionsCfg:
    def __post_init__(self):
        for name in PLAYER_ASSETS:
            team_color = "red" if "rp" in name else "blue"
            cfg = TEAMS_CONFIG.get(team_color, {})
            robot_type = cfg.get("robot_type", "g1")
            
            scale = 0.25 # Default G1 scale
            if robot_type == "k1" or robot_type == "booster":
                 # Use K1_ACTION_SCALE dict if possible, but JointPositionActionCfg expects a single scale or list?
                 # Actually JointPositionActionCfg uses `scale` param. 
                 # In k1.py K1_ACTION_SCALE is a dict {joint_name: scale}.
                 # But JointPositionActionCfg normally takes a float or dict.
                 # Let's import it and check how to use.
                 scale = K1_ACTION_SCALE
            
            setattr(self, f"{name}_joint_pos", mdp.JointPositionActionCfg(
                asset_name=name, joint_names=[".*"], scale=scale, use_default_offset=True
            ))

@configclass
class EventsCfg:
    def __post_init__(self):
        # Startup / Reset for each robot
        for name in PLAYER_ASSETS:
             # Randomize materials
             setattr(self, f"{name}_physics_material", EventTerm(
                 func=mdp.randomize_rigid_body_material, mode="startup",
                 params={"asset_cfg": SceneEntityCfg(name, body_names=".*"), "static_friction_range": (0.3, 1.0), "dynamic_friction_range": (0.3, 1.0), "restitution_range": (0.0, 0.0), "num_buckets": 64}
             ))
             # Reset Base
             setattr(self, f"{name}_reset_base", EventTerm(
                 func=mdp.reset_root_state_uniform, mode="reset",
                 params={"asset_cfg": SceneEntityCfg(name), "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-0.0, 0.0)}, "velocity_range": {}}
             ))
             # Reset Joints
             setattr(self, f"{name}_reset_joints", EventTerm(
                 func=mdp.reset_joints_by_scale, mode="reset",
                 params={"asset_cfg": SceneEntityCfg(name), "position_range": (-0.0, 0.0), "velocity_range": (-0.0, 0.0)}
             ))

    reset_ball = EventTerm(
        func=mdp.reset_root_state_uniform, mode="reset",
        params={"asset_cfg": SceneEntityCfg("ball"), "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-3.14, 3.14)}, "velocity_range": {}}
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            
            # IMPORTANT: Observation order must match the pretrained policy!
            # Expected order per robot (99 dims total):
            # 1. base_lin_vel (3)
            # 2. base_ang_vel (3)
            # 3. projected_gravity (3)
            # 4. velocity_commands (3)
            # 5. joint_pos_rel (29)
            # 6. joint_vel_rel (29)
            # 7. last_action (dynamic dim) - per robot, sliced from the full action

            current_action_idx = 0
            
            for idx, name in enumerate(PLAYER_ASSETS):
                prefix = name + "_"
                
                # Determine robot type and action dim
                team_color = "red" if "rp" in name else "blue"
                cfg = TEAMS_CONFIG.get(team_color, {}) # default to red/blue team config logic from name
                # Fallback logic for name parsing if TEAMS_CONFIG doesn't map directly
                # However, PLAYER_ASSETS come from get_all_player_names which reads TEAMS_CONFIG
                # A simpler way is to check the robot instance if possible, but here we only have config.
                # Let's rely on TEAMS_CONFIG.
                
                robot_type = cfg.get("robot_type", "g1")
                action_dim_per_robot = 22 if (robot_type == "k1" or robot_type == "booster") else 29
                
                # 1. base_lin_vel
                setattr(self, prefix + "base_lin_vel", ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg(name)}, clip=(-100, 100)))
                # 2. base_ang_vel
                setattr(self, prefix + "base_ang_vel", ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg(name)}, scale=0.2, clip=(-100, 100)))
                # 3. projected_gravity
                setattr(self, prefix + "projected_gravity", ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg(name)}))
                # 4. velocity_commands (MUST come before joint obs!)
                setattr(self, prefix + "velocity_commands", ObsTerm(func=mdp.generated_commands, params={"command_name": f"base_velocity_{name}"}))
                # 5. joint_pos_rel
                setattr(self, prefix + "joint_pos_rel", ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg(name)}, clip=(-100, 100)))
                # 6. joint_vel_rel
                setattr(self, prefix + "joint_vel_rel", ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg(name)}, scale=0.05, clip=(-100, 100)))
                
                # 7. last_action per robot (slice from the full action tensor)
                # Create a closure to capture the correct slice indices
                start_idx = current_action_idx
                end_idx = start_idx + action_dim_per_robot
                current_action_idx = end_idx
                
                # Define a function factory to create the slicing function with correct closure
                def make_action_slice_func(s_idx, e_idx):
                    def action_slice(env):
                        return env.action_manager.action[:, s_idx:e_idx]
                    return action_slice
                
                setattr(self, prefix + "last_action", ObsTerm(
                    func=make_action_slice_func(start_idx, end_idx), 
                    clip=(-12, 12)
                )) 

    policy: PolicyCfg = PolicyCfg()

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = None

@configclass
class CommandsCfg:
    def __post_init__(self):
        for name in PLAYER_ASSETS:
             setattr(self, f"base_velocity_{name}", mdp.UniformLevelVelocityCommandCfg(
                asset_name=name,
                resampling_time_range=(10.0, 10.0),
                rel_standing_envs=0.02,
                rel_heading_envs=1.0,
                heading_command=False,
                debug_vis=True,
                ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
                limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.2, 0.2)),
             ))

@configclass
class RewardsCfg:
    # Add alive reward for all (global)
    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    def __post_init__(self):
        for name in PLAYER_ASSETS:
             # Track lin vel
             setattr(self, f"{name}_track_lin_vel", RewTerm(
                 func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, 
                 params={"command_name": f"base_velocity_{name}", "std": math.sqrt(0.25), "asset_cfg": SceneEntityCfg(name)}
             ))
             setattr(self, f"{name}_track_ang_vel", RewTerm(
                 func=mdp.track_ang_vel_z_exp, weight=0.5, 
                 params={"command_name": f"base_velocity_{name}", "std": math.sqrt(0.25), "asset_cfg": SceneEntityCfg(name)}
             ))


@configclass
class RobocupSoccerEnvCfg(MultiPlayerSoccerEnvCfg):
    """Configuration for the robocup soccer environment."""
    scene: SoccerSceneCfg = SoccerSceneCfg(num_envs=4, env_spacing=2.5)
    soccer: SoccerGameCfg = create_dynamic_game_cfg()

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    
    events: EventsCfg = EventsCfg()
    # Basic Terminations
    terminations: TerminationsCfg = TerminationsCfg()

    # Viewer configuration
    viewer: ViewerCfg = ViewerCfg(
        eye=(0.0, -20.0, 10.0),
        lookat=(0.0, 0.0, 0.0),
    )

    def __post_init__(self):
        super().__post_init__()
