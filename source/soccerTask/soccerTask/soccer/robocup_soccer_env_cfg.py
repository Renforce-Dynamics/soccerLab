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

# Load Config
config_str = os.environ.get("SOCCER_MATCH_CONFIG", "{}")
MATCH_CONFIG = json.loads(config_str) if config_str else {}
FIELD_CONFIG = MATCH_CONFIG.get("field", {"length": 9.0, "width": 6.0})
TEAMS_CONFIG = MATCH_CONFIG.get("teams", {})

def get_robot_cfg(robot_type):
    return G1_CYLINDER_CFG

def create_dynamic_game_cfg():
    def build_team(team_name, color, start_x_sign):
        team_cfg = TEAMS_CONFIG.get(team_name, {})
        count = team_cfg.get("count", 1)
        robot_type = team_cfg.get("robot_type", "g1")
        spawn_pos = team_cfg.get("spawn_positions", [])
        robot_cfg = get_robot_cfg(robot_type)
        players = []
        if spawn_pos and len(spawn_pos) >= count:
             for i in range(count):
                 pos = spawn_pos[i]
                 # Check if pos is valid, fallback if too short
                 if len(pos) < 2: 
                     pos = [start_x_sign * 3.0, 0.0]
                 
                 yaw = None
                 if len(pos) >= 3:
                     yaw = pos[2]
                     
                 players.append(SoccerTeamCfg.PlayerCfg(
                     name=f"{team_name[0]}p{i}", 
                     init_pos=(pos[0], pos[1]),
                     init_yaw=yaw
                 ))
        else:
            field_len = float(FIELD_CONFIG.get("length", 9.0))
            y_spacing = 1.0
            start_y = -((count - 1) * y_spacing) / 2
            for i in range(count):
                players.append(SoccerTeamCfg.PlayerCfg(name=f"{team_name[0]}p{i}", init_pos=(start_x_sign * (field_len/4), start_y + i * y_spacing)))
        return SoccerTeamCfg(team_name=team_name, team_color_cfg=sim_utils.PreviewSurfaceCfg(diffuse_color=color), players=players, robot_cfg=robot_cfg)

    @configclass
    class DynamicSoccerGameCfg(SoccerGameCfg):
        group_1_cfg: SoccerTeamCfg = build_team("red", (0.85, 0.2, 1.0), -1)
        group_2_cfg: SoccerTeamCfg = build_team("blue", (0.15, 0.45, 1.0), 1)
    return DynamicSoccerGameCfg()

# Helper to get all player asset names
def get_all_player_names():
    names = []
    # Red
    cfg = TEAMS_CONFIG.get("red", {})
    count = cfg.get("count", 1)
    for i in range(count): names.append(f"robot_rp{i}")
    # Blue
    cfg = TEAMS_CONFIG.get("blue", {})
    count = cfg.get("count", 1)
    for i in range(count): names.append(f"robot_bp{i}")
    # Fallback to defaults used in build_team (count=1 if missing)
    if not names: 
         names = ["robot_rp0", "robot_bp0"]
    return names

PLAYER_ASSETS = get_all_player_names()

@configclass
class ActionsCfg:
    def __post_init__(self):
        for name in PLAYER_ASSETS:
            setattr(self, f"{name}_joint_pos", mdp.JointPositionActionCfg(
                asset_name=name, joint_names=[".*"], scale=0.25, use_default_offset=True
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
        params={"asset_cfg": SceneEntityCfg("ball"), "pose_range": {"x": (0.5, 1.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)}, "velocity_range": {}}
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
            # 7. last_action (29) - per robot, sliced from the full action
            
            num_robots = len(PLAYER_ASSETS)
            action_dim_per_robot = 29  # G1 has 29 joints
            
            for idx, name in enumerate(PLAYER_ASSETS):
                prefix = name + "_"
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
                start_idx = idx * action_dim_per_robot
                end_idx = start_idx + action_dim_per_robot
                
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
