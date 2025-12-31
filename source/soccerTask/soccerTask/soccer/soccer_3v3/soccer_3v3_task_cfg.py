from isaaclab.utils import configclass
from locomotion_rl_lab.locomotion import mdp

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg, ActionTermCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from soccerLab.mdp import events, observations, actions

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=observations.soccerLab_get_team_lin_vel, clip=(-100, 100))

        def __post_init__(self):
            # self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True
            
    policy:PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=0.15)

@configclass
class CommandsCfg:
    pass

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class CurriculumsCfg:
    pass

@configclass
class ActionsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2), clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5), clip=(-100, 100))
        actions = ObsTerm(func=mdp.last_action, clip=(-12, 12))

        def __post_init__(self):
            # self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True
    
    # Method 0
    red_team_action = actions.TeamNaviActionCfg(
        team_name="group_1_cfg",
        policy_path="data/ckpts/g1/g1_29d_loco_walk.pt",
        low_level_observations=PolicyCfg(),
        low_level_actions=mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
        )
    )
    
    # Method 1
    bp0_action = actions.PreTrainedVelPolicyActionCfg(
        asset_name="robot_bp0",
        policy_path="data/ckpts/g1/g1_29d_loco_walk.pt",
        low_level_observations=PolicyCfg(),
        low_level_actions=mdp.JointPositionActionCfg(
            asset_name="robot_bp0", joint_names=[".*"], scale=0.25, use_default_offset=True
        )
    )
    
    def __post_init__(self):
        # Method 2
        from IsaacNPC.utils.terms_tools import npc_make_robot_action_term
        from IsaacNPC.template.g1.vel_policy_cfg import G1VelPolicyActionsCfg
        action_cfg = G1VelPolicyActionsCfg()
        npc_make_robot_action_term(action_cfg, "robot_bp1", self, term_type=ActionTermCfg)
        
        from IsaacNPC.template.g1.npc_zero_vel_policy_cfg import G1NPCVelPolicyActionsCfg
        action_cfg = G1NPCVelPolicyActionsCfg()
        npc_make_robot_action_term(action_cfg, "robot_bp2", self, term_type=ActionTermCfg)
    
@configclass
class EventsCfg:
    
    # startup
    physics_material = EventTerm(
        func=events.soccerLab_team_randomize_func,
        mode="startup",
        params={
            "low_func": mdp.randomize_rigid_body_material,
            "team_name": "group_1_cfg",
            "asset_cfg": None,
            "static_friction_range": (0.65, 0.65),
            "dynamic_friction_range": (0.65, 0.65),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    
    reset_joint = EventTerm(
        func=events.soccerLab_team_reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
            "team_names": ["group_1_cfg"]
        },
    )
    
    reset_base = EventTerm(
        func=events.soccerLab_team_reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.05, 0.05), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "team_names": ["group_1_cfg"]
        },
    )
    
    def __post_init__(self):
        # Method 2
        from IsaacNPC.utils.terms_tools import npc_make_robot_event_term
        from IsaacNPC.template.g1.vel_policy_cfg import G1VelPolicyEventsCfg
        event_cfg = G1VelPolicyEventsCfg()
        npc_make_robot_event_term(event_cfg, "robot_bp0", self, term_type=EventTerm)
        npc_make_robot_event_term(event_cfg, "robot_bp1", self, term_type=EventTerm)
        npc_make_robot_event_term(event_cfg, "robot_bp2", self, term_type=EventTerm)
        