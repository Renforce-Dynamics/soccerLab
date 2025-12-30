from isaaclab.utils import configclass
from locomotion_rl_lab.locomotion import mdp
from soccerLab.mdp.actions.team.team_navi_action import TeamNaviActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm

@configclass
class PolicyCfg(ObsGroup):
    base_lin_vel        = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
    base_ang_vel        = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
    projected_gravity   = ObsTerm(func=mdp.projected_gravity)
    velocity_commands   = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos_rel       = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
    joint_vel_rel       = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
    actions             = ObsTerm(func=mdp.last_action, clip=(-12, 12))

    def __post_init__(self):
        # self.history_length = 5
        self.enable_corruption = True
        self.concatenate_terms = True

@configclass
class ActionsCfg:
    red_action = TeamNaviActionCfg(
        team_name="group_1_cfg",
        policy_path="data/ckpts/g1/g1_29d_loco_walk.pt",
        low_level_observations=PolicyCfg(),
        low_level_actions=mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
        )
    )