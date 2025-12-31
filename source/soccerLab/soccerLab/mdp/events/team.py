import copy
import torch
from isaaclab.utils import configclass
from dataclasses import MISSING
from typing import List, TYPE_CHECKING
from . import single
from soccerLab.utils import team_tools

from isaaclab.managers import ManagerTermBase, EventTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import mdp

if TYPE_CHECKING:
    from isaaclab.assets.articulation import Articulation
    from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv, ManagerBasedRLEnvCfg


    from soccerTask.soccer.multi_player_soccer_env_cfg import MultiPlayerSoccerEnvCfg
    from soccerLab.soccer_game_cfg import SoccerGameCfg, SoccerTeamCfg


def soccerLab_team_reset_root_state_uniform(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    team_names: str = None,
) -> List["Articulation"]:
    if env_ids is None: env_ids = torch.arange(env.num_envs, device=env.device)
    robots = team_tools.soccerLab_get_team_robots(env, team_names)
    for robot in robots:
        single.reset_root_state_uniform(env, env_ids=env_ids, pose_range=pose_range, velocity_range=velocity_range, asset=robot)

def soccerLab_team_reset_joints_by_scale(
    env             : "ManagerBasedRLEnv",
    env_ids         : torch.Tensor,
    position_range  : tuple[float, float],
    velocity_range  : tuple[float, float],
    team_names      : List[str] = None
) -> List["Articulation"]:
    if env_ids is None: env_ids = torch.arange(env.num_envs, device=env.device)
    robot_names = team_tools.soccerLab_get_team_robot_names(env, team_names)
    for robot_name in robot_names:
        mdp.reset_joints_by_scale(
            env=env, env_ids=env_ids,
            position_range=position_range,
            velocity_range=velocity_range,
            asset_cfg=SceneEntityCfg(robot_name)
        )

class soccerLab_team_randomize_func(ManagerTermBase):
    def __init__(self, cfg: "EventTermCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.low_func_type: ManagerTermBase = cfg.params.pop("low_func")
        self.team_name: str = cfg.params.pop("team_name")
        self.team_cfg = team_tools.soccerLab_get_team_cfg(env, [self.team_name])[0]
        self.low_funcs = []
        for player in self.team_cfg.players:
            _cfg = copy.deepcopy(cfg)
            _cfg.params["asset_cfg"] = SceneEntityCfg(
                name=player.asset_name, joint_names=".*", body_names=".*"
            )
            self.low_funcs.append(
                self.low_func_type(
                    cfg=_cfg,
                    env=env
                )
            )

    def __call__(self, *args, **kwargs):
        for term in self.low_funcs:
            term(*args, **kwargs)
    