import torch
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.assets.articulation import Articulation
    from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

    from soccerTask.soccer.multi_player_soccer_env_cfg import MultiPlayerSoccerEnvCfg
    from soccerLab.soccer_game_cfg import SoccerGameCfg, SoccerTeamCfg

def soccerLab_get_team_cfg(env:"ManagerBasedRLEnv", team_names: List[str]=None) -> List["SoccerTeamCfg"]:
    env_cfg: "MultiPlayerSoccerEnvCfg" = env.cfg
    if team_names is None:
        teams = env_cfg.soccer.teams()
    else:
        teams = [getattr(env_cfg.soccer, name) for name in team_names]
    return teams

def soccerLab_get_team_robots(env:"ManagerBasedRLEnv", team_names: List[str]=None) -> List["Articulation"]:
    env_cfg: "MultiPlayerSoccerEnvCfg" = env.cfg
    if team_names is None:
        teams = env_cfg.soccer.teams()
    else:
        teams = [getattr(env_cfg.soccer, name) for name in team_names]
    robot_names = []
    for team in teams:
        robot_names += team.get_player_entities("robot")
    return [env.scene[robot_name] for robot_name in robot_names]