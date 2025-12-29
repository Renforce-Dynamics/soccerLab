
from typing import List, TYPE_CHECKING
from . import single

if TYPE_CHECKING:
    from isaaclab.assets.articulation import Articulation
    from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

    from soccerTask.soccer.multi_player_soccer_env_cfg import MultiPlayerSoccerEnvCfg
    from soccerLab.soccer_game_cfg import SoccerGameCfg, SoccerTeamCfg


def soccerLab_get_team_robots(env:"ManagerBasedRLEnv") -> List["Articulation"]:
    env_cfg: "MultiPlayerSoccerEnvCfg" = env.cfg
    robot_names = env_cfg.soccer.group_1_cfg.get_player_entities("robot")
    return [env.scene[robot_name] for robot_name in robot_names]


def soccerLab_get_team_robot_trans_infos(env:"ManagerBasedRLEnv") -> List["Articulation"]:
    soccerLab_get_team_robots(env)
