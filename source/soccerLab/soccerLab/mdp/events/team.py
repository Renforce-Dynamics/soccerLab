import torch
from typing import List, TYPE_CHECKING
from . import single
from ..utils import soccerLab_get_team_robots

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
) -> List["Articulation"]:
    if env_ids is None: env_ids = torch.arange(env.num_envs, device=env.device)
    robots = soccerLab_get_team_robots(env)
    for robot in robots:
        single.reset_root_state_uniform(env, env_ids=env_ids, pose_range=pose_range, velocity_range=velocity_range, asset=robot)
