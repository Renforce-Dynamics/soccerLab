from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    
from soccerLab.soccer_game_cfg import SoccerGameCfg, SoccerTeamCfg
    
class TeamAction(ActionTerm):
    cfg: TeamActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: TeamActionCfg, env: ManagerBasedRLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)
        self.soccer_cfg : SoccerGameCfg = env.cfg.soccer
        self.team_cfg   : SoccerTeamCfg = getattr(self.soccer_cfg, cfg.team_name)
        

@configclass
class TeamActionCfg(ActionTermCfg):
    team_name: str = MISSING
