from __future__ import annotations

import torch
import copy
from dataclasses import MISSING
from typing import TYPE_CHECKING, List

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    
from soccerLab.soccer_game_cfg import SoccerGameCfg, SoccerTeamCfg
from soccerLab.mdp.actions.single.pretrained_vel_policy_action import PreTrainedVelPolicyAction, PreTrainedVelPolicyActionCfg
from soccerLab.utils.func_tools import has_param
    
class TeamNaviAction(ActionTerm):
    cfg: TeamNaviActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: TeamNaviActionCfg, env: ManagerBasedRLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)
        self.soccer_cfg : SoccerGameCfg = env.cfg.soccer
        self.team_cfg   : SoccerTeamCfg = getattr(self.soccer_cfg, cfg.team_name)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.init_actions()
        
        
    velocity_actions: List[ActionTerm]
    def init_actions(self):
        self.velocity_actions = []
        for player in self.team_cfg.players:
            low_level_observations = copy.deepcopy(self.cfg.low_level_observations)
            for term in vars(low_level_observations).values():
                if isinstance(term, ObsTerm):
                    if has_param(term.func, "asset_cfg"):
                        term.params["asset_cfg"] = SceneEntityCfg(player.asset_name)
            
            low_level_actions = copy.deepcopy(self.cfg.low_level_actions)
            low_level_actions.asset_name = player.asset_name
            
            action = PreTrainedVelPolicyAction(
                PreTrainedVelPolicyActionCfg(
                    asset_name = player.asset_name,
                    policy_path = self.cfg.policy_path,
                    low_level_observations = low_level_observations,
                    low_level_actions = low_level_actions
                ),
                self._env
            )
            self.velocity_actions.append(action)
        
    @property
    def action_dim(self) -> int:
        return len(self.team_cfg.players) * 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        for idx, action in enumerate(self.velocity_actions):
            action.process_actions(actions[:, idx*3:idx*3+3])

    def apply_actions(self):
        for action in self.velocity_actions:
            action.apply_actions()

@configclass
class TeamNaviActionCfg(ActionTermCfg):
    class_type              : ActionTerm = TeamNaviAction
    asset_name              : str = "ball"
    team_name               : str = MISSING
    policy_path             : str = MISSING
    low_level_observations  : ObservationGroupCfg = MISSING
    low_level_actions       : ActionTermCfg = MISSING