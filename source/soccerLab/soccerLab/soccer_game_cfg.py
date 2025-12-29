
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import copy
from typing import Tuple, List, Literal
from dataclasses import dataclass, MISSING

from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager

@configclass
class SoccerTeamCfg:
    @configclass
    class PlayerCfg:
        name: str = None
        init_pos: Tuple[float, float] = (0.0, 0.0)
        init_facing: Tuple[float, float, float] = (0.0, 0.0, 1.0) # Euler form
        
        # General terms
        robot_cfg: ArticulationCfg = None

    team_name: str = MISSING
    players: List[PlayerCfg] = MISSING
    robot_cfg: ArticulationCfg = None
    lowest_level_action_cfg: ActionTermCfg = MISSING
    lowest_level_observation_cfg: ObservationGroupCfg = MISSING

    def __post_init__(self):
        for idx, player in enumerate(self.players):
            if self.robot_cfg is not None:
                player.robot_cfg = copy.deepcopy(self.robot_cfg)
            if player.name is None:
                player.player_name = f"{self.team_name}{idx:2d}"
            asset_init_pos = player.robot_cfg.init_state
            player.robot_cfg.init_state = asset_init_pos.replace(
                pos = player.init_pos + (asset_init_pos.pos[2],),
                rot = asset_init_pos.rot
            )
        self.robot_cfg = None  # clear to avoid misuse

    def get_player_entities(self, entity_type: Literal["robot", "height_scanner", "contact_forces"]):
        """
        We usually assign robot name with f'robot_{prim_name}'
        """
        return [f"{entity_type}_{player.name}" for player in self.players]

@configclass
class SoccerGameCfg:
    group_1_cfg: SoccerTeamCfg = MISSING
    group_2_cfg: SoccerTeamCfg = MISSING
    
    def teams(self) -> List[SoccerTeamCfg]:
        return [var for var in vars(self).values() if isinstance(var, SoccerTeamCfg)]