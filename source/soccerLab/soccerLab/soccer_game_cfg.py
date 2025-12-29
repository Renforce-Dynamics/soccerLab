from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg

from typing import Tuple, List, Literal
from dataclasses import dataclass, MISSING

@configclass
class SoccerTeamCfg:
    @configclass
    class PlayerCfg:
        name: str = None
        init_pos: Tuple[float, float] = (0.0, 0.0)
        init_facing: Tuple[float, float, float] = (0.0, 0.0, 1.0) # Euler form
        robot_cfg: ArticulationCfg = None

    team_name: str = MISSING
    players: List[PlayerCfg] = MISSING
    robot_cfg: ArticulationCfg = None

    def __post_init__(self):
        for idx, player in enumerate(self.players):
            if self.robot_cfg is not None:
                player.robot_cfg = self.robot_cfg
            if player.name is None:
                player.player_name = f"{self.team_name}{idx:2d}"
            asset_init_pos = player.robot_cfg.init_state
            player.robot_cfg.init_state = asset_init_pos.replace(
                pos = player.init_pos + (asset_init_pos.pos[0]),
                rot = asset_init_pos.rot
            )

    def get_player_entities(self, entity_type=Literal["robot", "height_scanner", "contact_forces"]):
        """
        We usually assign robot name with f'robot_{prim_name}'
        """
        return [f"{entity_type}_{player.name}" for player in self.players]

@configclass
class SoccerGameCfg:
    group_1_cfg: SoccerTeamCfg = MISSING
    group_2_cfg: SoccerTeamCfg = MISSING