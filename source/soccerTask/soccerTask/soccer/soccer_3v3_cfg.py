from isaaclab.utils import configclass
from soccerLab.soccer_game_cfg import SoccerGameCfg, SoccerTeamCfg

from isaaclab_assets.robots.unitree import UNITREE_A1_CFG, UNITREE_GO1_CFG

@configclass
class Soccer3v3Cfg(SoccerGameCfg):
    group_1_cfg: SoccerTeamCfg = SoccerTeamCfg(
        team_name="red",
        players=[
            SoccerTeamCfg.PlayerCfg(
                name = "rp0",
                init_pos=(-1, -1)
            ),
            SoccerTeamCfg.PlayerCfg(
                name = "rp1",
                init_pos=(0, -1)
            ),
            SoccerTeamCfg.PlayerCfg(
                name = "rp2",
                init_pos=(1, -1)
            ),
        ],
        robot_cfg=UNITREE_A1_CFG
    )
    group_2_cfg: SoccerTeamCfg = SoccerTeamCfg(
        team_name="blue",
        players=[
            SoccerTeamCfg.PlayerCfg(
                name = "bp0",
                init_pos=(-1, 1)
            ),
            SoccerTeamCfg.PlayerCfg(
                name = "bp1",
                init_pos=(0, 1)
            ),
            SoccerTeamCfg.PlayerCfg(
                name = "bp2",
                init_pos=(1, 1)
            ),
        ],
        robot_cfg=UNITREE_GO1_CFG
    )