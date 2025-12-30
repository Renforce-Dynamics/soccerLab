from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.envs import mdp

@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=0.15)
    pass