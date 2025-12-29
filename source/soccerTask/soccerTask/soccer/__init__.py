import gymnasium as gym
from . import multi_player_soccer_env_cfg

gym.register(
    id="Soccer_test",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": multi_player_soccer_env_cfg.MultiPlayerSoccerEnvCfg,
    },
)
