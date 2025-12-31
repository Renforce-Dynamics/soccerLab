import gymnasium as gym
from . import soccer_3v3_env_cfg

gym.register(
    id="soccerLab_soccer_vel_3v3_g1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_3v3_env_cfg.Soccer3v3EnvCfg,
    },
)
