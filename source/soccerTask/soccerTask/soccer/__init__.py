import gymnasium as gym
from . import soccer_3v3


# Register Robocup Soccer Task
gym.register(
    id="Robocup-Soccer",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robocup_soccer_env_cfg:RobocupSoccerEnvCfg",
        "rsl_rl_cfg_entry_point": "soccerTask.train.locomotion.velocity.g1.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
