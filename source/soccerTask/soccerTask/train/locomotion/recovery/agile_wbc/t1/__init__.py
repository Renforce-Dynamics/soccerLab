"""T1 robot recovery environment registration."""

import gymnasium as gym

gym.register(
    id="Loco-T1-Recovery",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recovery_env_cfg:T1RecoveryEnvCfg",
        "rsl_rl_cfg_entry_point": "soccerTask.train.locomotion.recovery.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
