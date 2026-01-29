"""T1 robot recovery environment registration."""

import gymnasium as gym

gym.register(
    id="Loco-T1-Recovery-Simple",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # Simple stand-up environment using the simplified configuration
        "env_cfg_entry_point": f"{__name__}.simple_recovery_env_cfg:T1SimpleRecoveryEnvCfg",
        "rsl_rl_cfg_entry_point": "soccerTask.train.locomotion.recovery.target_pose.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
