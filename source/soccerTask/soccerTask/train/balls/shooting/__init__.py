# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
gym.register(
    id="g1-shooting",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "soccerTask.train.balls.shooting.shooting_env_cfg:ShootingEnvCfg",
        "play_env_cfg_entry_point": "soccerTask.train.balls.shooting.shooting_env_cfg:ShootingPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "soccerTask.train.balls.shooting.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
