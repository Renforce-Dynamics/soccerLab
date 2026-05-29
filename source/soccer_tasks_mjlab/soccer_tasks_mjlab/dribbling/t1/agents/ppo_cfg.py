"""RL configuration for T1 dribbling task (pure PPO, mjlab backend).

Matches hyperparameters from the original IsaacLab training config.
"""

from __future__ import annotations

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def dribbling_ppo_runner_cfg(
    max_iterations: int = 30_000,
) -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for T1 dribbling task."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(768, 256, 128),
            activation="elu",
            obs_normalization=False,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=2,
            num_mini_batches=4,
            learning_rate=1.0e-4,
            schedule="adaptive",
            gamma=0.994,
            lam=0.9,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="t1_dribbling",
        save_interval=100,
        num_steps_per_env=60,
        max_iterations=max_iterations,
    )
