"""AMPRunnerCfg for G1 kicking AMP task (mjlab backend).

Ported from G1_kicking/kick_task/rsl_rl_amp_cfg.py.
Uses beyondAMP.mjlab.rsl_rl for the mjlab-compatible AMP runner/algorithm.
"""

from __future__ import annotations

import os
import pathlib

from beyondAMP.mjlab.obs_groups import AMPObsBaiscTerms
from beyondAMP.mjlab.rsl_rl import (
    AMPPPOAlgorithmCfg,
    AMPRunnerCfg,
    RslRlPpoActorCriticCfg,
)
from beyondAMP.motion.motion_dataset import MotionDatasetCfg


# ---------------------------------------------------------------------------
# G1 key body names and anchor (same as G1_kicking/robot/g1_keys.py)
# ---------------------------------------------------------------------------

G1_KEY_BODY_NAMES: list[str] = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

G1_ANCHOR_NAME: str = "torso_link"

# ---------------------------------------------------------------------------
# Motion file resolution
# ---------------------------------------------------------------------------

_SOCCERLAB_ROOT = pathlib.Path(__file__).resolve().parents[6]  # soccerLab/
_DEFAULT_KICK_DIR = _SOCCERLAB_ROOT / "data" / "datasets" / "g1_kick_skill"


def _resolve_kick_motion_files() -> list[str]:
    """Find available kick motion .npz files."""
    kick_dir = os.getenv("AMP_KICK_MOTION_DIR", str(_DEFAULT_KICK_DIR))
    motion_list_file = os.path.join(kick_dir, "motion_files_kick.txt")

    if os.path.isfile(motion_list_file):
        with open(motion_list_file) as f:
            files = [
                os.path.join(kick_dir, line.strip())
                for line in f
                if line.strip() and not line.startswith("#")
            ]
            files = [f for f in files if os.path.isfile(f)]
            if files:
                return files

    # Fallback: use the single known motion file
    fallback = str(_DEFAULT_KICK_DIR / "wo_cf_shoot_74_06.npz")
    if os.path.isfile(fallback):
        return [fallback]

    return []


# ---------------------------------------------------------------------------
# Base config
# ---------------------------------------------------------------------------

def _base_runner_cfg(
    *,
    run_name: str,
    amp_reward_coef: float,
    amp_task_reward_lerp: float,
    max_iterations: int = 120_000,
) -> AMPRunnerCfg:
    return AMPRunnerCfg(
        num_steps_per_env=24,
        max_iterations=max_iterations,
        save_interval=100,
        experiment_name="g1_kick_skill",
        run_name=run_name,
        empirical_normalization=True,
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        ),
        algorithm=AMPPPOAlgorithmCfg(
            class_name="AMPPPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        amp_data=MotionDatasetCfg(
            motion_files=_resolve_kick_motion_files(),
            body_names=G1_KEY_BODY_NAMES,
            anchor_name=G1_ANCHOR_NAME,
            amp_obs_terms=AMPObsBaiscTerms,
        ),
        amp_discr_hidden_dims=[256, 256],
        amp_reward_coef=amp_reward_coef,
        amp_task_reward_lerp=amp_task_reward_lerp,
    )


# ---------------------------------------------------------------------------
# Stage-specific factories
# ---------------------------------------------------------------------------

def g1_kick_amp_runner_cfg() -> AMPRunnerCfg:
    """Stage A: strong AMP prior, high task reward lerp."""
    return _base_runner_cfg(
        run_name="stageA",
        amp_reward_coef=0.5,
        amp_task_reward_lerp=0.85,
    )


def g1_kick_bootstrap_amp_runner_cfg() -> AMPRunnerCfg:
    """Bootstrap: weaker AMP prior, more room for task reward."""
    return _base_runner_cfg(
        run_name="bootstrap",
        amp_reward_coef=0.7,
        amp_task_reward_lerp=0.55,
    )


def g1_kick_stage_b_amp_runner_cfg() -> AMPRunnerCfg:
    """Stage B: slightly reduced AMP prior."""
    return _base_runner_cfg(
        run_name="stageB",
        amp_reward_coef=0.45,
        amp_task_reward_lerp=0.85,
    )


def g1_kick_stage_c_amp_runner_cfg() -> AMPRunnerCfg:
    """Stage C: weakest AMP prior, room for polish rewards."""
    return _base_runner_cfg(
        run_name="stageC",
        amp_reward_coef=0.35,
        amp_task_reward_lerp=0.75,
    )
