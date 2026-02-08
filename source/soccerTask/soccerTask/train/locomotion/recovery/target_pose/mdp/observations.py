from __future__ import annotations

from torch import Tensor

from isaaclab.envs import ManagerBasedRLEnv


def reset_phase_flag(env: ManagerBasedRLEnv, rest_duration_s: float) -> Tensor:
    """Binary flag indicating whether each env is currently in the reset/rest phase.

    This mirrors the logic used in ``disable_joints``: an env is considered in the rest
    phase when its episode time is less than ``rest_duration_s``.

    Returns:
        Tensor of shape [num_envs], with 1.0 for envs in rest phase and 0.0 otherwise.
    """
    env_time_s = env.episode_length_buf * env.step_dt
    in_rest = env_time_s < rest_duration_s
    return in_rest.to(env.device, dtype=env.episode_length_buf.dtype)


def reset_phase_progress(env: ManagerBasedRLEnv, rest_duration_s: float) -> Tensor:
    """Continuous value in [0, 1] indicating progress within the reset/rest phase.

    - 0.0  : at episode start
    - 1.0+ : at or beyond the end of the rest phase (clamped to 1.0)
    """
    env_time_s = env.episode_length_buf * env.step_dt
    progress = (env_time_s / max(rest_duration_s, 1e-6)).clamp(min=0.0, max=1.0)
    return progress.to(env.device, dtype=env.episode_length_buf.dtype)


def post_reset_flag(env: ManagerBasedRLEnv) -> Tensor:
    """Binary flag indicating whether each env is past the reset/rest phase.

    Useful to gate rewards/observations so that they only take effect after reset.
    Reads rest_duration_s from env.cfg.rest_duration_s.
    """
    rest_duration_s = env.cfg.rest_duration_s
    env_time_s = env.episode_length_buf * env.step_dt
    past_rest = env_time_s >= rest_duration_s
    return past_rest.to(env.device, dtype=env.episode_length_buf.dtype).unsqueeze(-1)


def last_action_masked(env: ManagerBasedRLEnv, action_name=None) -> Tensor:
    """Return last action masked by post_reset_flag.
    
    Reads rest_duration_s from env.cfg.rest_duration_s.
    """
    rest_duration_s = env.cfg.rest_duration_s
    if action_name is None:
        action = env.action_manager.action
    else:
        action = env.action_manager.get_term(action_name).raw_actions
    env_time_s = env.episode_length_buf * env.step_dt
    past_rest = (env_time_s >= rest_duration_s).to(env.device, dtype=action.dtype)
    return past_rest.unsqueeze(-1) * action
    