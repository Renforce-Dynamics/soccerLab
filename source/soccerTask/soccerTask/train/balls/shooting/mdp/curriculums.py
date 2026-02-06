from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_velocity_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor,
    command_name: str = "ball_target_velocity",
    reward_term_name: str = "track_ball_vel_xy",
    step_delta: float = 0.1,
    success_ratio: float = 0.8,
) -> torch.Tensor:
    """Curriculum to progressively widen ball target-velocity command ranges.

    This mirrors the common IsaacLab/locomotion_rl_lab pattern:
    when the tracking reward is consistently high, expand the command sampling
    range towards ``limit_ranges``.

    Returns a scalar tensor (useful for logging), e.g. current max |v_x|.
    """

    # Normalize env_ids to a torch tensor on the correct device
    if isinstance(env_ids, torch.Tensor):
        env_ids_t = env_ids.to(device=env.device)
    else:
        env_ids_t = torch.tensor(list(env_ids), device=env.device, dtype=torch.long)

    command_term = env.command_manager.get_term(command_name)
    ranges = command_term.cfg.ranges
    limit_ranges = getattr(command_term.cfg, "limit_ranges", None)
    if limit_ranges is None:
        # Nothing to do if the command cfg doesn't define limit ranges.
        return torch.tensor(ranges.lin_vel_x[1], device=env.device)

    reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    # episode_sums are accumulated per-env; normalize by episode length in seconds
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids_t]) / env.max_episode_length_s

    # Update only at episode boundaries (common IsaacLab pattern)
    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term_cfg.weight * success_ratio:
            delta = torch.tensor([-step_delta, step_delta], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)

