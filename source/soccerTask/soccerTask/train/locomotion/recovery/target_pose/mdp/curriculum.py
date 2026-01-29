from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv


def reset_pose_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],  # required by CurriculumTerm, not used explicitly
    init_pose_range: Mapping[str, tuple[float, float]] = {},
    init_velocity_range: Mapping[str, tuple[float, float]] = {},
    pose_range: Mapping[str, tuple[float, float]] = {},
    velocity_range: Mapping[str, tuple[float, float]] = {},
    alpha: float = 0.05,
    reward_term_name: str = "target_body_pos",
) -> torch.Tensor:
    """Curriculum that smoothly nudges reset ranges toward the curriculum term ranges.

    Aligned with the usage pattern of ``lin_vel_cmd_levels``:

    - Called periodically by :class:`CurriculumTermCfg` with the ``env`` / ``env_ids`` interface.
    - Updates the internal reset ranges only at episode boundaries and returns a scalar tensor
      that can be used as a measure of curriculum progress.

    More concretely:
    - ``pose_range`` / ``velocity_range`` are the curriculum target ranges (passed from the
      term config via ``params``).
    - On the first call we back up the original reset ranges; after that, at every episode end,
      the current ranges are exponentially moved toward the target ranges with ratio ``alpha``.
    """
    # Reset configuration used by the reset event (similar to how lin_vel_cmd_levels operates on command cfg).
    reset_cfg = env.event_manager.get_term_cfg("reset_base")

    # Only attempt to increase difficulty at episode boundaries *and* when reward is high enough.
    if env.common_step_counter % env.max_episode_length == 0:
        # Compute mean episode reward for the specified term over the active env_ids.
        reward_term = env.reward_manager.get_term_cfg(reward_term_name)
        reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

        # Mirror the lin_vel_cmd_levels pattern: only increase difficulty when reward
        # reaches at least 80% of the term's nominal weight.
        if reward > reward_term.weight * 0.8:
            # Use current reset ranges if present, otherwise fall back to the curriculum's
            # initial pose / velocity ranges provided via params.
            curr_pose_range: Mapping[str, tuple[float, float]] = reset_cfg.params.get("pose_range") or init_pose_range
            curr_vel_range: Mapping[str, tuple[float, float]] = reset_cfg.params.get("velocity_range") or init_velocity_range

            # Nudge current pose ranges toward curriculum pose ranges.
            new_pose_range: dict[str, tuple[float, float]] = {}
            for key, (curr_low, curr_high) in curr_pose_range.items():
                tgt_low, tgt_high = pose_range.get(key, (curr_low, curr_high))
                low = (1.0 - alpha) * float(curr_low) + alpha * float(tgt_low)
                high = (1.0 - alpha) * float(curr_high) + alpha * float(tgt_high)
                new_pose_range[key] = (low, high)

            # Nudge current velocity ranges toward curriculum velocity ranges.
            new_vel_range: dict[str, tuple[float, float]] = {}
            for key, (curr_low, curr_high) in curr_vel_range.items():
                tgt_low, tgt_high = velocity_range.get(key, (curr_low, curr_high))
                low = (1.0 - alpha) * float(curr_low) + alpha * float(tgt_low)
                high = (1.0 - alpha) * float(curr_high) + alpha * float(tgt_high)
                new_vel_range[key] = (low, high)

            reset_cfg.params["pose_range"] = new_pose_range
            reset_cfg.params["velocity_range"] = new_vel_range

    # For logging: return a scalar tensor that roughly indicates curriculum progress.
    # As in lin_vel_cmd_levels, we return a torch.Tensor instead of a Python float.
    angle_keys = ("roll", "pitch", "yaw")
    pose_range_for_log: Mapping[str, tuple[float, float]] = reset_cfg.params.get("pose_range", {})
    max_angle = 0.0
    for key in angle_keys:
        if key in pose_range_for_log:
            lo, hi = pose_range_for_log[key]
            max_angle = max(max_angle, abs(float(lo)), abs(float(hi)))

    return torch.tensor(max_angle, device=env.device)

