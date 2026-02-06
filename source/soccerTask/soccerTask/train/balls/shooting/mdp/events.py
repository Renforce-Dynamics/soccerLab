from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def reset_ball_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]] | None = None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> None:
    """Reset the ball root pose and velocity for the specified env ids.

    This mirrors IsaacLab-style reset terms (sample uniform noise and write to sim).

    Args:
        env: Environment instance.
        env_ids: Environments to reset.
        position_range: Dict with keys among {"x","y","z","roll","pitch","yaw"}.
        velocity_range: Dict with keys among {"x","y","z","roll","pitch","yaw"}.
        ball_cfg: Ball asset config.
    """

    if position_range is None:
        position_range = {}
    if velocity_range is None:
        velocity_range = {}

    ball: RigidObject = env.scene[ball_cfg.name]

    root_states = ball.data.default_root_state[env_ids].clone()

    # sample pose deltas
    pose_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    pose_ranges = torch.tensor([position_range.get(k, (0.0, 0.0)) for k in pose_keys], device=ball.device)
    pose_samples = math_utils.sample_uniform(
        pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), 6), device=ball.device
    )

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + pose_samples[:, 0:3]
    ori_delta = math_utils.quat_from_euler_xyz(pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], ori_delta)

    # sample velocity deltas
    vel_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    vel_ranges = torch.tensor([velocity_range.get(k, (0.0, 0.0)) for k in vel_keys], device=ball.device)
    vel_samples = math_utils.sample_uniform(
        vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), 6), device=ball.device
    )
    velocities = root_states[:, 7:13] + vel_samples

    ball.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    ball.write_root_velocity_to_sim(velocities, env_ids=env_ids)
