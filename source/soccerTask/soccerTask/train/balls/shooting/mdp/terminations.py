from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def ball_out_of_bounds(
    env: ManagerBasedRLEnv,
    max_distance: float = 10.0,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Terminate when the ball is too far from the env origin (xy distance).

    Returns boolean tensor of shape [num_envs].
    """
    ball: RigidObject = env.scene[ball_cfg.name]

    rel = ball.data.root_pos_w - env.scene.env_origins
    dist_xy = torch.linalg.vector_norm(rel[:, :2], dim=1)
    return dist_xy > max_distance
