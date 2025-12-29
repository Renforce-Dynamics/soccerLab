from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def base_pos_z(asset: "Articulation") -> torch.Tensor:
    return asset.data.root_pos_w[:, 2].unsqueeze(-1)


def base_lin_vel(asset: "Articulation") -> torch.Tensor:
    return asset.data.root_lin_vel_b