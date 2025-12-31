# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from dataclasses import MISSING
from typing import Literal

# import isaaclab.terrains.trimesh.mesh_terrains as mesh_terrains
import isaaclab.terrains.trimesh.utils as mesh_utils_terrains
from isaaclab.utils import configclass

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from .trimesh_terrains import wall_terrain, rails_terrain, soccer_terrain

"""
Different trimesh terrain configurations.
"""

@configclass
class MeshSoccerTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with box rails as extrusions."""

    function = soccer_terrain
    rail_thickness_range: tuple[float, float] = MISSING
    """The thickness of the inner and outer rails (in m)."""
    rail_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the rails (in m)."""
    goal_width: float = 3
    goal_depth: float = 0.5

@configclass
class MeshWallTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with box rails as extrusions."""

    function = wall_terrain

    rail_thickness_range: tuple[float, float] = MISSING
    """The thickness of the inner and outer rails (in m)."""
    rail_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the rails (in m)."""
    # platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""

@configclass
class MeshRailsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with box rails as extrusions."""

    function = rails_terrain

    rail_thickness_range: tuple[float, float] = MISSING
    """The thickness of the inner and outer rails (in m)."""
    rail_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the rails (in m)."""
    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""

