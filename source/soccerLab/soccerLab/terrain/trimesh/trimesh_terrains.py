from __future__ import annotations

import numpy as np
import scipy.spatial.transform as tf
import torch
import trimesh
from typing import TYPE_CHECKING

from .utils import *  # noqa: F401, F403
from .utils import make_border, make_plane

if TYPE_CHECKING:
    from mesh_terrains_cfg import MeshWallTerrainCfg, MeshRailsTerrainCfg

def soccer_terrain(difficulty: float, cfg: "MeshWallTerrainCfg"):
    # Simply return a ground plane, no walls
    meshes = []
    
    terrain_height = 1.0
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    # Ground plane's center Z coordinate is set so its top surface is at Z=0
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
    ground_meshes = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes.append(ground_meshes)
    origin = np.array([pos[0], pos[1], 0.0])
    return meshes, origin

def wall_terrain(
    difficulty: float, cfg: MeshWallTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a single wall around the border as an extrusion.

    The wall is created as an extrusion along the outer boundary of the terrain.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain (assuming it reuses relevant fields from MeshRailsTerrainCfg).

    Returns:
        A tuple containing the list of tri-meshes of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration for wall height
    wall_height = cfg.rail_height_range[1] - difficulty * (cfg.rail_height_range[1] - cfg.rail_height_range[0])
    # Use the first thickness value from the range for the wall thickness
    wall_thickness = cfg.rail_thickness_range[0]

    # initialize list of meshes
    meshes_list = list()
    # constants for terrain generation
    terrain_height = 1.0

    # The wall is centered at the terrain center (0.5 * size) and its height is centered at wall_height / 2
    wall_center_z = wall_height * 0.5
    wall_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], wall_center_z)

    # 1. Generate the outer wall
    # Outer size is the full terrain size (or slightly less, depending on how `make_border` is implemented
    # relative to size[0] and size[1] as the outer boundary, but here we assume size is the boundary limit)
    # Let's define the OUTER boundary as the full size (cfg.size)
    outer_size = (cfg.size[0], cfg.size[1])
    
    # Inner size is smaller by 2 * wall_thickness
    # Inner size = (size[0] - 2 * thickness, size[1] - 2 * thickness)
    inner_size = (cfg.size[0] - 2.0 * wall_thickness, cfg.size[1] - 2.0 * wall_thickness)

    # Ensure inner_size dimensions are non-negative, though wall_thickness should be small enough
    # If the inner size is too small (e.g., negative), it might result in a solid block,
    # but for a boundary wall, we assume thickness is less than half of min(size[0], size[1])
    
    # Use make_border to create the wall extrusion
    meshes_list += make_border(outer_size, inner_size, wall_height, wall_center)

    # 2. Generate the ground plane (similar to rails_terrain)
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    # Ground plane's center Z coordinate is set so its top surface is at Z=0
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
    ground_meshes = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground_meshes)

    # specify the origin of the terrain (similar to rails_terrain)
    origin = np.array([pos[0], pos[1], 0.0])

    return meshes_list, origin

def rails_terrain(
    difficulty: float, cfg: MeshRailsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with box rails as extrusions.

    The terrain contains two sets of box rails created as extrusions. The first set  (inner rails) is extruded from
    the platform at the center of the terrain, and the second set is extruded between the first set of rails
    and the terrain border. Each set of rails is extruded to the same height.

    .. image:: ../../_static/terrains/trimesh/rails_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. this is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    rail_height = cfg.rail_height_range[1] - difficulty * (cfg.rail_height_range[1] - cfg.rail_height_range[0])

    # initialize list of meshes
    meshes_list = list()
    # extract quantities
    rail_1_thickness, rail_2_thickness = cfg.rail_thickness_range
    rail_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], rail_height * 0.5)
    # constants for terrain generation
    terrain_height = 1.0
    rail_2_ratio = 0.6

    # generate first set of rails
    rail_1_inner_size = (cfg.platform_width, cfg.platform_width)
    rail_1_outer_size = (cfg.platform_width + 2.0 * rail_1_thickness, cfg.platform_width + 2.0 * rail_1_thickness)
    meshes_list += make_border(rail_1_outer_size, rail_1_inner_size, rail_height, rail_center)
    # generate second set of rails
    rail_2_inner_x = cfg.platform_width + (cfg.size[0] - cfg.platform_width) * rail_2_ratio
    rail_2_inner_y = cfg.platform_width + (cfg.size[1] - cfg.platform_width) * rail_2_ratio
    rail_2_inner_size = (rail_2_inner_x, rail_2_inner_y)
    rail_2_outer_size = (rail_2_inner_x + 2.0 * rail_2_thickness, rail_2_inner_y + 2.0 * rail_2_thickness)
    meshes_list += make_border(rail_2_outer_size, rail_2_inner_size, rail_height, rail_center)
    # generate the ground
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
    ground_meshes = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground_meshes)

    # specify the origin of the terrain
    origin = np.array([pos[0], pos[1], 0.0])

    return meshes_list, origin