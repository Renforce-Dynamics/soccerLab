from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from sim2simlib.model.config import Sim2SimCfg, ObservationsCfg, ActionsCfg


@dataclass
class BeyondMimicDatasetCfg:
    """Dataset configuration for motion loading.
    
    Specifies which datasets and splits to load for motion tracking.
    """
    dataset_dirs: list[str]
    """List of dataset directory paths containing NPZ motion files"""
    
    motion_body_names: list[str]
    
    robot_name: str
    """Robot name (must match directory name in dataset)"""
    
    splits: list[str]
    """Dataset splits to load (e.g., ["train"], ["test"], ["train", "val"])"""
    
    anchor_body_name: str
    """Name of the body to use as motion anchor"""
    
    device: str
    """Device for motion dataloader: "cuda" or "cpu" """
    
    dataset_body_names: Optional[list[str]]


@dataclass
class BeyondMimicObservationsCfg(ObservationsCfg):
    motion_observations_terms: list[str]
    
@dataclass
class Sim2SimBeyondMimicCfg(Sim2SimCfg):
    """Configuration for Sim2Sim Mimic model.
    """
    
    mimic_dataset_cfg: BeyondMimicDatasetCfg
    observation_cfg: BeyondMimicObservationsCfg
    policy_body_names: Optional[list[str]]
    