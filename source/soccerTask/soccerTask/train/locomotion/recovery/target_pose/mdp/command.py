from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING, List, Sequence
import json
import re

import torch
from torch import Tensor

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class TargetPoseCommand(CommandTerm):
    """Command that provides a single target body pose loaded from JSON."""

    cfg: "TargetPoseCommandCfg"

    def __init__(self, cfg: "TargetPoseCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        if cfg.initial_pose_path is None:
            raise ValueError("TargetPoseCommandCfg.initial_pose_path must be provided.")

        self.robot: Articulation = env.scene[cfg.asset_name]

        path = Path(cfg.initial_pose_path)
        if not path.is_absolute():
            path = Path.cwd() / path

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list) or len(data) == 0:
            raise RuntimeError(f"Initial pose JSON at {path} must be a non-empty list.")

        entry = data[0]
        self.data = entry
        all_body_names = entry.get("body_names", [])

        # Resolve which bodies to use based on keybody_names patterns.
        patterns = cfg.keybody_names or [".*"]
        selected_indices: list[int] = []
        for i, name in enumerate(all_body_names):
            for pat in patterns:
                if pat == ".*" or re.fullmatch(pat, name):
                    selected_indices.append(i)
                    break

        if not selected_indices:
            selected_indices = list(range(len(all_body_names)))

        self.body_names = [all_body_names[i] for i in selected_indices]
        index_tensor = torch.tensor(selected_indices, dtype=torch.long, device=self.device)

        body_pos_w_all = torch.tensor(entry["body_pos_w"], dtype=torch.float32, device=self.device)
        body_quat_w_all = torch.tensor(entry["body_quat_w"], dtype=torch.float32, device=self.device)

        body_pos_w = body_pos_w_all[index_tensor]
        body_quat_w = body_quat_w_all[index_tensor]

        # Indices into the robot articulation for the selected bodies.
        body_indices_np, _ = self.robot.find_bodies(self.body_names, preserve_order=True)
        self.body_indices = torch.tensor(body_indices_np, dtype=torch.long, device=self.device)

        self._raw_body_pos_w = body_pos_w
        self._raw_body_quat_w = body_quat_w

        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)

        # Initialize command tensors.
        self._update_command()

    def _resample_command(self, env_ids: Sequence[int]):
        # Static single-frame command: just recompute world-aligned poses.
        self._update_command()

    def _update_metrics(self):
        # Track simple body position error between robot and target.
        robot_pos = self.robot.data.body_pos_w[:, self.body_indices]
        diff = robot_pos - self._body_pos_w
        self.metrics["error_body_pos"] = torch.linalg.vector_norm(diff, dim=-1).mean(dim=-1)

    def _update_command(self):
        body_pos_w = self._raw_body_pos_w
        body_quat_w = self._raw_body_quat_w

        # Find anchor index by name if possible; otherwise fall back to first body.
        if self.body_names and self.cfg.anchor_body_name in self.body_names:
            anchor_index = self.body_names.index(self.cfg.anchor_body_name)
        else:
            anchor_index = 0

        anchor_pos_w = body_pos_w[anchor_index]

        # Compute body poses in the anchor frame.
        body_pos_rel = body_pos_w - anchor_pos_w.unsqueeze(0)
        body_quat_rel = body_quat_w  # keep orientations in world for now; reward will align in yaw frame

        # Set target anchor pose in world for each environment: align xy to env origin and fix target height.
        env_origins = self._env.scene.env_origins  # [num_envs, 3]
        target_height = (
            self.cfg.target_height
            if self.cfg.target_height is not None
            else float(self.data.get("root_state", self.data.get("default_root_state", [0.0, 0.0, 0.0]))[2])
        )

        anchor_pos_env = torch.stack(
            [env_origins[:, 0], env_origins[:, 1], torch.full_like(env_origins[:, 2], target_height)],
            dim=-1,
        )  # [num_envs, 3]

        # Anchor orientation is identity; relative positions are expressed in anchor frame.
        self._body_pos_w = anchor_pos_env.unsqueeze(1) + body_pos_rel.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self._body_quat_w = body_quat_rel.unsqueeze(0).repeat(self.num_envs, 1, 1)

        # Flattened command output in world frame.
        self._command = torch.cat(
            [
                self._body_pos_w.reshape(self.num_envs, -1),
                # self._body_quat_w.reshape(self.num_envs, -1),
            ],
            dim=-1,
        )

    @property
    def command(self) -> Tensor:
        """Flattened target body positions and orientations."""
        return self._command

    @property
    def body_pos_w(self) -> Tensor:
        """Target body positions in world frame, shape [num_envs, num_bodies, 3]."""
        return self._body_pos_w

    @property
    def body_quat_w(self) -> Tensor:
        """Target body orientations in world frame, shape [num_envs, num_bodies, 4]."""
        return self._body_quat_w

@configclass
class TargetPoseCommandCfg(CommandTermCfg):
    """Configuration for a fixed target pose command loaded from JSON."""
    class_type: type[TargetPoseCommand] = TargetPoseCommand
    resampling_time_range: tuple[float, float] = (1e5, 1e5)
    asset_name: str = "robot"
    initial_pose_path: str | None = MISSING
    anchor_body_name: str = "Trunk"
    target_height: float | None = None
    # Body name patterns to track. If None or [" .* "], all bodies are used.
    keybody_names: List[str] | None = None