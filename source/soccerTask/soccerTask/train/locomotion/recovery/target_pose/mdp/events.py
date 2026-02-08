# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Literal

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg


class disable_joints(ManagerTermBase):
    """Disable the joint for the given asset for a given duration.

    Note: This event requires the 'pre_sim_step' mode in the physics stepping loop:
    ```python

    if "pre_sim_step" in self.event_manager.available_modes:
        self.event_manager.apply(mode="pre_sim_step", dt=self.step_dt)
    ```
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,  # noqa: ARG002
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> None:
        """Disable the joints for the given asset for a given duration.

        This allows the robot to not take any actions during the rest phase.
        It should be called before the simulation step.
        Reads rest_duration_s from env.cfg.rest_duration_s.

        Args:
            env: The environment.
            env_ids: The environment ids.
            asset_cfg: The asset configuration.

        """
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        
        # Read rest_duration_s from env.cfg
        rest_duration_s = env.cfg.rest_duration_s

        # check which environments are in the rest phase
        env_in_rest_phase = env.episode_length_buf < int(rest_duration_s / env.step_dt)
        rest_env_ids = env_in_rest_phase.nonzero().flatten()

        # disable the joints
        asset._joint_effort_target_sim[rest_env_ids, :] = 0.0  # type: ignore

        # set the joint efforts to 0
        asset.root_physx_view.set_dof_actuation_forces(asset._joint_effort_target_sim, rest_env_ids)  # type: ignore
