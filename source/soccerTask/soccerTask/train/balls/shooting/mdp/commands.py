from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    """Extension of :class:`~isaaclab.envs.mdp.commands.UniformVelocityCommandCfg`.

    Adds an extra range container (``limit_ranges``) that can be used to widen the
    command ranges for play/eval or curriculum scheduling.
    """

    # Upper-bound range limits (e.g. used in play/eval or curriculum).
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
