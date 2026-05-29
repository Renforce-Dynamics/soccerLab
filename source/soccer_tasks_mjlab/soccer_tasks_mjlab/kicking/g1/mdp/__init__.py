"""MDP terms for G1 kicking task (mjlab backend)."""

from mjlab.envs.mdp.observations import (
    base_lin_vel,
    base_ang_vel,
    projected_gravity,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
)
from mjlab.envs.mdp.rewards import (
    action_rate_l2,
    joint_pos_limits,
)
from mjlab.envs.mdp.terminations import (
    time_out,
    root_height_below_minimum,
    bad_orientation,
)
from mjlab.envs.mdp.events import (
    reset_root_state_uniform,
    reset_joints_by_offset,
)
from mjlab.envs.mdp.actions import (
    JointPositionActionCfg,
)

from .rewards import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .commands import *  # noqa: F401, F403
from .events import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
from .curriculums import *  # noqa: F401, F403
