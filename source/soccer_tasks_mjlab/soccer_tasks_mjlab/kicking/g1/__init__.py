"""Register G1 kicking AMP tasks (mjlab backend).

Tasks use runner_cls=None because the AMP wrapper (beyondAMP.mjlab.rsl_rl.AMPEnvWrapper)
replaces mjlab's stock RslRlVecEnvWrapper. Train via scripts/factoryMjlab/train.py.

Registered tasks:
  - Soccer-Mjlab-Kick-AMP-G1          (Stage A, auto-promotes to B/C)
  - Soccer-Mjlab-Kick-Bootstrap-G1    (Bootstrap A0->A1)
  - Soccer-Mjlab-Kick-StageB-G1       (Starts at Stage B)
  - Soccer-Mjlab-Kick-StageC-G1       (Force-locked to Stage C)
"""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import (
    g1_kick_basic_env_cfg,
    g1_kick_bootstrap_env_cfg,
    g1_kick_stage_b_env_cfg,
    g1_kick_stage_c_env_cfg,
)
from .agents.amp_ppo_cfg import (
    g1_kick_amp_runner_cfg,
    g1_kick_bootstrap_amp_runner_cfg,
    g1_kick_stage_b_amp_runner_cfg,
    g1_kick_stage_c_amp_runner_cfg,
)


register_mjlab_task(
    task_id="Soccer-Mjlab-Kick-AMP-G1",
    env_cfg=g1_kick_basic_env_cfg(),
    play_env_cfg=g1_kick_basic_env_cfg(play=True),
    rl_cfg=g1_kick_amp_runner_cfg(),
    runner_cls=None,
)

register_mjlab_task(
    task_id="Soccer-Mjlab-Kick-Bootstrap-G1",
    env_cfg=g1_kick_bootstrap_env_cfg(),
    play_env_cfg=g1_kick_bootstrap_env_cfg(play=True),
    rl_cfg=g1_kick_bootstrap_amp_runner_cfg(),
    runner_cls=None,
)

register_mjlab_task(
    task_id="Soccer-Mjlab-Kick-StageB-G1",
    env_cfg=g1_kick_stage_b_env_cfg(),
    play_env_cfg=g1_kick_stage_b_env_cfg(play=True),
    rl_cfg=g1_kick_stage_b_amp_runner_cfg(),
    runner_cls=None,
)

register_mjlab_task(
    task_id="Soccer-Mjlab-Kick-StageC-G1",
    env_cfg=g1_kick_stage_c_env_cfg(),
    play_env_cfg=g1_kick_stage_c_env_cfg(play=True),
    rl_cfg=g1_kick_stage_c_amp_runner_cfg(),
    runner_cls=None,
)
