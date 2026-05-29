"""Register T1 dribbling task (mjlab backend, pure PPO).

Task: T1 robot walks towards and dribbles a soccer ball.
No AMP -- uses standard OnPolicyRunner.
Task ID: Soccer-Mjlab-Dribble-Flat-T1
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .agents.ppo_cfg import dribbling_ppo_runner_cfg
from .dribbling_env_cfg import make_dribbling_env_cfg

register_mjlab_task(
    task_id="Soccer-Mjlab-Dribble-Flat-T1",
    env_cfg=make_dribbling_env_cfg(),
    play_env_cfg=make_dribbling_env_cfg(play=True),
    rl_cfg=dribbling_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
