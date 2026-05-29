"""Train a soccer task on the mjlab backend.

Thin wrapper around mjlab's built-in training pipeline that additionally
imports ``soccer_tasks_mjlab`` to register Soccer-Mjlab-* tasks.

Example::

    python scripts/factoryMjlab/train.py Soccer-Mjlab-Dribble-Flat-T1 \\
        --headless --num_envs 64
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Soccer-Mjlab task")
    parser.add_argument("task_id", type=str, help="Registered task ID")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--log_root", type=str, default="logs/rsl_rl")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    # Register soccer tasks
    import soccer_tasks_mjlab  # noqa: F401
    import mjlab.tasks  # noqa: F401

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.gpu import select_gpus
    from mjlab.utils.os import dump_yaml, get_checkpoint_path
    from mjlab.utils.torch import configure_torch_backends

    # GPU setup
    selected_gpus, _ = select_gpus(args.gpu_ids)
    if selected_gpus is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
        os.environ["MUJOCO_GL"] = "egl"
        device = "cuda:0"

    configure_torch_backends()

    # Load configs
    env_cfg = load_env_cfg(args.task_id)
    rl_cfg = load_rl_cfg(args.task_id)

    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs

    if args.max_iterations is not None:
        rl_cfg.max_iterations = args.max_iterations

    # Logging
    log_root = (Path(args.log_root) / rl_cfg.experiment_name).resolve()
    log_dir = log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print(f"[INFO] Training: task={args.task_id} device={device}")
    print(f"[INFO] num_envs={env_cfg.scene.num_envs}")
    print(f"[INFO] Logging to: {log_dir}")

    # Detect AMP task by checking if rl_cfg is an AMPRunnerCfg
    is_amp_task = False
    try:
        from beyondAMP.mjlab.rsl_rl import AMPRunnerCfg
        is_amp_task = isinstance(rl_cfg, AMPRunnerCfg)
    except ImportError:
        pass

    # Create environment
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    if is_amp_task:
        # AMP path: use AMPEnvWrapper + AMPOnPolicyRunner
        from beyondAMP.mjlab.rsl_rl import AMPEnvWrapper
        from rsl_rl_amp.runners.amp_on_policy_runner import AMPOnPolicyRunner

        print("[INFO] AMP task detected — using AMPEnvWrapper + AMPOnPolicyRunner")
        env = AMPEnvWrapper(
            env,
            clip_actions=rl_cfg.clip_actions,
            motion_dataset=rl_cfg.amp_data,
        )

        # Dump configs
        dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
        dump_yaml(log_dir / "params" / "agent.yaml", asdict(rl_cfg))

        runner = AMPOnPolicyRunner(env, asdict(rl_cfg), log_dir=str(log_dir), device=device)
    else:
        # Standard path: RslRlVecEnvWrapper + stock runner
        env = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

        # Dump configs
        dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
        dump_yaml(log_dir / "params" / "agent.yaml", asdict(rl_cfg))

        runner_cls = load_runner_cls(args.task_id)
        if runner_cls is None:
            runner_cls = MjlabOnPolicyRunner

        runner = runner_cls(env, asdict(rl_cfg), str(log_dir), device)

    # Resume from checkpoint
    if args.resume and args.checkpoint:
        resume_path = get_checkpoint_path(log_root, None, args.checkpoint)
        print(f"[INFO] Loading checkpoint: {resume_path}")
        runner.load(str(resume_path))

    # Train
    runner.learn(
        num_learning_iterations=rl_cfg.max_iterations,
        init_at_random_ep_len=True,
    )

    env.close()
    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()
