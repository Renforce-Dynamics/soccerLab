import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Dump initial pose (default root + zero joints) to JSON.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    "--out",
    type=str,
    default="initial_pose.json",
    help="Output JSON file path (default: initial_pose.json)",
)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
from pathlib import Path
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from robotlib.soccerLab import booster_t1


@configclass
class InitialPoseSceneCfg(InteractiveSceneCfg):
    """Minimal scene with a single T1 robot on a ground plane."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation: use T1 robot from soccerLab
    robot: ArticulationCfg = booster_t1.T1_DELAYED_DC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def dump_initial_pose(sim: SimulationContext, scene: InteractiveScene, out_path: Path) -> None:
    """Set root to default_root_state, zero all joints, then dump full state to JSON."""
    # extract robot
    robot: Articulation = scene["robot"]

    # physics dt
    sim_dt = sim.get_physics_dt()

    # 1) set root to default_root_state
    root_states = robot.data.default_root_state.clone()
    robot.write_root_state_to_sim(root_states)

    # 2) zero all joints (position & velocity)
    joint_pos = torch.zeros_like(robot.data.default_joint_pos)
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    # 3) push state into sim and run one step so that derived quantities (body pos/vel)更新
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_dt)

    # 4) read back full information from the first (and only) env
    env_id = 0

    data = {
        "env_id": int(env_id),
        # root state: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        "root_state": robot.data.root_state_w[env_id].cpu().tolist(),
        "default_root_state": robot.data.default_root_state[env_id].cpu().tolist(),
        # joint-level
        "joint_pos": robot.data.joint_pos[env_id].cpu().tolist(),
        "joint_vel": robot.data.joint_vel[env_id].cpu().tolist(),
        "default_joint_pos": robot.data.default_joint_pos[env_id].cpu().tolist(),
        "default_joint_vel": robot.data.default_joint_vel[env_id].cpu().tolist(),
        "joint_names": list(robot.data.joint_names),
        # body-level (world frame)
        "body_pos_w": robot.data.body_pos_w[env_id].cpu().tolist(),
        "body_quat_w": robot.data.body_quat_w[env_id].cpu().tolist(),
        "body_lin_vel_w": robot.data.body_lin_vel_w[env_id].cpu().tolist(),
        "body_ang_vel_w": robot.data.body_ang_vel_w[env_id].cpu().tolist(),
        "body_names": list(robot.data.body_names),
    }

    # 5) save as a JSON list
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump([data], f, indent=2)

    print(f"[dump_initial_pose_json] Saved initial pose for T1 robot to: {out_path}")


def main():
    # simulation config
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    # use a reasonably small dt
    sim_cfg.dt = 0.01
    sim = SimulationContext(sim_cfg)

    # single env with a T1 robot
    scene_cfg = InitialPoseSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    out_path = Path(args_cli.out)
    dump_initial_pose(sim, scene, out_path)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

