"""Core CSV→NPZ replay logic. Import only after ``AppLauncher`` has started the app."""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

from robotlib.soccerLab.hi import HI_CFG
from robotlib.soccerLab.pi_plus import PI_PLUS_CFG

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **_kwargs):
        return iterable


ROBOT_CONFIGS = {
    "hi": {
        "cfg": HI_CFG,
        "has_header": True,
        "dof_slice": (7, 30),
        "joint_names": [
            "l_hip_pitch_joint",
            "l_hip_roll_joint",
            "l_hip_thigh_joint",
            "l_hip_calf_joint",
            "l_ankle_pitch_joint",
            "l_ankle_roll_joint",
            "r_hip_pitch_joint",
            "r_hip_roll_joint",
            "r_hip_thigh_joint",
            "r_hip_calf_joint",
            "r_ankle_pitch_joint",
            "r_ankle_roll_joint",
            "waist_yaw_joint",
            "l_shoulder_pitch_joint",
            "l_shoulder_roll_joint",
            "l_upper_arm_joint",
            "l_elbow_joint",
            "l_wrist_joint",
            "r_shoulder_pitch_joint",
            "r_shoulder_roll_joint",
            "r_upper_arm_joint",
            "r_elbow_joint",
            "r_wrist_joint",
        ],
    },
    "pi_plus": {
        "cfg": PI_PLUS_CFG,
        "has_header": True,
        "dof_slice": None,
        "joint_names": [
            "l_hip_pitch_joint",
            "l_hip_roll_joint",
            "l_thigh_joint",
            "l_calf_joint",
            "l_ankle_pitch_joint",
            "l_ankle_roll_joint",
            "r_hip_pitch_joint",
            "r_hip_roll_joint",
            "r_thigh_joint",
            "r_calf_joint",
            "r_ankle_pitch_joint",
            "r_ankle_roll_joint",
            "l_shoulder_pitch_joint",
            "l_shoulder_roll_joint",
            "l_upper_arm_joint",
            "l_elbow_joint",
            "l_wrist_joint",
            "r_shoulder_pitch_joint",
            "r_shoulder_roll_joint",
            "r_upper_arm_joint",
            "r_elbow_joint",
            "r_wrist_joint",
        ],
    },
}


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = None


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
        robot_config: dict,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self.robot_config = robot_config
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        has_header = self.robot_config["has_header"]

        if self.frame_range is None:
            skip_rows = 1 if has_header else 0
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=",", skiprows=skip_rows))
        else:
            if has_header:
                skip_rows = self.frame_range[0]
            else:
                skip_rows = self.frame_range[0] - 1
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=skip_rows,
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                )
            )

        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]
        self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]

        dof_slice = self.robot_config["dof_slice"]
        if dof_slice is not None:
            self.motion_dof_poss_input = motion[:, dof_slice[0] : dof_slice[1]]
        else:
            self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _interpolate_motion(self):
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    robot_config: dict,
    simulation_app,
    *,
    input_file: str,
    output_name: str,
    input_fps: int,
    output_fps: int,
    frame_range: tuple[int, int] | None,
):
    """Replay one clip and write ``output_name``.npz."""
    joint_names = robot_config["joint_names"]

    motion = MotionLoader(
        motion_file=input_file,
        input_fps=input_fps,
        output_fps=output_fps,
        device=sim.device,
        frame_range=frame_range,
        robot_config=robot_config,
    )

    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    log = {
        "fps": [output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False

    while simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[k] = np.stack(log[k], axis=0)

            output_file = f"{output_name}.npz"
            np.savez(output_file, **log)
            print(f"[INFO]: Motion saved locally to: {output_file}")
            print("[INFO]: File saved, breaking loop...")
            break


def run_all(args_cli, simulation_app, jobs: list[tuple[str, str]] | None = None) -> None:
    """Build scene once, then replay each (input_csv, output_path_without_suffix).

    If ``jobs`` is None, uses ``[(args_cli.input_file, args_cli.output_name)]``.
    """
    if jobs is None:
        jobs = [(args_cli.input_file, args_cli.output_name)]

    robot_config = ROBOT_CONFIGS[args_cli.robot]
    print(f"[INFO]: Using robot configuration: {args_cli.robot}")

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot = robot_config["cfg"].replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print(f"[INFO]: Setup complete for {args_cli.robot} robot...")
    print(f"[INFO]: Using {len(robot_config['joint_names'])} joints")

    for input_file, output_name in tqdm(jobs, desc="csv_to_npz", unit="clip"):
        run_simulator(
            sim,
            scene,
            robot_config,
            simulation_app,
            input_file=input_file,
            output_name=output_name,
            input_fps=args_cli.input_fps,
            output_fps=args_cli.output_fps,
            frame_range=args_cli.frame_range,
        )
