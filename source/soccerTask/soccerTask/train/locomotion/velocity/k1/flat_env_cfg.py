from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from ..velocity_env_cfg import RobotEnvCfg as BaseRobotEnvCfg
from ..velocity_env_cfg import RobotPlayEnvCfg as BaseRobotPlayEnvCfg
from robotlib.beyondMimic.robots.k1 import K1_CFG, K1_ACTION_SCALE
from locomotion_rl_lab.locomotion import mdp

@configclass
class RobotEnvCfg(BaseRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Override robot
        self.scene.robot = K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # Override action scale
        self.actions.JointPositionAction.scale = K1_ACTION_SCALE
        
        # Override sensors/events that use "torso_link" -> "Trunk" for K1
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Trunk"
        
        self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names="Trunk")
        self.events.base_external_force_torque.params["asset_cfg"] = SceneEntityCfg("robot", body_names="Trunk")

        # --- Rewards Overrides for K1 Joint/Body Names ---
        
        # Arms: K1 has Shoulder and Elbow, no Wrist
        # Names: ALeft_Shoulder_Pitch, Left_Shoulder_Roll, etc. Regex ".*Shoulder.*", ".*Elbow.*" should work.
        self.rewards.joint_deviation_arms.params["asset_cfg"].joint_names = [".*Shoulder.*", ".*Elbow.*"]
        
        # Legs: Hip_Roll, Hip_Yaw
        self.rewards.joint_deviation_legs.params["asset_cfg"].joint_names = [".*Hip_Roll", ".*Hip_Yaw"]
        
        # Waist: K1 has no waist joints in the list provided suitable for this reward?
        # Available: ...Head... but no Waist.
        self.rewards.joint_deviation_waists = None
        
        # Feet / Gait / Slide / Clearance
        # K1 feet/ankles: Left_Ankle_Pitch, Left_Ankle_Roll, etc.
        # Body names might be different from joint names.
        # Error available strings for bodies were: 'Trunk', 'Left_Hip_Pitch', ... 'left_foot_link', 'right_foot_link'
        # It seems 'left_foot_link' corresponds to the foot.
        # The 'gait', 'feet_slide', 'feet_clearance' rewards usually track the foot or ankle.
        # G1 used ".*ankle_roll.*".
        # K1 has "Left_Ankle_Roll" joint.
        # But these rewards often use body names or contact forces on bodies.
        # K1 body names: 'left_foot_link', 'right_foot_link'.
        
        self.rewards.gait.params["sensor_cfg"].body_names = [".*foot_link"]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [".*foot_link"]
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [".*foot_link"]
        self.rewards.feet_clearance.params["asset_cfg"].body_names = [".*foot_link"]
        
        # Fix: Exclude feet from undesired contacts (Aligned with booster_train reference)
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [r"^(?!left_foot_link$)(?!right_foot_link$).+$"]
        
        # Fix: Lower target height for K1 (0.78 -> 0.60)
        self.rewards.base_height.params["target_height"] = 0.60

        # [NEW] Increase gait frequency (reduce period) for K1
        # G1 uses 0.8s (1.25Hz). K1 should be much faster.
        # Setting period to 0.5s (2.0Hz) as a starting point.
        self.rewards.gait.params["period"] = 0.5


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
