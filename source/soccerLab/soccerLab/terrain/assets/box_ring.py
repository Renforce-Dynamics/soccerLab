from isaaclab.sim import UrdfFileCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import SceneEntityCfg

from robotlib import ROBOTLIB_ASSETLIB_DIR

BOXING_RING_URDF = "data/assets/assetslib/third_party/olympics/urdf/boxing_ring.urdf"

boxing_ring = UrdfFileCfg(
    asset_path=BOXING_RING_URDF,
    fix_base=True,
    make_instanceable=False,
    enable_self_collisions=False,
    merge_fixed_joints=True,
)

# Table
packing_table = AssetBaseCfg(
    prim_path="/World/envs/env_.*/PackingTable",
    init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ROBOTLIB_ASSETLIB_DIR}/Props/PackingTable/packing_table.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    ),
)

sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ROBOTLIB_ASSETLIB_DIR}/unitree/unitree_g1/urdf/g1_29dof.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ))