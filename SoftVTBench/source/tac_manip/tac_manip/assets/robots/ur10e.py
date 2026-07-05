"""Configuration for the Universal Robots.

The following configuration parameters are available:

* :obj:`UR10E_UMIXEN_HIGH_PD_CFG`: The UR10E arm with UMixense tactile gripper and stiffer PD control.

Reference: https://github.com/ros-industrial/universal_robot
"""

from isaaclab.actuators import ImplicitActuatorCfg
try:
    from isaaclab_assets.robots.universal_robots import UR10e_CFG
except ImportError:
    from isaaclab_assets.robots.universal_robots import UR10_CFG as UR10e_CFG
from tac_manip.assets import ASSETS_DATA_DIR

UR10e_UMIXENSE_CFG = UR10e_CFG.copy()
UR10e_UMIXENSE_CFG.spawn.usd_path = f"{ASSETS_DATA_DIR}/Robots/ur10e/ur10e.usd"
UR10e_UMIXENSE_CFG.spawn.variants = {"Gripper": "umi_xense"}
UR10e_UMIXENSE_CFG.spawn.rigid_props.disable_gravity = True
UR10e_UMIXENSE_CFG.init_state.joint_pos["joint[1,2]"] = -0.03

if {"shoulder", "elbow", "wrist"}.issubset(UR10e_UMIXENSE_CFG.actuators):
    UR10e_UMIXENSE_CFG.actuators["shoulder"].stiffness = 8000.0
    UR10e_UMIXENSE_CFG.actuators["shoulder"].damping = 80.0
    UR10e_UMIXENSE_CFG.actuators["elbow"].stiffness = 4000.0
    UR10e_UMIXENSE_CFG.actuators["elbow"].damping = 40.0
    UR10e_UMIXENSE_CFG.actuators["wrist"].stiffness = 2000.0
    UR10e_UMIXENSE_CFG.actuators["wrist"].damping = 20.0
elif "arm" in UR10e_UMIXENSE_CFG.actuators:
    UR10e_UMIXENSE_CFG.actuators["arm"].stiffness = 8000.0
    UR10e_UMIXENSE_CFG.actuators["arm"].damping = 80.0


# the actuator joints for gripper
UR10e_UMIXENSE_CFG.actuators["gripper"] = ImplicitActuatorCfg(
    joint_names_expr=["joint[1,2]"],
    effort_limit_sim=2.0,
    velocity_limit_sim=1.0,
    stiffness=100.0,
    damping=40.0,
    friction=0.0,
    armature=0.0,
)

"""Configuration of UR-10E arm with UMIXense tactile gripper."""
