from pxr import UsdGeom
import numpy as np
import omni
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.viewports import create_viewport_for_camera
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg


def set_camera_intrinsics(camera, resolution, intrinsic_matrix, f_stop, focus_distance):
    width, height = resolution[0], resolution[1]
    pixel_size = 3 * 1e-3

    ((fx, _, cx), (_, fy, cy), (_, _, _)) = intrinsic_matrix

    horizontal_aperture = pixel_size * width  # The aperture size in mm
    vertical_aperture = pixel_size * height
    focal_length_x = fx * pixel_size
    focal_length_y = fy * pixel_size
    focal_length = (focal_length_x + focal_length_y) / 2  # The focal length in mm

    # Set the camera parameters, note the unit conversion between Isaac Sim sensor and Kit
    camera.GetAttribute("focalLength").Set(focal_length)  # Convert from mm to cm (or 1/10th of a world unit)
    camera.GetAttribute("focusDistance").Set(focus_distance)  # The focus distance in meters
    camera.GetAttribute("fStop").Set(f_stop * 100.0)  # Convert the f-stop to Isaac Sim units
    camera.GetAttribute("horizontalAperture").Set(
        horizontal_aperture
    )  # Convert from mm to cm (or 1/10th of a world unit)
    camera.GetAttribute("verticalAperture").Set(vertical_aperture)
    near, far = 0.05, 1.0e5
    camera.GetAttribute("clippingRange").Set((near, far))


def create_camera_and_viewport(
    viewport_name: str,
    camera_prim: str,
    camera_prim_path: str,
    width: int = 1280,
    height: int = 720,
    focus_distance: float = 1.0,
    f_stop: float = 2.0,
    resolution: tuple = (1280, 720),
    camera_position: tuple = (0, 0, 1.50),
    camera_rotation: tuple = (0, 0, 90),
):
    """Create a new camera and viewport.

    Args:
        viewport_name: Name of the viewport
        camera_prim: Path to the camera prim
        camera_prim_path: Path to the camera prim
        width: Width of the viewport
        height: Height of the viewport
    """

    intrinsic_matrix = np.array(
        [[910.188232421875, 0.0, 643.2019653320312], [0.0, 910.2255249023438, 372.17559814453125], [0.0, 0.0, 1.0]]
    )

    set_camera_intrinsics(camera_prim, resolution, intrinsic_matrix, f_stop, focus_distance)


    UsdGeom.Xformable(camera_prim).AddTranslateOp().Set(camera_position)
    UsdGeom.Xformable(camera_prim).AddRotateXYZOp().Set(camera_rotation)

    print(f"Camera setup complete at {camera_prim} with position {camera_position} and rotation {camera_rotation}.")

    # Use create_viewport_for_camera function to create and bind viewport
    create_viewport_for_camera(
        viewport_name=viewport_name, camera_prim_path=camera_prim_path, width=width, height=height
    )

def create_new_viewports(cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg):

    """Create new viewports for teleoperation."""

    # default positions and rotations for teleoperation
    positions = [
        (0.58170437617265, 0.006142091410758917, 0.09775382394611476),
        (-0.09992210175134375, -0.2894954614604972, 0.06494836539123099),
        (-0.06257411441803443, 0.23754536751257382, 1.236703457750541),  # overlook camera
    ]

    rotations = [(83.73338, -2.2263883e-14, 89.07779), (80.931175, -9.9392335e-17, -0.91692996), (0, 0, 90)]
    
    if hasattr(cfg, "teleop_camera_positions") and cfg.teleop_camera_positions is not None:
        positions = cfg.teleop_camera_positions
    if hasattr(cfg, "teleop_camera_rotations") and cfg.teleop_camera_rotations is not None:
        rotations = cfg.teleop_camera_rotations

    stage = get_current_stage()
    print(f"Current windows: {omni.ui.Workspace.get_windows()}")
    # create new viewports and dock them right to the Main_Viewport
    main_viewport = omni.ui.Workspace.get_window("Viewport")
    print(f"main_viewport: {main_viewport}")
    num_viewport = 3

    for i in range(num_viewport):
        # create new viewport
        viewport_name = "Viewport_" + str(i + 2)
        camera_prim_path = "/World/Camera_" + str(i + 2)
        camera_prim = stage.DefinePrim(camera_prim_path, "Camera")

        create_camera_and_viewport(viewport_name, camera_prim, camera_prim_path, camera_position=positions[i], camera_rotation=rotations[i])
        print(f"## created new Viewport_{i}")