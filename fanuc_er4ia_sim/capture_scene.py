"""
Stage 1 of the perception pipeline (runs under Isaac's interpreter).

Builds the cell, renders the world camera, and writes out everything the later
stages need: the image, the camera's exact intrinsics/extrinsics, and the
ground-truth object poses so the VLM's guesses can actually be scored.

    $ISAAC/python.sh capture_scene.py --out run/
"""

import argparse, json, os, sys

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="run")
parser.add_argument("--camera", default="table", choices=["table", "side", "wrist"])
parser.add_argument(
    "--views",
    type=int,
    default=1,
    help="number of varied cube-layout images to save in separate subfolders",
)
parser.add_argument(
    "--validation-views",
    type=int,
    default=0,
    help="reserve this many views in a separate val/ folder",
)
parser.add_argument(
    "--seed", type=int, default=42, help="seed for reproducible capture variations"
)
args = parser.parse_args()
if args.views < 1 or not 0 <= args.validation_views < args.views:
    parser.error("--views must be positive and --validation-views in [0, views)")
# resolve before Isaac starts: Kit changes the working directory
args.out = os.path.abspath(args.out)

from isaacsim import SimulationApp

sim_app = SimulationApp({"headless": True})

import numpy as np
import cv2
from pxr import UsdGeom
import omni.usd
from isaacsim.core.api import World

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_config as C, cell

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
cell.build(stage)
# Keep the arrangement identical across capture and execution while shaders load.
from pxr import UsdPhysics

for prim in stage.Traverse():
    if prim.GetName().startswith(("SortCube", "SortVial", "SortCap")) and prim.HasAPI(
        UsdPhysics.RigidBodyAPI
    ):
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(True)

_CAMS = {
    "table": (cell.TABLE_CAM_PATH, C.TABLE_CAM),
    "side": (cell.SIDE_CAM_PATH, C.SIDE_CAM),
    "wrist": (cell.WRIST_CAM_PATH, C.WRIST_CAM),
}
cam_path, cfg = _CAMS[args.camera]
cfg = dict(cfg)
# Capture more top-face detail for 10 mm cubes; execution videos stay at 1080p.
if args.camera == "table" and C.SORT_TASK == "cube_stack":
    cfg["resolution"] = tuple(2 * v for v in cfg["resolution"])

from isaacsim.sensors.camera import Camera

cam = Camera(prim_path=cam_path, resolution=cfg["resolution"])
cell.enable_ccd(world)
world.reset()
cam.initialize()

# park the arm out of the overhead camera's view before the snapshot
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction

arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
arm.initialize()
names = list(arm.dof_names)
q = arm.get_joint_positions()
for j, deg in C.CAPTURE_POSE_DEG.items():
    q[names.index(j)] = np.deg2rad(deg)
for jn, v in cell.jaw_targets(C.JAW_OPEN).items():
    q[names.index(jn)] = v
arm.set_joint_positions(q)
arm.apply_action(ArticulationAction(joint_positions=q))

# MDL materials need frames to compile before capture
for _ in range(90):
    world.step(render=True)

from isaacsim.core.prims import SingleXFormPrim

camera_prim = stage.GetPrimAtPath(cam_path)
gcam = UsdGeom.Camera(camera_prim)
camera_xf = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(0)
W, H = cfg["resolution"]
camera_data = {
    "camera": args.camera,
    "resolution": [W, H],
    "focal_length_mm": float(gcam.GetFocalLengthAttr().Get()),
    "horizontal_aperture_mm": float(gcam.GetHorizontalApertureAttr().Get()),
    "vertical_aperture_mm": float(gcam.GetVerticalApertureAttr().Get()),
    # row-major 4x4, USD convention (row-vector times matrix)
    "camera_to_world": [
        [float(camera_xf[r][c]) for c in range(4)] for r in range(4)
    ],
    "table_top_z": float(C.TABLE_TOP_Z + C.SURFACE_DZ),
    "robot_base_xy": [0.0, 0.0],
}


def save_view(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rgba = cam.get_rgba()
    if rgba is None:
        raise RuntimeError("Camera returned no image; check the host GPU")
    rgb = rgba[:, :, :3].astype(np.uint8)
    img_path = os.path.join(out_dir, "world_camera.png")
    if not cv2.imwrite(img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write camera image: {img_path}")
    with open(os.path.join(out_dir, "camera.json"), "w") as f:
        json.dump(camera_data, f, indent=2)

    gt = {"objects": [], "bins": {}}
    scene_state = {"objects": {}}
    for prim in stage.Traverse():
        if prim.GetName().startswith(
            ("SortCube", "SortVial", "SortCap")
        ) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            pos, quat = SingleXFormPrim(str(prim.GetPath())).get_world_pose()
            scene_state["objects"][prim.GetName()] = {
                "position": pos.tolist(),
                "orientation": quat.tolist(),
            }

    for i, (x, y, colour) in enumerate(C.SORT_CUBES):
        gt["objects"].append(
            {
                "name": f"SortCube{i}",
                "category": "cube",
                "colour": colour,
                "xyz": [x, y, C.CUBE_EDGE / 2],
                "size": C.CUBE_EDGE,
                "yaw_deg": C.SORT_CUBE_YAWS_DEG[i],
            }
        )
    for i, (x, y) in enumerate(C.SORT_CAPS):
        gt["objects"].append(
            {
                "name": f"SortCap{i}",
                "category": "cap",
                "colour": "yellow",
                "xyz": [x, y, C.CAP_HEIGHT / 2],
                "orientation": C.CAP_ORIENTATIONS[i],
                "size": C.CAP_RADIUS * 2,
            }
        )
    for i, (x, y_ctr, yaw) in enumerate(C.SORT_VIALS):
        gt["objects"].append(
            {
                "name": f"SortVial{i}",
                "category": "vial",
                "colour": "clear",
                "xyz": [x, y_ctr, C.VIAL_RADIUS],
                "size": C.VIAL_RADIUS * 2,
                "length": C.VIAL_LENGTH,
                "yaw_deg": yaw,
            }
        )

    with open(os.path.join(out_dir, "ground_truth.json"), "w") as f:
        json.dump(gt, f, indent=2)
    with open(os.path.join(out_dir, "scene_state.json"), "w") as f:
        json.dump(scene_state, f, indent=2)
    print(f"[capture] image      {img_path}  ({W}x{H})")
    print(f"[capture] gt objects {len(gt['objects'])}")


if args.views == 1:
    save_view(args.out)
else:
    if os.path.exists(args.out) and os.listdir(args.out):
        raise FileExistsError(f"Capture output directory is not empty: {args.out}")
    os.makedirs(args.out, exist_ok=True)
    initial_cubes = tuple(C.SORT_CUBES)
    initial_yaws = tuple(C.SORT_CUBE_YAWS_DEG)
    for view_index in range(args.views):
        view_seed = args.seed + view_index
        C.SORT_CUBES, C.SORT_CUBE_YAWS_DEG = C.sample_capture_view(view_seed)
        for cube_index, (x, y, _) in enumerate(C.SORT_CUBES):
            angle = np.deg2rad(C.SORT_CUBE_YAWS_DEG[cube_index])
            orientation = np.array(
                [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]
            )
            SingleXFormPrim(f"/World/SortCube{cube_index}").set_world_pose(
                np.array(
                    [
                        x,
                        y,
                        C.TABLE_TOP_Z + C.SURFACE_DZ + C.CUBE_EDGE / 2,
                    ]
                ),
                orientation,
            )
        for _ in range(20):
            world.step(render=True)
        split = (
            "val" if view_index >= args.views - args.validation_views else "train"
        )
        split_index = (
            view_index - (args.views - args.validation_views)
            if split == "val"
            else view_index
        )
        view_dir = os.path.join(args.out, split, f"view_{split_index + 1:04d}")
        save_view(view_dir)
    C.SORT_CUBES = list(initial_cubes)
    C.SORT_CUBE_YAWS_DEG = list(initial_yaws)
sim_app.close()
