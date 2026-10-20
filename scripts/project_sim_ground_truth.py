"""Project Isaac Sim cube poses into image boxes for supervised baseline labels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def project_ground_truth(
    run: Path,
    *,
    image_name: str = "world_camera.png",
    camera_name: str = "camera.json",
    ground_truth_name: str = "ground_truth.json",
    scene_state_name: str = "scene_state.json",
) -> dict[str, Any]:
    """Create normalized 2D cube boxes from the saved camera and scene poses."""
    image_path = run / image_name
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"could not read image: {image_path}")
    height, width = image.shape[:2]

    camera_path = run / camera_name
    ground_truth_path = run / ground_truth_name
    scene_state_path = run / scene_state_name
    camera = json.loads(camera_path.read_text())
    ground_truth = json.loads(ground_truth_path.read_text())
    scene_state = json.loads(scene_state_path.read_text())
    resolution = camera.get("resolution")
    if resolution != [width, height]:
        raise ValueError(
            f"{camera_path} resolution {resolution!r} does not match "
            f"image size {[width, height]}"
        )

    try:
        camera_to_world = np.asarray(camera["camera_to_world"], dtype=np.float64)
        world_to_camera = np.linalg.inv(camera_to_world)
        focal_length = float(camera["focal_length_mm"])
        horizontal_aperture = float(camera["horizontal_aperture_mm"])
        vertical_aperture = float(camera["vertical_aperture_mm"])
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        raise ValueError(f"{camera_path} has invalid camera calibration") from exc
    if (
        camera_to_world.shape != (4, 4)
        or not np.isfinite(camera_to_world).all()
        or not all(
            math.isfinite(value) and value > 0
            for value in (focal_length, horizontal_aperture, vertical_aperture)
        )
    ):
        raise ValueError(f"{camera_path} has invalid camera calibration")

    objects = ground_truth.get("objects")
    states = scene_state.get("objects")
    if not isinstance(objects, list) or not isinstance(states, dict):
        raise ValueError(
            f"{ground_truth_path} and {scene_state_path} need object collections"
        )

    focal_x = focal_length * width / horizontal_aperture
    focal_y = focal_length * height / vertical_aperture
    boxes = []
    for obj in objects:
        if not isinstance(obj, dict) or obj.get("category") != "cube":
            continue
        name = obj.get("name")
        colour = str(obj.get("colour", "")).lower()
        state = states.get(name)
        if not isinstance(name, str) or not isinstance(state, dict):
            raise ValueError(f"{ground_truth_path} cube has no saved scene pose")
        try:
            center = np.asarray(state["position"], dtype=np.float64)
            edge = float(obj["size"])
            yaw = math.radians(float(obj["yaw_deg"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{ground_truth_path} cube {name!r} has invalid pose") from exc
        if (
            center.shape != (3,)
            or not np.isfinite(center).all()
            or not math.isfinite(edge)
            or edge <= 0
            or not math.isfinite(yaw)
            or colour not in ("yellow", "blue", "red")
        ):
            raise ValueError(f"{ground_truth_path} cube {name!r} has invalid pose")

        cosine, sine = math.cos(yaw), math.sin(yaw)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        projected = []
        for x in (-edge / 2, edge / 2):
            for y in (-edge / 2, edge / 2):
                world_xy = center[:2] + rotation @ np.array([x, y])
                for z in (-edge / 2, edge / 2):
                    camera_xyz = np.array(
                        [world_xy[0], world_xy[1], center[2] + z, 1.0]
                    ) @ world_to_camera
                    depth = -camera_xyz[2]
                    if not math.isfinite(depth) or depth <= 0:
                        raise ValueError(f"cube {name!r} is behind the camera")
                    projected.append(
                        (
                            width / 2 + focal_x * camera_xyz[0] / depth,
                            height / 2 - focal_y * camera_xyz[1] / depth,
                        )
                    )

        points = np.asarray(projected)
        x_min = max(0.0, float(points[:, 0].min()))
        y_min = max(0.0, float(points[:, 1].min()))
        x_max = min(float(width), float(points[:, 0].max()))
        y_max = min(float(height), float(points[:, 1].max()))
        if x_min >= x_max or y_min >= y_max:
            continue
        boxes.append(
            {
                "name": name,
                "label": f"{colour} cube",
                "category": "cube",
                "colour": colour,
                "center": [
                    (x_min + x_max) / (2 * width),
                    (y_min + y_max) / (2 * height),
                ],
                "bbox": [
                    x_min / width,
                    y_min / height,
                    x_max / width,
                    y_max / height,
                ],
                "graspable": True,
            }
        )

    return {
        "coordinate_convention": "unit",
        "model": "Isaac Sim ground truth projection",
        "task": "cube_detection",
        "objects": boxes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", nargs="+", required=True, type=Path, help="captured view folders"
    )
    parser.add_argument(
        "--output-name",
        default="sim_ground_truth_detections.json",
        help="per-view output JSON filename",
    )
    args = parser.parse_args()

    for run in args.runs:
        output = run / args.output_name
        if output.exists():
            parser.error(f"refusing to overwrite existing labels: {output}")
    for run in args.runs:
        labels = project_ground_truth(run)
        output = run / args.output_name
        output.write_text(json.dumps(labels, indent=2) + "\n")
        print(f"{output}: projected {len(labels['objects'])} cube boxes")


if __name__ == "__main__":
    main()
