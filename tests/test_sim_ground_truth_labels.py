"""Tests for projecting simulator cube poses into image-coordinate boxes."""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.project_sim_ground_truth import project_ground_truth


class SimGroundTruthLabelTests(unittest.TestCase):
    def test_projects_cube_corners_using_camera_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.assertTrue(
                cv2.imwrite(
                    str(run / "world_camera.png"),
                    np.zeros((100, 100, 3), dtype=np.uint8),
                )
            )
            camera_to_world = np.eye(4)
            camera_to_world[3, 2] = 2.0
            (run / "camera.json").write_text(
                json.dumps(
                    {
                        "resolution": [100, 100],
                        "focal_length_mm": 10.0,
                        "horizontal_aperture_mm": 10.0,
                        "vertical_aperture_mm": 10.0,
                        "camera_to_world": camera_to_world.tolist(),
                    }
                )
            )
            (run / "ground_truth.json").write_text(
                json.dumps(
                    {
                        "objects": [
                            {
                                "name": "SortCube0",
                                "category": "cube",
                                "colour": "blue",
                                "size": 0.2,
                                "yaw_deg": 0,
                            },
                            {
                                "name": "SortCap0",
                                "category": "cap",
                                "colour": "yellow",
                                "size": 0.04,
                                "height": 0.02,
                            },
                        ]
                    }
                )
            )
            (run / "scene_state.json").write_text(
                json.dumps(
                    {
                        "objects": {
                            "SortCube0": {
                                "position": [0, 0, 0],
                                "orientation": [1, 0, 0, 0],
                            },
                            "SortCap0": {"position": [0.2, 0.2, 0]},
                        }
                    }
                )
            )

            labels = project_ground_truth(run)

            self.assertEqual(labels["coordinate_convention"], "unit")
            self.assertEqual(labels["model"], "Isaac Sim ground truth projection")
            self.assertEqual(
                [obj["class_name"] for obj in labels["objects"]],
                ["blue_cube", "cap"],
            )
            box = labels["objects"][0]["bbox"]
            self.assertAlmostEqual(box[0], 0.447368, places=5)
            self.assertAlmostEqual(box[1], 0.447368, places=5)
            self.assertAlmostEqual(box[2], 0.552632, places=5)
            self.assertAlmostEqual(box[3], 0.552632, places=5)

    def test_rejects_mismatched_image_and_camera_resolutions(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.assertTrue(
                cv2.imwrite(
                    str(run / "world_camera.png"),
                    np.zeros((100, 100, 3), dtype=np.uint8),
                )
            )
            (run / "camera.json").write_text(json.dumps({"resolution": [200, 100]}))
            (run / "ground_truth.json").write_text(json.dumps({"objects": []}))
            (run / "scene_state.json").write_text(json.dumps({"objects": {}}))

            with self.assertRaisesRegex(ValueError, "does not match"):
                project_ground_truth(run)

    def test_projects_cap_and_vial_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.assertTrue(
                cv2.imwrite(
                    str(run / "world_camera.png"),
                    np.zeros((100, 100, 3), dtype=np.uint8),
                )
            )
            camera_to_world = np.eye(4)
            camera_to_world[3, 2] = 2.0
            (run / "camera.json").write_text(
                json.dumps(
                    {
                        "resolution": [100, 100],
                        "focal_length_mm": 10.0,
                        "horizontal_aperture_mm": 10.0,
                        "vertical_aperture_mm": 10.0,
                        "camera_to_world": camera_to_world.tolist(),
                    }
                )
            )
            (run / "ground_truth.json").write_text(
                json.dumps(
                    {
                        "objects": [
                            {
                                "name": "SortCap0",
                                "category": "cap",
                                "size": 0.02,
                                "height": 0.01,
                            },
                            {
                                "name": "SortVial0",
                                "category": "vial",
                                "size": 0.02,
                                "length": 0.04,
                                "yaw_deg": 30,
                            },
                        ]
                    }
                )
            )
            (run / "scene_state.json").write_text(
                json.dumps(
                    {
                        "objects": {
                            "SortCap0": {"position": [0.0, 0.0, 0.0]},
                            "SortVial0": {"position": [0.1, 0.1, 0.0]},
                        }
                    }
                )
            )

            labels = project_ground_truth(run)

            self.assertEqual(
                [obj["class_name"] for obj in labels["objects"]], ["cap", "vial"]
            )
            self.assertTrue(
                all(
                    0 <= coordinate <= 1
                    for obj in labels["objects"]
                    for coordinate in obj["bbox"]
                )
            )


if __name__ == "__main__":
    unittest.main()
