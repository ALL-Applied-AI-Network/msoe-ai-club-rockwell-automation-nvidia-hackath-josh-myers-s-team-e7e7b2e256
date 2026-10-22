import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.orchestrator import Problem3Orchestrator


class VisionPipelineOrchestratorTests(unittest.TestCase):
    def test_builds_five_class_dataset_from_grouped_simulator_captures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "generated"
            config_path = root / "problem.yaml"
            config_path.write_text(
                "data:\n"
                f"  directory: {dataset_root.as_posix()}\n"
                "  annotation_source: simulator_ground_truth\n"
                "  validation_split: 0.2\n"
                "project:\n  seed: 7\n"
                "components:\n  cameras:\n    views:\n"
                "      - id: overhead_left\n        role: observation\n"
                "        placement: overhead\n"
                "      - id: overhead_right\n        role: observation\n"
                "        placement: overhead\n"
                "      - id: wrist\n        role: close_view\n"
                "        placement: robot_wrist\n"
                "model:\n  class_names:\n"
                "    - yellow_cube\n    - blue_cube\n    - red_cube\n"
                "    - cap\n    - vial\n"
            )
            camera_to_world = np.eye(4)
            camera_to_world[3, 2] = 2
            run_paths = []
            for split in ("train", "val"):
                run = root / split / "view_0001"
                run.mkdir(parents=True)
                run_paths.append(run)
                self.assertTrue(
                    cv2.imwrite(
                        str(run / "world_camera.png"),
                        np.zeros((100, 100, 3), dtype=np.uint8),
                    )
                )
                (run / "camera.json").write_text(
                    json.dumps(
                        {
                            "camera": "table",
                            "resolution": [100, 100],
                            "focal_length_mm": 10,
                            "horizontal_aperture_mm": 10,
                            "vertical_aperture_mm": 10,
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
                                }
                            ]
                        }
                    )
                )
                (run / "scene_state.json").write_text(
                    json.dumps(
                        {
                            "objects": {
                                "SortCube0": {"position": [0, 0, 0]}
                            }
                        }
                    )
                )

            orchestrator = Problem3Orchestrator(config_path)
            output = orchestrator.run_data_generation(
                [str(path) for path in run_paths]
            )

            self.assertEqual(output, dataset_root / "dataset")
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(
                manifest["class_names"],
                ["yellow_cube", "blue_cube", "red_cube", "cap", "vial"],
            )
            self.assertEqual(len(manifest["runs"]["train"]), 1)
            self.assertEqual(len(manifest["runs"]["val"]), 1)
            self.assertEqual(manifest["boxes_by_class"]["train"]["blue_cube"], 1)
            self.assertEqual(manifest["boxes_by_class"]["val"]["blue_cube"], 1)
            self.assertTrue((root / "train" / "view_0001" / "sim_ground_truth_all_detections.json").is_file())


if __name__ == "__main__":
    unittest.main()
