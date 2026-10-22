import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.core.DigitalTwinDataCollector import DigitalTwinDataCollector
from src.core.models import SideCameraFrame


class DigitalTwinDataCollectorTests(unittest.TestCase):
    def test_maps_side_capture_to_side_camera_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "problem.yaml"
            config.write_text(
                "components:\n  cameras:\n    views:\n"
                "      - id: overhead\n        role: observation\n"
                "        placement: overhead\n"
                "      - id: side\n        role: observation\n"
                "        placement: side\n"
                "      - id: wrist\n        role: close_view\n"
                "        placement: robot_wrist\n"
            )
            run = root / "unseen_side" / "view_0001"
            run.mkdir(parents=True)
            self.assertTrue(
                cv2.imwrite(
                    str(run / "world_camera.png"),
                    np.zeros((10, 10, 3), dtype=np.uint8),
                )
            )
            (run / "camera.json").write_text(json.dumps({"camera": "side"}))

            collector = DigitalTwinDataCollector.from_config(None, config)
            frames = collector.collect([str(run)], root / "manifest")

            self.assertEqual(len(frames), 1)
            self.assertIsInstance(frames[0], SideCameraFrame)
            self.assertEqual(frames[0].camera_id, "side")
            self.assertEqual(frames[0].metadata["split_group"], "unseen_side")
            self.assertTrue((root / "manifest" / "frame_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
