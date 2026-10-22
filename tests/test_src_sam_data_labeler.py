import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.core.SAMDataLabeler import SAMDataLabeler
from src.core.models import CameraFrame


class FakeSamProcessor:
    def set_image(self, image):
        self.width, self.height = image.size
        return {}

    def set_text_prompt(self, *, prompt, state):
        mask = np.zeros((1, self.height, self.width), dtype=np.uint8)
        mask[0, 2:7, 3:9] = 1
        return {"masks": mask, "scores": np.array([0.9])}


class SamDataLabelerTests(unittest.TestCase):
    def test_labels_and_exports_reusable_masks_for_mixed_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = []
            for split in ("train_table", "unseen_side"):
                image_path = root / split / "world_camera.png"
                image_path.parent.mkdir()
                self.assertTrue(
                    cv2.imwrite(
                        str(image_path),
                        np.zeros((10, 12, 3), dtype=np.uint8),
                    )
                )
                frames.append(
                    CameraFrame(
                        camera_id="overhead",
                        image_path=image_path,
                        timestamp_ns=1,
                        calibration_id="camera-v1",
                        role="observation",
                        placement="overhead",
                        metadata={"split_group": split},
                    )
                )

            labeler = SAMDataLabeler(FakeSamProcessor(), ("blue_cube", "cap"))
            labels = labeler.label(frames)
            self.assertEqual(
                [[annotation.class_name for annotation in frame] for frame in labels],
                [["blue_cube", "cap"], ["blue_cube", "cap"]],
            )
            self.assertTrue(
                all(
                    annotation.segmentation_mask is not None
                    for frame_labels in labels
                    for annotation in frame_labels
                )
            )

            dataset = labeler.export_yolo(
                frames,
                labels,
                root / "dataset",
                validation_group="unseen_side",
            )

            self.assertTrue((dataset / "mask_index.json").is_file())
            self.assertEqual(
                len(list((dataset / "masks").rglob("*.png"))), 4
            )
            self.assertEqual(
                len(json.loads((dataset / "mask_index.json").read_text())),
                4,
            )


if __name__ == "__main__":
    unittest.main()
