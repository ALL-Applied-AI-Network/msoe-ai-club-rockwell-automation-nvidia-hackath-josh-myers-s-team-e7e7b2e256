"""Tests for converting SAM 3 masks to YOLO-exportable cube labels."""

import unittest
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from scripts.label_with_sam3 import detections_from_masks, label_run
from scripts.build_yolo_dataset import build_dataset


class Sam3LabelingTests(unittest.TestCase):
    def test_mask_bounds_are_normalized_and_keep_mask_identity(self):
        masks = np.zeros((2, 1, 10, 20), dtype=np.uint8)
        masks[0, 0, 2:5, 4:9] = 1
        masks[1, 0, 6:9, 12:16] = 1

        detections = detections_from_masks(
            masks,
            np.array([0.9, 0.8]),
            colour="blue",
            width=20,
            height=10,
            threshold=0.5,
        )

        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0]["bbox"], [0.2, 0.2, 0.45, 0.5])
        self.assertEqual(detections[0]["center"], [0.325, 0.35])
        self.assertEqual(detections[0]["mask_area_px"], 15)
        self.assertEqual(detections[1]["_mask_index"], 1)
        self.assertEqual(detections[0]["colour"], "blue")

    def test_filters_low_score_and_empty_masks(self):
        masks = np.zeros((2, 8, 8), dtype=np.uint8)
        masks[0, 1:3, 1:3] = 1

        detections = detections_from_masks(
            masks,
            [0.4, 0.95],
            colour="red",
            width=8,
            height=8,
            threshold=0.5,
        )

        self.assertEqual(detections, [])

    def test_rejects_mask_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "dimensions do not match"):
            detections_from_masks(
                np.zeros((1, 4, 4), dtype=np.uint8),
                [0.9],
                colour="yellow",
                width=8,
                height=8,
                threshold=0.5,
            )

    def test_rejects_mismatched_scores(self):
        with self.assertRaisesRegex(ValueError, "different number"):
            detections_from_masks(
                np.zeros((1, 8, 8), dtype=np.uint8),
                [],
                colour="yellow",
                width=8,
                height=8,
                threshold=0.5,
            )

    def test_outputs_feed_yolo_dataset_exporter(self):
        class FakeProcessor:
            def set_image(self, image):
                self.width, self.height = image.size
                return {}

            def set_text_prompt(self, *, prompt, state):
                x = {"yellow cube": 1, "blue cube": 8, "red cube": 15}[prompt]
                mask = np.zeros((1, self.height, self.width), dtype=np.uint8)
                mask[0, 2:5, x : x + 3] = 1
                return {"masks": mask, "scores": np.array([0.95])}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "train" / "view_0001"
            run.mkdir(parents=True)
            image = np.zeros((10, 24, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(run / "world_camera.png"), image))

            count = label_run(
                run,
                FakeProcessor(),
                image_name="world_camera.png",
                output_name="sam3_detections.json",
                preview_name="sam3_preview.png",
                threshold=0.5,
                overwrite=False,
            )

            detections = json.loads((run / "sam3_detections.json").read_text())
            self.assertEqual(count, 3)
            self.assertEqual(len(detections["objects"]), 3)
            self.assertTrue((run / "sam3_preview.png").is_file())

            manifest = build_dataset(
                [run],
                root / "dataset",
                val_fraction=0,
                detections_name="sam3_detections.json",
            )
            self.assertEqual(
                manifest["boxes_by_class"]["train"],
                {"yellow_cube": 1, "blue_cube": 1, "red_cube": 1},
            )


if __name__ == "__main__":
    unittest.main()
