"""Tests for normalizing Cosmos outputs used as detection labels."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from perceive import normalise_label_coordinates


class PerceiveLabelTests(unittest.TestCase):
    def _image(self, path: Path, size=(100, 200)):
        width, height = size
        self.assertTrue(
            cv2.imwrite(str(path), np.zeros((height, width, 3), dtype=np.uint8))
        )

    def test_normalizes_qwen_grid_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.png"
            self._image(image)
            parsed = {
                "objects": [{"center": [250, 500], "bbox": [200, 300, 400, 700]}]
            }

            result = normalise_label_coordinates(parsed, image)

            self.assertEqual(result["coordinate_convention"], "unit")
            self.assertEqual(result["objects"][0]["center"], [0.25, 0.5])
            self.assertEqual(result["objects"][0]["bbox"], [0.2, 0.3, 0.4, 0.7])

    def test_reorders_reversed_bbox_corners(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.png"
            self._image(image)
            parsed = {
                "objects": [
                    {"bbox": [0.4, 0.7, 0.2, 0.3], "center": [0.3, 0.5]}
                ]
            }

            result = normalise_label_coordinates(parsed, image)

            self.assertEqual(
                result["objects"][0]["bbox"], [0.2, 0.3, 0.4, 0.7]
            )

    def test_normalizes_absolute_pixel_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.png"
            self._image(image, size=(1200, 1200))
            parsed = {
                "objects": [
                    {"center": [1100, 1050], "bbox": [1050, 1001, 1150, 1101]}
                ]
            }

            result = normalise_label_coordinates(parsed, image)

            self.assertEqual(
                result["objects"][0]["bbox"],
                [1050 / 1200, 1001 / 1200, 1150 / 1200, 1101 / 1200],
            )


if __name__ == "__main__":
    unittest.main()
