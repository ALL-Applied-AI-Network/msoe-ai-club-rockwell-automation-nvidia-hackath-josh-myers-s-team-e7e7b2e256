import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from scripts.build_yolo_dataset import build_dataset


class BuildYoloDatasetTests(unittest.TestCase):
    def _make_run(self, root: Path, name: str, objects: list[dict]) -> Path:
        run = root / name
        run.mkdir()
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(run / "world_camera.png"), image))
        (run / "detections.json").write_text(
            json.dumps(
                {
                    "coordinate_convention": "unit",
                    "model": "Cosmos3-Nano",
                    "objects": objects,
                }
            )
        )
        return run

    def test_exports_normalized_boxes_and_keeps_runs_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blue = {
                "category": "cube",
                "colour": "blue",
                "bbox": [0.1, 0.2, 0.3, 0.6],
            }
            yellow = {
                "category": "cube",
                "colour": "yellow",
                "bbox": [0.4, 0.3, 0.8, 0.7],
            }
            distractor = {
                "category": "cap",
                "colour": "yellow",
                "bbox": [0.2, 0.2, 0.3, 0.3],
            }
            runs = [
                self._make_run(root, "capture_a", [blue, distractor]),
                self._make_run(root, "capture_b", [yellow]),
            ]
            output = root / "dataset"

            manifest = build_dataset(runs, output, val_fraction=0.5, seed=42)

            self.assertEqual(
                set(manifest["runs"]["train"]).intersection(manifest["runs"]["val"]),
                set(),
            )
            self.assertEqual(manifest["teachers"], ["Cosmos3-Nano"])
            self.assertEqual(
                manifest["boxes_by_class"]["train"]["blue_cube"]
                + manifest["boxes_by_class"]["val"]["blue_cube"],
                1,
            )
            yaml_data = (output / "data.yaml").read_text()
            self.assertIn("yellow_cube", yaml_data)
            self.assertTrue(list((output / "previews").rglob("*.jpg")))
            labels = list((output / "labels").rglob("*.txt"))
            self.assertEqual(len(labels), 2)
            all_lines = [line for label in labels for line in label.read_text().splitlines()]
            self.assertIn("1 0.20000000 0.40000000 0.20000000 0.40000000", all_lines)
            self.assertIn("0 0.60000000 0.50000000 0.40000000 0.40000000", all_lines)

    def test_rejects_validation_split_with_only_one_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self._make_run(root, "only_run", [])
            with self.assertRaisesRegex(ValueError, "at least two distinct runs"):
                build_dataset([run], root / "dataset")

    def test_rejects_unknown_cube_colour_instead_of_silently_dropping_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self._make_run(
                root,
                "bad_label",
                [{"category": "cube", "colour": "unknown", "bbox": [0.1, 0.1, 0.2, 0.2]}],
            )
            with self.assertRaisesRegex(ValueError, "unsupported colour"):
                build_dataset([run], root / "dataset", val_fraction=0)

    def test_refuses_to_overwrite_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self._make_run(root, "capture", [])
            output = root / "dataset"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                build_dataset([run], output, val_fraction=0)

    def test_validation_group_keeps_capture_batches_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_group = root / "train"
            val_group = root / "val"
            train_group.mkdir()
            val_group.mkdir()
            train_runs = [
                self._make_run(train_group, f"view_{index}", [])
                for index in range(3)
            ]
            val_runs = [
                self._make_run(val_group, f"view_{index}", [])
                for index in range(2)
            ]

            manifest = build_dataset(
                train_runs + val_runs,
                root / "dataset",
                validation_group="val",
            )

            self.assertEqual(len(manifest["runs"]["train"]), 3)
            self.assertEqual(len(manifest["runs"]["val"]), 2)
            self.assertTrue(all(name.startswith("train/") for name in manifest["runs"]["train"]))
            self.assertTrue(all(name.startswith("val/") for name in manifest["runs"]["val"]))

    def test_marks_simulator_ground_truth_distinct_from_pseudo_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train").mkdir()
            (root / "val").mkdir()
            first = self._make_run(root / "train", "first", [])
            second = self._make_run(root / "val", "second", [])
            for run in (first, second):
                detections_path = run / "detections.json"
                detections = json.loads(detections_path.read_text())
                detections["model"] = "Isaac Sim ground truth projection"
                detections_path.write_text(json.dumps(detections))

            manifest = build_dataset(
                [first, second],
                root / "dataset",
                validation_group="val",
            )

            self.assertEqual(manifest["label_status"], "simulator_ground_truth")

    def test_exports_configured_cap_and_vial_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_group = root / "train"
            val_group = root / "val"
            train_group.mkdir()
            val_group.mkdir()
            cap = self._make_run(
                train_group,
                "cap",
                [{"category": "cap", "class_name": "cap", "bbox": [0.1, 0.1, 0.2, 0.2]}],
            )
            vial = self._make_run(
                val_group,
                "vial",
                [{"category": "vial", "class_name": "vial", "bbox": [0.3, 0.2, 0.6, 0.8]}],
            )

            manifest = build_dataset(
                [cap, vial],
                root / "dataset",
                validation_group="val",
                class_names=("cap", "vial"),
            )

            self.assertEqual(manifest["class_names"], ["cap", "vial"])
            self.assertEqual(manifest["boxes_by_class"]["train"]["cap"], 1)
            self.assertEqual(manifest["boxes_by_class"]["val"]["vial"], 1)
            data_yaml = (root / "dataset" / "data.yaml").read_text()
            self.assertIn("cap", data_yaml)
            self.assertIn("vial", data_yaml)


if __name__ == "__main__":
    unittest.main()
