"""Unit tests for the PartModelFactory and single-part distillation pipeline."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import cv2
import numpy as np

from src.core.PartModelFactory import PartModelFactory, TargetPartSpec


class PartModelFactoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.factory = PartModelFactory()

    def test_parse_target_part_cube(self) -> None:
        blue_cube = self.factory.parse_target_part("blue cube")
        self.assertEqual(blue_cube.category, "cube")
        self.assertEqual(blue_cube.color, "blue")
        self.assertEqual(blue_cube.class_names, ("blue_cube",))

        any_cube = self.factory.parse_target_part("cube")
        self.assertEqual(any_cube.category, "cube")
        self.assertIsNone(any_cube.color)
        self.assertEqual(
            any_cube.class_names, ("yellow_cube", "blue_cube", "red_cube")
        )

    def test_parse_target_part_cap_state_aware(self) -> None:
        cap_standard = self.factory.parse_target_part("yellow cap", state_aware=False)
        self.assertEqual(cap_standard.category, "cap")
        self.assertEqual(cap_standard.class_names, ("cap",))

        cap_state = self.factory.parse_target_part("yellow cap", state_aware=True)
        self.assertEqual(cap_state.category, "cap")
        self.assertTrue(cap_state.state_aware)
        self.assertEqual(
            cap_state.class_names, ("cap_narrow_end_up", "cap_wide_end_up")
        )

    def test_parse_target_part_vial(self) -> None:
        vial = self.factory.parse_target_part("clear vial")
        self.assertEqual(vial.category, "vial")
        self.assertEqual(vial.class_names, ("vial",))

    def test_generate_scenes_focuses_on_target_part(self) -> None:
        target_cap = self.factory.parse_target_part("yellow cap", state_aware=True)
        scenes = self.factory.generate_scenes(target_cap, num_scenes=3, seed=42)

        self.assertEqual(len(scenes), 3)
        for scene in scenes:
            cap_objects = [o for o in scene.objects if o.category == "cap"]
            self.assertGreater(len(cap_objects), 0)
            for cap in cap_objects:
                self.assertIn(cap.orientation, ("narrow_end_up", "narrow_end_down"))

    def test_distill_labels_filters_and_maps_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run = Path(tmpdir) / "run_01"
            run.mkdir()

            # Create mock simulated scenario detections containing cube, vial, and caps in both states
            mock_detections = {
                "coordinate_convention": "unit",
                "objects": [
                    {
                        "name": "SortCap0",
                        "category": "cap",
                        "orientation": "narrow_end_up",
                        "bbox": [0.1, 0.1, 0.2, 0.2],
                    },
                    {
                        "name": "SortCap1",
                        "category": "cap",
                        "orientation": "narrow_end_down",
                        "bbox": [0.3, 0.3, 0.4, 0.4],
                    },
                    {
                        "name": "SortCube0",
                        "category": "cube",
                        "colour": "blue",
                        "bbox": [0.5, 0.5, 0.6, 0.6],
                    },
                    {
                        "name": "SortVial0",
                        "category": "vial",
                        "colour": "clear",
                        "bbox": [0.7, 0.7, 0.8, 0.8],
                    },
                ],
            }
            (run / "sim_ground_truth_all_detections.json").write_text(
                json.dumps(mock_detections)
            )

            target_part = self.factory.parse_target_part("yellow cap", state_aware=True)
            saved = self.factory.distill_labels([run], target_part, output_name="part_detections.json")

            self.assertEqual(len(saved), 1)
            distilled = json.loads(saved[0].read_text())
            self.assertEqual(distilled["coordinate_convention"], "unit")

            # Cubes and vials must be filtered out! Only the 2 caps should remain:
            self.assertEqual(len(distilled["objects"]), 2)
            self.assertEqual(
                distilled["objects"][0]["class_name"], "cap_narrow_end_up"
            )
            self.assertEqual(
                distilled["objects"][1]["class_name"], "cap_wide_end_up"
            )

    def test_end_to_end_single_part_dataset_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_paths = []

            camera_to_world = np.eye(4)
            camera_to_world[3, 2] = 2.0

            # Create 2 runs (for train / val)
            for split_name in ("run_train", "run_val"):
                run = root / split_name
                run.mkdir()
                run_paths.append(run)

                cv2.imwrite(
                    str(run / "world_camera.png"),
                    np.zeros((100, 100, 3), dtype=np.uint8),
                )
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
                mock_detections = {
                    "coordinate_convention": "unit",
                    "objects": [
                        {
                            "name": "SortCap0",
                            "category": "cap",
                            "orientation": "narrow_end_up",
                            "bbox": [0.2, 0.2, 0.4, 0.4],
                        },
                        {
                            "name": "SortCube0",
                            "category": "cube",
                            "colour": "blue",
                            "bbox": [0.6, 0.6, 0.8, 0.8],
                        },
                    ],
                }
                (run / "sim_ground_truth_all_detections.json").write_text(
                    json.dumps(mock_detections)
                )

            out_dir = root / "output_part"
            target_part, dataset_dir, artifact = self.factory.create_model(
                prompt="yellow cap",
                run_paths=run_paths,
                output_dir=out_dir,
                state_aware=True,
                skip_training=True,
                val_fraction=0.5,
            )

            self.assertIsNone(artifact)
            self.assertTrue(dataset_dir.is_dir())
            self.assertTrue((dataset_dir / "data.yaml").is_file())
            manifest = json.loads((dataset_dir / "manifest.json").read_text())
            self.assertEqual(
                manifest["class_names"],
                ["cap_narrow_end_up", "cap_wide_end_up"],
            )
            self.assertEqual(
                manifest["boxes_by_class"]["train"]["cap_narrow_end_up"], 1
            )
            self.assertEqual(
                manifest["boxes_by_class"]["val"]["cap_narrow_end_up"], 1
            )
            self.assertEqual(
                manifest["boxes_by_class"]["train"]["cap_wide_end_up"], 0
            )
            self.assertEqual(
                manifest["boxes_by_class"]["val"]["cap_wide_end_up"], 0
            )

    @patch("src.core.PartModelFactory.PartModelFactory.train_student")
    @patch("src.core.PartModelFactory.PartModelFactory.distill_labels")
    @patch("src.core.PartModelFactory.PartModelFactory.build_part_dataset")
    def test_create_model_render_video_default_and_no_video(
        self, mock_build, mock_distill, mock_train
    ) -> None:
        from src.core.models import ModelArtifact

        mock_train.return_value = ModelArtifact(
            name="test_model",
            version="1.0",
            weights_path=Path("dummy/path/best.pt"),
            class_names=("cap",),
            metrics={"mAP50": 0.95},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run_01"
            run_dir.mkdir()
            frames_dir = run_dir / "frames"
            frames_dir.mkdir()
            (frames_dir / "f000.png").write_text("fake")

            out_dir = Path(tmpdir) / "out"

            # 1. Default render_video=True
            with patch(
                "scripts.render_student_video.StudentVideoRenderer"
            ) as mock_renderer_cls:
                mock_renderer_instance = mock_renderer_cls.return_value
                _, _, artifact = self.factory.create_model(
                    prompt="yellow cap",
                    run_paths=[run_dir],
                    output_dir=out_dir,
                    render_video=True,
                )
                self.assertTrue(mock_renderer_instance.render_video.called)

            # 2. Disabled render_video=False
            with patch(
                "scripts.render_student_video.StudentVideoRenderer"
            ) as mock_renderer_cls:
                mock_renderer_instance = mock_renderer_cls.return_value
                _, _, artifact = self.factory.create_model(
                    prompt="yellow cap",
                    run_paths=[run_dir],
                    output_dir=out_dir,
                    render_video=False,
                )
                self.assertFalse(mock_renderer_instance.render_video.called)

    def test_cli_no_video_flag(self) -> None:
        import subprocess

        res = subprocess.run(
            [".venv/bin/python", "scripts/generate_part_model.py", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("--no-video", res.stdout)


if __name__ == "__main__":
    unittest.main()
