import tempfile
import unittest
from pathlib import Path

from src.config import load_problem3_config


class ConfigInheritanceTests(unittest.TestCase):
    def test_problem3_config_inherits_shared_training_settings(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "problem3_grasping_sorting.yaml"
        )

        config = load_problem3_config(config_path)

        self.assertEqual(config["project"]["seed"], 42)
        self.assertEqual(config["training"]["epochs"], 50)
        self.assertEqual(
            [camera["placement"] for camera in config["components"]["cameras"]["views"]],
            ["overhead", "side", "robot_wrist"],
        )
        self.assertEqual(
            config["model"]["class_names"],
            ["yellow_cube", "blue_cube", "red_cube", "cap", "vial"],
        )

    def test_merges_nested_base_settings_with_task_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "common.yaml").write_text(
                "project:\n  seed: 42\npaths:\n  data_root: data\n"
            )
            (root / "problem.yaml").write_text(
                "base_config: common.yaml\n"
                "project:\n  name: cube-vision\n"
                "components:\n  cameras:\n    views:\n"
                "      - id: overhead\n        role: observation\n"
                "        placement: overhead\n"
            )

            config = load_problem3_config(root / "problem.yaml")

            self.assertEqual(config["project"], {"seed": 42, "name": "cube-vision"})
            self.assertEqual(config["paths"]["data_root"], "data")
            self.assertNotIn("base_config", config)

    def test_rejects_config_inheritance_cycles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.yaml").write_text("base_config: b.yaml\n")
            (root / "b.yaml").write_text("base_config: a.yaml\n")

            with self.assertRaisesRegex(ValueError, "inheritance cycle"):
                load_problem3_config(root / "a.yaml")


if __name__ == "__main__":
    unittest.main()
