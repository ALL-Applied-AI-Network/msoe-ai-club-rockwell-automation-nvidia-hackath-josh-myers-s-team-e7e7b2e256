"""Unit tests for natural language SceneGenerator and Digital Twin layout spawner."""

import math
from pathlib import Path
import tempfile
import unittest

from src.core.models import SceneSpec, SpawnedObject
from src.core.SceneGenerator import (
    CUBE_EDGE,
    CUBE_STACK_SITE_XY,
    DEFAULT_TABLE_TOP_Z,
    ParsedObjectRequest,
    REGION_BOUNDS,
    SceneLayoutSolver,
    ScenePromptParser,
    TextToSceneSpawner,
    VIAL_LENGTH,
    WORKSPACE_X_BOUNDS,
    WORKSPACE_Y_BOUNDS,
    _point_to_segment_dist,
)


class ScenePromptParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = ScenePromptParser()

    def test_parses_word_numbers_colors_and_regions(self) -> None:
        requests = self.parser.parse("Place two red cubes and one blue cube near the center")
        self.assertEqual(len(requests), 2)

        self.assertEqual(requests[0].category, "cube")
        self.assertEqual(requests[0].count, 2)
        self.assertEqual(requests[0].color, "red")
        self.assertEqual(requests[0].region, "center")

        self.assertEqual(requests[1].category, "cube")
        self.assertEqual(requests[1].count, 1)
        self.assertEqual(requests[1].color, "blue")
        self.assertEqual(requests[1].region, "center")

    def test_parses_digits_vials_and_caps(self) -> None:
        requests = self.parser.parse("Spawn 3 vials on the left and 2 caps on the right")
        self.assertEqual(len(requests), 2)

        self.assertEqual(requests[0].category, "vial")
        self.assertEqual(requests[0].count, 3)
        self.assertEqual(requests[0].region, "left")

        self.assertEqual(requests[1].category, "cap")
        self.assertEqual(requests[1].count, 2)
        self.assertEqual(requests[1].region, "right")

    def test_parses_explicit_coordinates_and_yaw(self) -> None:
        requests = self.parser.parse(
            "Place a yellow cube at (0.35, 0.10) angled at 30 degrees"
        )
        self.assertEqual(len(requests), 1)
        req = requests[0]
        self.assertEqual(req.category, "cube")
        self.assertEqual(req.count, 1)
        self.assertEqual(req.color, "yellow")
        self.assertIsNotNone(req.explicit_position)
        self.assertAlmostEqual(req.explicit_position[0], 0.35)
        self.assertAlmostEqual(req.explicit_position[1], 0.10)
        self.assertAlmostEqual(req.yaw_deg, 30.0)

    def test_parses_yaw_angle_does_not_corrupt_count(self) -> None:
        requests = self.parser.parse(
            "Place 3 yellow cubes, 2 blue cubes, and a clear vial angled at 45 degrees"
        )
        self.assertEqual(len(requests), 3)
        self.assertEqual(requests[0].category, "cube")
        self.assertEqual(requests[0].count, 3)
        self.assertEqual(requests[0].color, "yellow")

        self.assertEqual(requests[1].category, "cube")
        self.assertEqual(requests[1].count, 2)
        self.assertEqual(requests[1].color, "blue")

        self.assertEqual(requests[2].category, "vial")
        self.assertEqual(requests[2].count, 1)
        self.assertEqual(requests[2].color, "clear")
        self.assertAlmostEqual(requests[2].yaw_deg, 45.0)

    def test_parses_cap_orientation(self) -> None:
        down_requests = self.parser.parse("Spawn a cap narrow end down")
        self.assertEqual(down_requests[0].orientation, "narrow_end_down")

        up_requests = self.parser.parse("Spawn a cap upright")
        self.assertEqual(up_requests[0].orientation, "narrow_end_up")

    def test_empty_prompt_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.parser.parse("")
        with self.assertRaises(ValueError):
            self.parser.parse("   \n\t  ")

    def test_fallback_custom_vlm_parser(self) -> None:
        mock_vlm = lambda prompt: [
            ParsedObjectRequest(category="cube", count=4, color="blue", region="front")
        ]
        parser_with_vlm = ScenePromptParser(vlm_parser=mock_vlm)
        requests = parser_with_vlm.parse("Any abstract instructions here")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].count, 4)
        self.assertEqual(requests[0].color, "blue")
        self.assertEqual(requests[0].region, "front")


class SceneLayoutSolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.solver = SceneLayoutSolver()

    def test_solver_places_objects_within_workspace(self) -> None:
        requests = [
            ParsedObjectRequest(category="cube", count=3, color="red", region="center"),
            ParsedObjectRequest(category="vial", count=1, region="left"),
            ParsedObjectRequest(category="cap", count=2, region="right"),
        ]
        spawned = self.solver.solve(requests, seed=42)
        self.assertEqual(len(spawned), 6)

        for obj in spawned:
            x, y, z = obj.position
            self.assertGreaterEqual(x, WORKSPACE_X_BOUNDS[0] - 0.05)
            self.assertLessEqual(x, WORKSPACE_X_BOUNDS[1] + 0.05)
            self.assertGreaterEqual(y, WORKSPACE_Y_BOUNDS[0] - 0.05)
            self.assertLessEqual(y, WORKSPACE_Y_BOUNDS[1] + 0.05)
            self.assertGreater(z, DEFAULT_TABLE_TOP_Z)

    def test_solver_avoids_collisions_between_cubes(self) -> None:
        requests = [
            ParsedObjectRequest(category="cube", count=5, color="yellow", region="scattered")
        ]
        spawned = self.solver.solve(requests, seed=100)
        self.assertEqual(len(spawned), 5)

        for i in range(len(spawned)):
            for j in range(i + 1, len(spawned)):
                p1 = spawned[i].position[:2]
                p2 = spawned[j].position[:2]
                dist = math.dist(p1, p2)
                self.assertGreaterEqual(dist, self.solver.min_clearance)

    def test_solver_avoids_collisions_with_vial(self) -> None:
        requests = [
            ParsedObjectRequest(category="vial", count=1, region="center", yaw_deg=0.0),
            ParsedObjectRequest(category="cube", count=3, color="blue", region="center"),
        ]
        spawned = self.solver.solve(requests, seed=200)
        vial = next(obj for obj in spawned if obj.category == "vial")
        cubes = [obj for obj in spawned if obj.category == "cube"]

        vx, vy, _ = vial.position
        v_start = (vx - VIAL_LENGTH / 2.0, vy)
        v_end = (vx + VIAL_LENGTH / 2.0, vy)

        for cube in cubes:
            cx, cy, _ = cube.position
            dist = _point_to_segment_dist((cx, cy), v_start, v_end)
            self.assertGreaterEqual(dist, self.solver.min_clearance)

    def test_solver_avoids_cube_stack_site(self) -> None:
        requests = [
            ParsedObjectRequest(category="cube", count=6, color="red", region="scattered")
        ]
        spawned = self.solver.solve(requests, seed=42)
        for obj in spawned:
            dist = math.dist(obj.position[:2], CUBE_STACK_SITE_XY)
            self.assertGreaterEqual(dist, 0.04)

    def test_seed_determinism(self) -> None:
        requests = [
            ParsedObjectRequest(category="cube", count=2, color="red"),
            ParsedObjectRequest(category="vial", count=1),
        ]
        first = self.solver.solve(requests, seed=777)
        second = self.solver.solve(requests, seed=777)
        third = self.solver.solve(requests, seed=888)

        self.assertEqual([o.position for o in first], [o.position for o in second])
        self.assertNotEqual([o.position for o in first], [o.position for o in third])

    def test_out_of_bounds_explicit_position_raises(self) -> None:
        requests = [
            ParsedObjectRequest(
                category="cube",
                count=1,
                explicit_position=(5.0, 5.0),
            )
        ]
        with self.assertRaises(ValueError):
            self.solver.solve(requests, seed=42)


class TextToSceneSpawnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spawner = TextToSceneSpawner()

    def test_generate_and_round_trip_serialization(self) -> None:
        prompt = "Place 2 red cubes and 1 vial on the left"
        spec = self.spawner.generate(prompt, seed=42)

        self.assertEqual(spec.prompt, prompt)
        self.assertEqual(len(spec.objects), 3)
        self.assertEqual(spec.objects[0].category, "cube")
        self.assertEqual(spec.objects[0].color, "red")
        self.assertEqual(spec.objects[2].category, "vial")

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_scene.json"
            saved_path = self.spawner.save_spec(spec, file_path)
            self.assertTrue(saved_path.is_file())

            loaded_spec = self.spawner.load_spec(saved_path)
            self.assertEqual(loaded_spec.prompt, spec.prompt)
            self.assertEqual(loaded_spec.seed, spec.seed)
            self.assertEqual(len(loaded_spec.objects), len(spec.objects))
            self.assertEqual(
                loaded_spec.objects[0].class_name, spec.objects[0].class_name
            )
            self.assertAlmostEqual(
                loaded_spec.objects[0].position[0], spec.objects[0].position[0]
            )
            self.assertAlmostEqual(
                loaded_spec.objects[0].position[1], spec.objects[0].position[1]
            )


if __name__ == "__main__":
    unittest.main()
