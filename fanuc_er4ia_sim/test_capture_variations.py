"""Validate deterministic, safely spaced layouts for multi-view captures."""

import math
import unittest

import scene_config as C


class CaptureVariationTests(unittest.TestCase):
    def test_seed_reproduces_layout_and_orientations(self):
        self.assertEqual(C.sample_capture_view(42), C.sample_capture_view(42))

    def test_different_seeds_vary_layout_and_orientations(self):
        first_cubes, first_yaws = C.sample_capture_view(42)
        second_cubes, second_yaws = C.sample_capture_view(43)

        self.assertNotEqual(first_cubes, second_cubes)
        self.assertNotEqual(first_yaws, second_yaws)
        self.assertEqual(
            [cube[2] for cube in first_cubes],
            [cube[2] for cube in second_cubes],
        )
        self.assertEqual(len(first_cubes), 6)
        self.assertEqual(len(first_yaws), len(first_cubes))

    def test_sampled_cubes_keep_clearance_from_each_other_and_obstacles(self):
        cubes, _ = C.sample_capture_view(42)
        obstacles = list(C.SORT_CAPS) + [C.CUBE_STACK_XY]

        for index, (x, y, _) in enumerate(cubes):
            for other_x, other_y, _ in cubes[index + 1 :]:
                self.assertGreaterEqual(math.dist((x, y), (other_x, other_y)), 0.065)
            for obstacle_x, obstacle_y in obstacles:
                self.assertGreaterEqual(
                    math.dist((x, y), (obstacle_x, obstacle_y)), 0.065
                )


if __name__ == "__main__":
    unittest.main()
