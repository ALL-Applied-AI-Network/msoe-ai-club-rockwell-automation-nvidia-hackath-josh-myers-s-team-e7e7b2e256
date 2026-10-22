"""Generate Digital Twin scenes from natural language text prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Callable, Sequence

from .models import SceneSpec, SpawnedObject

# Standard cell workspace constants
DEFAULT_TABLE_TOP_Z = 0.90 - 0.0051  # 0.8949 m
CUBE_EDGE = 0.010
CUBE_MASS = 0.004
VIAL_LENGTH = 0.1400
VIAL_RADIUS = 0.017 / 2
CAP_RADIUS = 0.03175 / 2
CAP_HEIGHT = 0.01570

# Safe reachable bounds for FANUC ER-4iA arm on the tabletop
WORKSPACE_X_BOUNDS = (0.24, 0.48)
WORKSPACE_Y_BOUNDS = (-0.22, 0.28)

REGION_BOUNDS = {
    "center": ((0.28, 0.44), (-0.12, 0.12)),
    "middle": ((0.28, 0.44), (-0.12, 0.12)),
    "left": ((0.25, 0.46), (0.08, 0.26)),
    "right": ((0.25, 0.46), (-0.20, -0.08)),
    "front": ((0.36, 0.48), (-0.18, 0.24)),
    "far": ((0.36, 0.48), (-0.18, 0.24)),
    "back": ((0.24, 0.34), (-0.18, 0.24)),
    "near": ((0.24, 0.34), (-0.18, 0.24)),
    "scattered": (WORKSPACE_X_BOUNDS, WORKSPACE_Y_BOUNDS),
    "random": (WORKSPACE_X_BOUNDS, WORKSPACE_Y_BOUNDS),
    "table": (WORKSPACE_X_BOUNDS, WORKSPACE_Y_BOUNDS),
}

CUBE_STACK_SITE_XY = (0.28, 0.10)
FIXTURE_XY = (0.555, -0.640)

WORD_NUMBERS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


@dataclass
class ParsedObjectRequest:
    category: str
    count: int = 1
    color: str | None = None
    region: str | None = None
    explicit_position: tuple[float, float] | None = None
    yaw_deg: float | None = None
    orientation: str = "upright"
    metadata: dict[str, Any] = field(default_factory=dict)


def _dist_sq(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def _point_to_segment_dist(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    ab = (b[0] - a[0], b[1] - a[1])
    ab_len_sq = ab[0] ** 2 + ab[1] ** 2
    if ab_len_sq == 0.0:
        return math.sqrt(_dist_sq(p, a))
    ap = (p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, (ap[0] * ab[0] + ap[1] * ab[1]) / ab_len_sq))
    projection = (a[0] + t * ab[0], a[1] + t * ab[1])
    return math.sqrt(_dist_sq(p, projection))


def _segments_dist(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
) -> float:
    # 2D segment to segment distance
    return min(
        _point_to_segment_dist(a1, b1, b2),
        _point_to_segment_dist(a2, b1, b2),
        _point_to_segment_dist(b1, a1, a2),
        _point_to_segment_dist(b2, a1, a2),
    )


class ScenePromptParser:
    """Extract object categories, counts, colors, and layout hints from text."""

    def __init__(self, vlm_parser: Callable[[str], list[ParsedObjectRequest]] | None = None) -> None:
        self.vlm_parser = vlm_parser

    def parse(self, prompt: str) -> list[ParsedObjectRequest]:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty")

        if self.vlm_parser is not None:
            try:
                requests = self.vlm_parser(prompt)
                if requests:
                    return requests
            except Exception:
                # Fall back to rule-based parser on any VLM error
                pass

        return self._rule_based_parse(prompt)

    def _rule_based_parse(self, prompt: str) -> list[ParsedObjectRequest]:
        normalized = prompt.lower()

        # Protect parenthesized coordinates like (0.35, 0.10) from comma splitting
        coord_pattern = r"\([^)]+\)"
        coords = re.findall(coord_pattern, normalized)
        placeholder_map: dict[str, str] = {}
        for idx, coord in enumerate(coords):
            key = f"__COORD_{idx}__"
            placeholder_map[key] = coord
            normalized = normalized.replace(coord, key)

        # Split on commas, 'and', semicolons, periods
        clauses = re.split(r"[,;.\n]|\band\b", normalized)
        # Restore placeholders
        restored_clauses = []
        for clause in clauses:
            for key, val in placeholder_map.items():
                clause = clause.replace(key, val)
            restored_clauses.append(clause)
        clauses = restored_clauses

        requests: list[ParsedObjectRequest] = []

        # Find global spatial region hint if mentioned
        global_region = None
        for reg in REGION_BOUNDS:
            if re.search(r"\b" + re.escape(reg) + r"\b", normalized):
                global_region = reg
                break

        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue

            # Identify category
            category = None
            if re.search(r"\b(cube|cubes|block|blocks)\b", clause):
                category = "cube"
            elif re.search(r"\b(vial|vials|tube|tubes|bottle|bottles)\b", clause):
                category = "vial"
            elif re.search(r"\b(cap|caps|puck|pucks)\b", clause):
                category = "cap"

            if not category:
                continue

            # Explicit position: e.g. at (0.35, 0.10) or at 0.35, 0.10
            explicit_pos = None
            pos_match = re.search(
                r"\((-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\)|at\s+(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)",
                clause,
            )
            if pos_match:
                groups = [g for g in pos_match.groups() if g is not None]
                if len(groups) >= 2:
                    explicit_pos = (float(groups[0]), float(groups[1]))

            # Explicit angle / yaw
            explicit_yaw = None
            yaw_match = re.search(r"(-?\d+\.?\d*)\s*(?:deg|degrees)", clause)
            if yaw_match:
                explicit_yaw = float(yaw_match.group(1))

            # Clean clause to avoid mistaking angles or coords for object counts
            clean_for_count = re.sub(r"(-?\d+\.?\d*)\s*(?:deg|degrees)", "", clause)
            clean_for_count = re.sub(r"\([^)]+\)", "", clean_for_count)
            clean_for_count = re.sub(
                r"at\s+(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)", "", clean_for_count
            )

            # Extract count
            count = 1
            num_match = re.search(r"\b(\d+)\b", clean_for_count)
            if num_match:
                count = int(num_match.group(1))
            else:
                for word, val in WORD_NUMBERS.items():
                    if re.search(r"\b" + word + r"\b", clean_for_count):
                        count = val
                        break

            # Extract color
            color = None
            for candidate in ("red", "yellow", "blue", "green", "white", "clear"):
                if re.search(r"\b" + candidate + r"\b", clause):
                    color = candidate
                    break
            if category == "cube" and not color:
                color = "yellow"
            elif category == "vial" and not color:
                color = "clear"
            elif category == "cap" and not color:
                color = "yellow"

            # Clause-specific region
            region = global_region
            for reg in REGION_BOUNDS:
                if re.search(r"\b" + re.escape(reg) + r"\b", clause):
                    region = reg
                    break

            # Orientation for cap
            orientation = "upright"
            if category == "cap":
                if "narrow end down" in clause or "upside down" in clause or "dome down" in clause:
                    orientation = "narrow_end_down"
                else:
                    orientation = "narrow_end_up"

            requests.append(
                ParsedObjectRequest(
                    category=category,
                    count=max(1, count),
                    color=color,
                    region=region,
                    explicit_position=explicit_pos,
                    yaw_deg=explicit_yaw,
                    orientation=orientation,
                )
            )

        if not requests:
            # Fallback if no specific object pattern matched: default to requested count of cubes
            requests.append(
                ParsedObjectRequest(
                    category="cube",
                    count=1,
                    color="yellow",
                    region=global_region or "scattered",
                )
            )

        return requests


class SceneLayoutSolver:
    """Solve collision-free physical poses for requested objects in the cell."""

    def __init__(
        self,
        table_z: float = DEFAULT_TABLE_TOP_Z,
        stack_site: tuple[float, float] = CUBE_STACK_SITE_XY,
        min_clearance: float = 0.035,
    ) -> None:
        self.table_z = table_z
        self.stack_site = stack_site
        self.min_clearance = min_clearance

    def solve(
        self,
        requests: Sequence[ParsedObjectRequest],
        seed: int | None = 42,
        max_attempts: int = 5000,
    ) -> tuple[SpawnedObject, ...]:
        rng = random.Random(seed)
        spawned: list[SpawnedObject] = []

        # Track placed geometries for collision avoidance
        # Points: (x, y, radius)
        placed_circles: list[tuple[float, float, float]] = []
        # Segments for vials: ((x1, y1), (x2, y2), radius)
        placed_segments: list[tuple[tuple[float, float], tuple[float, float], float]] = []

        # Avoid stack site and fixture
        placed_circles.append((self.stack_site[0], self.stack_site[1], 0.04))
        placed_circles.append((FIXTURE_XY[0], FIXTURE_XY[1], 0.12))

        for req in requests:
            for item_idx in range(req.count):
                placed = False
                color = req.color or "yellow"
                cat = req.category

                # Determine radius and class name
                if cat == "cube":
                    radius = CUBE_EDGE * 0.8
                    class_name = f"{color}_cube"
                    z_pos = self.table_z + CUBE_EDGE / 2.0
                elif cat == "vial":
                    radius = VIAL_RADIUS
                    class_name = "vial"
                    z_pos = self.table_z + VIAL_RADIUS
                elif cat == "cap":
                    radius = CAP_RADIUS
                    class_name = "cap"
                    z_pos = self.table_z + CAP_HEIGHT / 2.0
                else:
                    radius = 0.02
                    class_name = cat
                    z_pos = self.table_z + 0.01

                # If explicit position is given for this item
                if req.explicit_position is not None and item_idx == 0:
                    x, y = req.explicit_position
                    yaw = req.yaw_deg if req.yaw_deg is not None else 0.0
                    self._check_workspace(x, y)
                    spawned.append(
                        SpawnedObject(
                            category=cat,
                            class_name=class_name,
                            position=(x, y, z_pos),
                            yaw_deg=yaw,
                            orientation=req.orientation,
                            color=color,
                        )
                    )
                    self._record_geometry(cat, x, y, yaw, radius, placed_circles, placed_segments)
                    continue

                # Get spatial bounds for region
                bounds = REGION_BOUNDS.get(
                    req.region or "scattered",
                    (WORKSPACE_X_BOUNDS, WORKSPACE_Y_BOUNDS),
                )
                (x_min, x_max), (y_min, y_max) = bounds

                for _ in range(max_attempts):
                    x = rng.uniform(x_min, x_max)
                    y = rng.uniform(y_min, y_max)
                    yaw = (
                        req.yaw_deg
                        if req.yaw_deg is not None
                        else rng.uniform(-45.0, 45.0)
                    )

                    if self._is_collision_free(
                        cat, x, y, yaw, radius, placed_circles, placed_segments
                    ):
                        spawned.append(
                            SpawnedObject(
                                category=cat,
                                class_name=class_name,
                                position=(x, y, z_pos),
                                yaw_deg=yaw,
                                orientation=req.orientation,
                                color=color,
                            )
                        )
                        self._record_geometry(
                            cat, x, y, yaw, radius, placed_circles, placed_segments
                        )
                        placed = True
                        break

                if not placed:
                    raise ValueError(
                        f"Cannot place {cat} ({item_idx + 1}/{req.count}) "
                        f"in region {req.region!r} without collision."
                    )

        return tuple(spawned)

    def _check_workspace(self, x: float, y: float) -> None:
        if not (
            WORKSPACE_X_BOUNDS[0] - 0.05 <= x <= WORKSPACE_X_BOUNDS[1] + 0.05
            and WORKSPACE_Y_BOUNDS[0] - 0.05 <= y <= WORKSPACE_Y_BOUNDS[1] + 0.05
        ):
            raise ValueError(
                f"Position ({x:.3f}, {y:.3f}) is outside tabletop workspace."
            )

    def _is_collision_free(
        self,
        category: str,
        x: float,
        y: float,
        yaw: float,
        radius: float,
        circles: list[tuple[float, float, float]],
        segments: list[tuple[tuple[float, float], tuple[float, float], float]],
    ) -> bool:
        if category == "vial":
            yaw_rad = math.radians(yaw)
            dx = (VIAL_LENGTH / 2.0) * math.cos(yaw_rad)
            dy = (VIAL_LENGTH / 2.0) * math.sin(yaw_rad)
            p1 = (x - dx, y - dy)
            p2 = (x + dx, y + dy)

            # Check workspace bounds for both endpoints
            for pt in (p1, p2):
                if not (
                    WORKSPACE_X_BOUNDS[0] <= pt[0] <= WORKSPACE_X_BOUNDS[1]
                    and WORKSPACE_Y_BOUNDS[0] <= pt[1] <= WORKSPACE_Y_BOUNDS[1]
                ):
                    return False

            # Check against existing circles
            for cx, cy, cr in circles:
                if _point_to_segment_dist((cx, cy), p1, p2) < (cr + self.min_clearance):
                    return False

            # Check against existing segments
            for s1, s2, sr in segments:
                if _segments_dist(p1, p2, s1, s2) < (sr + self.min_clearance + 0.01):
                    return False

            return True

        # Point / disc objects (cube, cap)
        p = (x, y)
        for cx, cy, cr in circles:
            if math.sqrt(_dist_sq(p, (cx, cy))) < (radius + cr + self.min_clearance):
                return False

        for s1, s2, sr in segments:
            if _point_to_segment_dist(p, s1, s2) < (radius + sr + self.min_clearance):
                return False

        return True

    def _record_geometry(
        self,
        category: str,
        x: float,
        y: float,
        yaw: float,
        radius: float,
        circles: list[tuple[float, float, float]],
        segments: list[tuple[tuple[float, float], tuple[float, float], float]],
    ) -> None:
        if category == "vial":
            yaw_rad = math.radians(yaw)
            dx = (VIAL_LENGTH / 2.0) * math.cos(yaw_rad)
            dy = (VIAL_LENGTH / 2.0) * math.sin(yaw_rad)
            p1 = (x - dx, y - dy)
            p2 = (x + dx, y + dy)
            segments.append((p1, p2, radius))
        else:
            circles.append((x, y, radius))


class TextToSceneSpawner:
    """High-level generator bridging text inputs to Digital Twin scenes."""

    def __init__(
        self,
        parser: ScenePromptParser | None = None,
        solver: SceneLayoutSolver | None = None,
    ) -> None:
        self.parser = parser or ScenePromptParser()
        self.solver = solver or SceneLayoutSolver()

    def generate(
        self,
        prompt: str,
        seed: int = 42,
        metadata: dict[str, Any] | None = None,
    ) -> SceneSpec:
        """Parse natural language prompt and generate valid collision-free scene."""
        requests = self.parser.parse(prompt)
        objects = self.solver.solve(requests, seed=seed)
        return SceneSpec(
            prompt=prompt,
            objects=objects,
            seed=seed,
            metadata=metadata or {},
        )

    def save_spec(self, spec: SceneSpec, path: Path | str) -> Path:
        """Serialize scene spec to a JSON file."""
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(spec.to_dict(), indent=2) + "\n")
        return target

    def load_spec(self, path: Path | str) -> SceneSpec:
        """Load a scene spec from a JSON file."""
        target = Path(path).resolve()
        data = json.loads(target.read_text())
        return SceneSpec.from_dict(data)
