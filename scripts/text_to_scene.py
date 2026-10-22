#!/usr/bin/env python3
"""Generate a Digital Twin scene layout JSON from a natural language text prompt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.SceneGenerator import TextToSceneSpawner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert natural language text prompt into a Digital Twin scene specification."
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="Natural language description (e.g. 'Place 2 red cubes and 1 vial on the left')",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path("output/scene_spec.json"),
        help="Output path for the scene JSON file (default: output/scene_spec.json)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for reproducible object layout (default: 42)",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the generated scene JSON to stdout instead of saving to file",
    )
    args = parser.parse_args()

    spawner = TextToSceneSpawner()
    spec = spawner.generate(args.prompt, seed=args.seed)

    serialized = json.dumps(spec.to_dict(), indent=2)
    if args.print_only:
        print(serialized)
        return

    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(serialized + "\n")
    print(f"Generated Digital Twin scene with {len(spec.objects)} objects.")
    print(f"Saved to: {out_path}")
    for idx, obj in enumerate(spec.objects):
        x, y, z = obj.position
        print(
            f"  [{idx + 1}] {obj.class_name} at ({x:.3f}, {y:.3f}, {z:.3f}) yaw={obj.yaw_deg:.1f}°"
        )


if __name__ == "__main__":
    main()

