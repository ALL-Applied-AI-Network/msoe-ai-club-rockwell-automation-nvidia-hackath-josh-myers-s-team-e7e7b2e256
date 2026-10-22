#!/usr/bin/env python3
"""Generate a focused, lightweight student detector for a specific part from text."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.PartModelFactory import PartModelFactory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill foundation model reasoning into a lightweight part-specific detector."
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="Target part prompt (e.g. 'yellow cap', 'clear vial', 'blue cube')",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        type=Path,
        help="Captured simulator scenario folders (e.g. output/runs/.../view_*)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("artifacts/part_model"),
        help="Output directory for the distilled dataset and trained student weights",
    )
    parser.add_argument(
        "--state-aware",
        action="store_true",
        help="Distill functional orientations (e.g. cap_narrow_end_up vs cap_wide_end_up)",
    )
    parser.add_argument(
        "--validation-group",
        help="Hold out all samples whose parent capture folder has this name",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of runs to hold out for validation (if not using validation-group)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Prepare and verify the distilled dataset without training YOLO",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable automatic generation of demonstration video after student model training",
    )
    parser.add_argument(
        "--weights",
        default="weights/yolo26n.pt",
        help="Pretrained YOLO base weights (default: weights/yolo26n.pt)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    factory = PartModelFactory(base_weights=args.weights)
    train_options = {
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
    }

    target_part, dataset_dir, artifact = factory.create_model(
        prompt=args.prompt,
        run_paths=args.runs,
        output_dir=args.output_dir,
        state_aware=args.state_aware,
        skip_training=args.skip_training,
        train_options=train_options,
        val_fraction=args.val_fraction,
        validation_group=args.validation_group,
        render_video=not args.no_video,
    )

    print("=" * 60)
    print(f"Target Part:   {target_part.prompt}")
    print(f"Category:      {target_part.category}")
    print(f"State-Aware:   {target_part.state_aware}")
    print(f"Model Classes: {list(target_part.class_names)}")
    print(f"Dataset Dir:   {dataset_dir}")
    print(f"Manifest:      {dataset_dir / 'manifest.json'}")
    print(f"Previews:      {dataset_dir / 'previews'}")
    if artifact:
        print(f"Trained Model: {artifact.weights_path}")
        print(f"Metrics:       {artifact.metrics}")
    print("=" * 60)


if __name__ == "__main__":
    main()

