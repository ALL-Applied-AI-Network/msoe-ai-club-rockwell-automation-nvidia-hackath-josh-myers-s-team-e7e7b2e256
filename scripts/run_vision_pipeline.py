"""Build a leakage-aware object dataset and optionally train YOLO26n."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.orchestrator import Problem3Orchestrator
from scripts.label_with_sam3 import _load_sam3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        type=Path,
        help="captured Isaac scenario folders containing world_camera.png and camera.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/problem3_grasping_sorting.yaml"),
    )
    parser.add_argument(
        "--annotation-source",
        choices=("simulator_ground_truth", "sam3"),
        help="defaults to data.annotation_source in the config",
    )
    parser.add_argument(
        "--validation-group",
        help="hold out all samples whose parent capture folder has this name",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        help="required with --annotation-source sam3",
    )
    parser.add_argument("--sam-device", choices=("cuda",), default="cuda")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--weights", help="override configured pretrained YOLO weights")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    args = parser.parse_args()

    sam3_model = None
    if args.annotation_source == "sam3":
        if args.sam_checkpoint is None:
            parser.error("--sam-checkpoint is required with --annotation-source sam3")
        if not args.sam_checkpoint.is_file():
            parser.error(f"SAM checkpoint not found: {args.sam_checkpoint}")
        sam3_model = _load_sam3(args.sam_checkpoint.resolve(), args.sam_device, 0.5)

    orchestrator = Problem3Orchestrator(
        args.config,
        sam3_model=sam3_model,
    )
    if args.annotation_source is not None:
        orchestrator.config.setdefault("data", {})[
            "annotation_source"
        ] = args.annotation_source
    if args.validation_group is not None:
        orchestrator.config.setdefault("data", {})[
            "validation_group"
        ] = args.validation_group
    for name, value in (
        ("epochs", args.epochs),
        ("imgsz", args.imgsz),
        ("batch", args.batch),
    ):
        if value is not None:
            if value <= 0:
                parser.error(f"--{name} must be positive")
            orchestrator.yolo_model.train_options[name] = value
    if args.weights is not None:
        orchestrator.yolo_model.weights = args.weights

    dataset_dir = orchestrator.run_data_generation(
        [str(run.resolve()) for run in args.runs]
    )
    print(f"Dataset: {dataset_dir}")
    print(f"Review label previews: {dataset_dir / 'previews'}")
    print(f"Dataset manifest: {dataset_dir / 'manifest.json'}")
    if not args.skip_training:
        if sam3_model is not None:
            import gc
            import torch

            orchestrator.sam_labeler.sam3_model = None
            del sam3_model
            gc.collect()
            torch.cuda.empty_cache()
        artifact = orchestrator.train_detector(dataset_dir)
        print(f"Best model: {artifact.weights_path}")
        print(f"Metrics: {artifact.metrics}")


if __name__ == "__main__":
    main()
