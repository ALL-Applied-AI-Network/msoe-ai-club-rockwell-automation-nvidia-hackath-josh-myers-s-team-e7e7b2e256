"""Train a YOLO26 nano detector from a reviewed dataset built by this project."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="YOLO data.yaml")
    parser.add_argument("--model", default="yolo26n.pt", help="model/checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="training image size; use a large size for tiny tabletop cubes",
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", help="Ultralytics device, such as 0 or cpu")
    parser.add_argument("--project", type=Path, default=Path("artifacts/yolo26n"))
    parser.add_argument("--name", default="cube-colour-v1")
    parser.add_argument("--no-val", action="store_true", help="training-only smoke test")
    args = parser.parse_args()

    data = args.data.resolve()
    if not data.is_file():
        parser.error(f"dataset YAML not found: {data}")
    if args.epochs <= 0 or args.imgsz <= 0 or args.batch <= 0 or args.workers < 0:
        parser.error("epochs/imgsz/batch must be positive and workers non-negative")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is optional. Install it with `uv sync --extra yolo` "
            "and rerun this command."
        ) from exc

    model = YOLO(args.model)
    train_options = {
        "data": str(data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(args.project),
        "name": args.name,
        "val": not args.no_val,
    }
    if args.device is not None:
        train_options["device"] = args.device
    model.train(**train_options)


if __name__ == "__main__":
    main()
