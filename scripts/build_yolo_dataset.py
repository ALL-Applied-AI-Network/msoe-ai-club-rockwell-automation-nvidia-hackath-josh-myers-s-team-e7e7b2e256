"""Convert reviewed teacher detections into a YOLO image dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import re
import shutil
import tempfile
from typing import Any

import cv2
import yaml


CLASS_NAMES = ("yellow_cube", "blue_cube", "red_cube")
COLOUR_CLASS_NAMES = {
    "yellow": "yellow_cube",
    "blue": "blue_cube",
    "red": "red_cube",
}


@dataclass(frozen=True)
class SourceRun:
    """One captured image and its pseudo-labels, grouped for leakage-free splits."""

    path: Path
    image: Path
    teacher: str
    boxes: tuple[tuple[int, tuple[float, float, float, float]], ...]
    masks: tuple[tuple[int, Path], ...]


def _load_run(
    run_path: Path,
    image_name: str,
    detections_name: str,
    class_names: tuple[str, ...],
) -> SourceRun:
    run_path = run_path.resolve()
    image_path = run_path / image_name
    detections_path = run_path / detections_name
    if not run_path.is_dir():
        raise ValueError(f"run directory not found: {run_path}")
    if not image_path.is_file():
        raise ValueError(f"run image not found: {image_path}")
    if not detections_path.is_file():
        raise ValueError(f"detections file not found: {detections_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"could not read image: {image_path}")
    height, width = image.shape[:2]
    try:
        detections = json.loads(detections_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {detections_path}: {exc}") from exc
    if not isinstance(detections, dict) or not isinstance(
        detections.get("objects"), list
    ):
        raise ValueError(f"{detections_path} must contain an objects list")
    if detections.get("coordinate_convention") != "unit":
        raise ValueError(
            f"{detections_path} must use normalized coordinates "
            "(coordinate_convention: unit)"
        )

    class_ids = {name: index for index, name in enumerate(class_names)}
    boxes = []
    masks = []
    for index, obj in enumerate(detections["objects"]):
        if not isinstance(obj, dict):
            continue
        class_name = obj.get("class_name")
        category = obj.get("category")
        colour = str(obj.get("colour", "")).lower()
        if class_name is None and category == "cube":
            class_name = COLOUR_CLASS_NAMES.get(colour)
            if class_name is None and any(name.endswith("_cube") for name in class_names):
                raise ValueError(
                    f"{detections_path}: cube {index} has unsupported colour "
                    f"{obj.get('colour')!r}; review the pseudo-label"
                )
        elif class_name is None and isinstance(category, str):
            class_name = category
        if class_name not in class_ids:
            continue
        bbox = obj.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(
                f"{detections_path}: object {index} needs bbox "
                "[x_min, y_min, x_max, y_max]"
            )
        try:
            x_min, y_min, x_max, y_max = (float(value) for value in bbox)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{detections_path}: object {index} has a non-numeric bbox"
            ) from exc
        if (
            not all(math.isfinite(value) for value in (x_min, y_min, x_max, y_max))
            or not (0 <= x_min < x_max <= 1)
            or not (0 <= y_min < y_max <= 1)
        ):
            raise ValueError(
                f"{detections_path}: object {index} has an invalid normalized bbox"
            )
        boxes.append((class_ids[class_name], (x_min, y_min, x_max, y_max)))
        mask_name = obj.get("mask_path")
        if mask_name is not None:
            relative_mask = Path(mask_name)
            if relative_mask.is_absolute() or ".." in relative_mask.parts:
                raise ValueError(
                    f"{detections_path}: object {index} has an unsafe mask_path"
                )
            mask_path = detections_path.parent / relative_mask
            if not mask_path.resolve().is_relative_to(run_path):
                raise ValueError(
                    f"{detections_path}: object {index} mask resolves outside "
                    f"the source run"
                )
            if not mask_path.is_file():
                raise ValueError(
                    f"{detections_path}: object {index} mask not found: {mask_path}"
                )
            masks.append((class_ids[class_name], mask_path))

    teacher = detections.get("model", "unknown")
    if not isinstance(teacher, str):
        raise ValueError(f"{detections_path} has an invalid model name")
    return SourceRun(run_path, image_path, teacher, tuple(boxes), tuple(masks))


def _sample_id(run_index: int, run: SourceRun) -> str:
    source_name = f"{run.path.parent.name}_{run.path.name}"
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", source_name).strip("_")
    return f"run{run_index:04d}_{safe_name or 'capture'}"


def _write_preview(source: SourceRun, boxes, class_names, destination: Path) -> None:
    image = cv2.imread(str(source.image))
    if image is None:
        raise ValueError(f"could not read image: {source.image}")
    height, width = image.shape[:2]
    scale = min(1.0, 1400 / max(width, height))
    if scale < 1:
        image = cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    for class_id, (x_min, y_min, x_max, y_max) in boxes:
        left, top = round(x_min * image.shape[1]), round(y_min * image.shape[0])
        right, bottom = round(x_max * image.shape[1]), round(y_max * image.shape[0])
        color = (0, 220, 0)
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        cv2.putText(
            image,
            class_names[class_id],
            (left, max(18, top - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"could not write label preview: {destination}")


def build_dataset(
    run_paths: list[Path],
    output: Path,
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
    image_name: str = "world_camera.png",
    detections_name: str = "detections.json",
    validation_group: str | None = None,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> dict[str, Any]:
    """Build a YOLO dataset, assigning whole source runs to each split."""
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    if not run_paths:
        raise ValueError("provide at least one captured run directory")
    if (
        not class_names
        or len(set(class_names)) != len(class_names)
        or any(not name or not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in class_names)
    ):
        raise ValueError("class_names must be unique, non-empty YOLO-safe names")
    if val_fraction > 0 and len(run_paths) < 2:
        raise ValueError(
            "a validation split needs at least two distinct runs; collect more "
            "captures or explicitly use --val-fraction 0 for a training-only "
            "smoke test"
        )

    runs = [
        _load_run(path, image_name, detections_name, class_names)
        for path in run_paths
    ]
    run_names = [f"{run.path.parent.name}/{run.path.name}" for run in runs]
    if len(set(run.path for run in runs)) != len(runs):
        raise ValueError("source run directories must be unique")
    output = output.resolve()
    for run in runs:
        if output == run.path or run.path in output.parents:
            raise ValueError(f"output directory cannot be inside source run {run.path}")
    if output.exists():
        raise FileExistsError(
            f"output directory already exists: {output}; choose a new path"
        )

    if validation_group is not None:
        groups = {run.path.parent.name for run in runs}
        if validation_group not in groups:
            raise ValueError(
                f"validation group {validation_group!r} not found among "
                f"source parent directories: {sorted(groups)}"
            )
        if len(groups) < 2:
            raise ValueError("a group-held-out split needs at least two source groups")
        validation = {
            index
            for index, run in enumerate(runs)
            if run.path.parent.name == validation_group
        }
        if len(validation) == len(runs):
            raise ValueError("validation group must leave at least one training run")
    else:
        shuffled = list(range(len(runs)))
        random.Random(seed).shuffle(shuffled)
        val_count = (
            min(len(runs) - 1, max(1, round(len(runs) * val_fraction)))
            if val_fraction > 0
            else 0
        )
        validation = set(shuffled[:val_count])

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    split_runs = {"train": [], "val": []}
    split_counts = {"train": [0] * len(class_names), "val": [0] * len(class_names)}
    mask_index = []
    published = False
    try:
        for index, run in enumerate(runs):
            split = "val" if index in validation else "train"
            split_runs[split].append(run_names[index])
            sample = _sample_id(index, run)
            image_destination = staging / "images" / split / (
                sample + run.image.suffix.lower()
            )
            label_destination = staging / "labels" / split / f"{sample}.txt"
            preview_destination = staging / "previews" / split / f"{sample}.jpg"
            image_destination.parent.mkdir(parents=True, exist_ok=True)
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            preview_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run.image, image_destination)
            lines = []
            for class_id, (x_min, y_min, x_max, y_max) in run.boxes:
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                box_width = x_max - x_min
                box_height = y_max - y_min
                lines.append(
                    f"{class_id} {center_x:.8f} {center_y:.8f} "
                    f"{box_width:.8f} {box_height:.8f}"
                )
                split_counts[split][class_id] += 1
            for mask_index_in_run, (class_id, mask_source) in enumerate(run.masks):
                mask_destination = (
                    staging
                    / "masks"
                    / split
                    / f"{sample}_{mask_index_in_run:04d}.png"
                )
                mask_destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(mask_source, mask_destination)
                mask_index.append(
                    {
                        "image": image_destination.relative_to(staging).as_posix(),
                        "class_id": class_id,
                        "class_name": class_names[class_id],
                        "mask": mask_destination.relative_to(staging).as_posix(),
                    }
                )
            label_destination.write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )
            _write_preview(run, run.boxes, class_names, preview_destination)

        data = {
            "path": output.as_posix(),
            "train": "images/train",
            "val": "images/val",
            "names": list(class_names),
        }
        (staging / "data.yaml").write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=False)
        )
        manifest = {
            "teachers": sorted({run.teacher for run in runs}),
            "label_status": (
                "simulator_ground_truth"
                if {run.teacher for run in runs}
                == {"Isaac Sim ground truth projection"}
                else "pseudo_labels_require_human_review"
            ),
            "seed": seed,
            "val_fraction": val_fraction,
            "class_names": list(class_names),
            "runs": split_runs,
            "boxes_by_class": {
                split: dict(zip(class_names, counts))
                for split, counts in split_counts.items()
            },
            "instance_masks": len(mask_index),
        }
        if mask_index:
            (staging / "mask_index.json").write_text(
                json.dumps(mask_index, indent=2) + "\n"
            )
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(output)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        type=Path,
        required=True,
        help="captured run directories containing world_camera.png and detections.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validation-group",
        help="hold out all runs whose parent folder has this name",
    )
    parser.add_argument("--image-name", default="world_camera.png")
    parser.add_argument("--detections-name", default="detections.json")
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=list(CLASS_NAMES),
        help="ordered YOLO class names (default: yellow_cube blue_cube red_cube)",
    )
    args = parser.parse_args()

    try:
        manifest = build_dataset(
            args.runs,
            args.output,
            val_fraction=args.val_fraction,
            seed=args.seed,
            image_name=args.image_name,
            detections_name=args.detections_name,
            validation_group=args.validation_group,
            class_names=tuple(args.class_names),
        )
    except (FileExistsError, OSError, ValueError) as exc:
        parser.error(str(exc))
    output = args.output.resolve()
    print(f"YOLO dataset written to {output}")
    print(f"Review teacher labels in {output / 'previews'} before training.")
    print(json.dumps(manifest["boxes_by_class"], indent=2))


if __name__ == "__main__":
    main()
