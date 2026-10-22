from collections.abc import Sequence
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .models import Annotation, CameraFrame
from scripts.build_yolo_dataset import build_dataset
from scripts.label_with_sam3 import _as_numpy, detections_from_masks


class SAMDataLabeler:
    """Create segmentation labels from digital-twin frames using SAM3."""

    def __init__(self, sam3_model: Any, class_names: Sequence[str]) -> None:
        self.sam3_model = sam3_model
        self.class_names = tuple(class_names)
        if not self.class_names or len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names must be non-empty and unique")

    def label(self, frames: Sequence[CameraFrame]) -> list[list[Annotation]]:
        """Return one annotation list per frame."""
        if self.sam3_model is None:
            raise RuntimeError(
                "SAMDataLabeler requires an initialized SAM 3 processor; "
                "load the gated checkpoint and inject its processor"
            )
        if not frames:
            raise ValueError("provide at least one frame to label")

        results: list[list[Annotation]] = []
        for frame in frames:
            with Image.open(frame.image_path) as image:
                rgb_image = image.convert("RGB")
            width, height = rgb_image.size
            state = self.sam3_model.set_image(rgb_image)
            frame_annotations = []
            for class_name in self.class_names:
                prompt = class_name.removesuffix("_cube").replace("_", " ")
                category = "cube" if class_name.endswith("_cube") else class_name
                colour = (
                    class_name.removesuffix("_cube")
                    if category == "cube"
                    else None
                )
                prompt = f"{prompt} cube" if category == "cube" else prompt
                prediction = self.sam3_model.set_text_prompt(
                    prompt=prompt,
                    state=state,
                )
                detections = detections_from_masks(
                    prediction["masks"],
                    prediction["scores"],
                    class_name=class_name,
                    category=category,
                    colour=colour,
                    width=width,
                    height=height,
                    threshold=0.0,
                )
                masks = _as_numpy(prediction["masks"])
                if masks.ndim == 4 and masks.shape[1] == 1:
                    masks = masks[:, 0]
                for detection in detections:
                    mask_index = detection.pop("_mask_index")
                    frame_annotations.append(
                        Annotation(
                            class_name=class_name,
                            bbox_xyxy=tuple(detection["bbox"]),
                            confidence=detection["score"],
                            source="sam3",
                            segmentation_mask=np.asarray(masks[mask_index] > 0),
                        )
                    )
            results.append(frame_annotations)
        return results

    def export_yolo(
        self,
        frames: Sequence[CameraFrame],
        labels: Sequence[Sequence[Annotation]],
        output_dir: Path,
        validation_group: str | None = None,
    ) -> Path:
        """Convert SAM3 masks/bounding boxes into a versioned YOLO dataset."""
        if len(frames) != len(labels):
            raise ValueError("frames and labels must have the same length")
        if not frames:
            raise ValueError("provide at least one frame to export")
        output_dir = Path(output_dir).resolve()
        if output_dir.exists():
            raise FileExistsError(f"dataset output already exists: {output_dir}")
        output_dir.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(
            prefix=f".{output_dir.name}.labels-",
            dir=output_dir.parent,
        ) as temporary_directory:
            staging = Path(temporary_directory)
            run_paths = []
            for index, (frame, annotations) in enumerate(zip(frames, labels)):
                split = frame.metadata.get("split_group")
                group = (
                    split
                    if isinstance(split, str) and Path(split).name == split
                    else "captures"
                )
                run = staging / "runs" / group / f"frame_{index:06d}"
                run.mkdir(parents=True)
                image_destination = run / "world_camera.png"
                shutil.copy2(frame.image_path, image_destination)
                with Image.open(frame.image_path) as source_image:
                    image_width, image_height = source_image.size
                objects = []
                for mask_index, annotation in enumerate(annotations):
                    detection = {
                        "class_name": annotation.class_name,
                        "category": (
                            "cube"
                            if annotation.class_name.endswith("_cube")
                            else annotation.class_name
                        ),
                        "bbox": list(annotation.bbox_xyxy),
                        "score": annotation.confidence,
                    }
                    if annotation.class_name.endswith("_cube"):
                        detection["colour"] = annotation.class_name.removesuffix(
                            "_cube"
                        )
                    if annotation.segmentation_mask is not None:
                        mask_directory = run / "sam3_masks"
                        mask_directory.mkdir(exist_ok=True)
                        mask_path = mask_directory / f"mask_{mask_index:05d}.png"
                        mask = np.asarray(annotation.segmentation_mask, dtype=np.uint8)
                        if mask.ndim != 2:
                            raise ValueError(
                                f"mask for {annotation.class_name!r} must be 2D"
                            )
                        if mask.shape != (image_height, image_width):
                            raise ValueError(
                                f"mask for {annotation.class_name!r} does not match "
                                "its source image dimensions"
                            )
                        if not cv2.imwrite(str(mask_path), mask * 255):
                            raise OSError(f"could not write segmentation mask: {mask_path}")
                        detection["mask_path"] = f"sam3_masks/{mask_path.name}"
                    objects.append(detection)
                (run / "sam3_detections.json").write_text(
                    json.dumps(
                        {
                            "model": "facebook/sam3.1",
                            "coordinate_convention": "unit",
                            "objects": objects,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                run_paths.append(run)

            build_dataset(
                run_paths,
                output_dir,
                val_fraction=0.2 if len(run_paths) > 1 else 0.0,
                validation_group=validation_group,
                detections_name="sam3_detections.json",
                class_names=self.class_names,
            )
            return output_dir
