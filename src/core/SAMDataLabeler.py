from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .models import Annotation, CameraFrame


class SAMDataLabeler:
    """Create segmentation labels from digital-twin frames using SAM3."""

    def __init__(self, sam3_model: Any, class_names: Sequence[str]) -> None:
        self.sam3_model = sam3_model
        self.class_names = tuple(class_names)

    def label(self, frames: Sequence[CameraFrame]) -> list[list[Annotation]]:
        """Return one annotation list per frame."""
        raise NotImplementedError

    def export_yolo(
        self,
        frames: Sequence[CameraFrame],
        labels: Sequence[Sequence[Annotation]],
        output_dir: Path,
    ) -> Path:
        """Convert SAM3 masks/bounding boxes into a versioned YOLO dataset."""
        raise NotImplementedError
