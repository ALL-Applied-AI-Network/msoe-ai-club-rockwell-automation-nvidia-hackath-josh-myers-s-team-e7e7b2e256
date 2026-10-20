from typing import Any

from .models import CameraFrame, Detection, Pose


class MaskingModel:
    """Refine a YOLO detection into an object mask and final grasp point."""

    def __init__(self, segmentation_model: Any) -> None:
        self.segmentation_model = segmentation_model

    def refine(self, frame: CameraFrame, detection: Detection) -> Pose:
        """Estimate the final world-frame pose to pass to RobotController."""
        raise NotImplementedError
