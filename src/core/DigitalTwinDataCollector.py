from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..config import camera_definitions, load_problem3_config
from .models import ArmCameraFrame, CameraFrame, OverheadCameraFrame


class DigitalTwinDataCollector:
    """Collect randomized images and metadata from the digital twin."""

    def __init__(self, twin: Any, camera_ids: Sequence[str]) -> None:
        self.twin = twin
        self.camera_ids = tuple(camera_ids)

    @classmethod
    def from_config(cls, twin: Any, config_path: Path) -> "DigitalTwinDataCollector":
        config = load_problem3_config(config_path)
        return cls(twin, [camera.camera_id for camera in camera_definitions(config)])

    def collect(self, scenario_ids: Sequence[str], output_dir: Path) -> list[CameraFrame]:
        """Render scenarios and return synchronized frames for labeling."""
        raise NotImplementedError

    def frame_from_camera_config(
        self,
        config_path: Path,
        camera_id: str,
        image_path: Path,
        timestamp_ns: int,
        calibration_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> CameraFrame:
        """Build the configured overhead or arm frame subclass."""
        config = load_problem3_config(config_path)
        definitions = {camera.camera_id: camera for camera in camera_definitions(config)}
        try:
            definition = definitions[camera_id]
        except KeyError as exc:
            raise ValueError(f"Camera {camera_id!r} is not defined in {config_path}") from exc
        frame_type = (
            OverheadCameraFrame if definition.placement == "overhead" else ArmCameraFrame
        )
        return frame_type.from_yaml(
            config_path,
            camera_id,
            image_path,
            timestamp_ns,
            calibration_id,
            metadata,
        )
