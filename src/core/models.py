"""Domain objects shared by the Problem 3 pipeline stages."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar


@dataclass(frozen=True)
class CameraFrame:
    camera_id: str
    image_path: Path
    timestamp_ns: int
    calibration_id: str
    role: str
    placement: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(
        cls,
        config_path: Path,
        camera_id: str,
        image_path: Path,
        timestamp_ns: int,
        calibration_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> "CameraFrame":
        """Construct a frame using role/placement values from the YAML file."""
        from ..config import camera_definition_from_yaml

        definition = camera_definition_from_yaml(config_path, camera_id)
        expected_placement = getattr(cls, "expected_placement", None)
        if expected_placement and definition.placement != expected_placement:
            raise ValueError(
                f"Camera {camera_id!r} has placement {definition.placement!r}; "
                f"{cls.__name__} requires {expected_placement!r}"
            )
        return cls(
            camera_id=camera_id,
            image_path=image_path,
            timestamp_ns=timestamp_ns,
            calibration_id=calibration_id,
            role=definition.role,
            placement=definition.placement,
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class OverheadCameraFrame(CameraFrame):
    """Frame captured by one of the configured workspace overhead cameras."""

    expected_placement: ClassVar[str] = "overhead"


@dataclass(frozen=True)
class ArmCameraFrame(CameraFrame):
    """Frame captured by the camera mounted on the robot arm/wrist."""

    expected_placement: ClassVar[str] = "robot_wrist"


@dataclass(frozen=True)
class Annotation:
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]
    segmentation: list[tuple[float, float]] | None = None
    confidence: float = 1.0
    source: str = "synthetic_sam3"


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    source_camera: str


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    frame_id: str = "world"


@dataclass(frozen=True)
class ModelArtifact:
    name: str
    version: str
    weights_path: Path
    class_names: tuple[str, ...]
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TrialResult:
    trial_id: str
    success: bool
    placement_error_m: float | None
    cycle_time_s: float | None
    recovery_used: bool = False
    notes: str = ""
