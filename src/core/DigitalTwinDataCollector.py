from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from ..config import camera_definitions, load_problem3_config
from .models import (
    ArmCameraFrame,
    CameraFrame,
    OverheadCameraFrame,
    SideCameraFrame,
)


class DigitalTwinDataCollector:
    """Collect randomized images and metadata from the digital twin."""

    def __init__(
        self,
        twin: Any,
        camera_ids: Sequence[str],
        config_path: Path | None = None,
    ) -> None:
        self.twin = twin
        self.camera_ids = tuple(camera_ids)
        self.config_path = config_path

    @classmethod
    def from_config(cls, twin: Any, config_path: Path) -> "DigitalTwinDataCollector":
        config = load_problem3_config(config_path)
        return cls(
            twin,
            [camera.camera_id for camera in camera_definitions(config)],
            config_path,
        )

    def collect(self, scenario_ids: Sequence[str], output_dir: Path) -> list[CameraFrame]:
        """Load Isaac-rendered scenario folders as configured camera frames.

        ``scenario_ids`` are paths to capture folders containing the saved image
        and camera calibration. Rendering remains the responsibility of the
        simulator-specific capture command.
        """
        if not scenario_ids:
            raise ValueError("provide at least one captured scenario directory")
        if self.config_path is None:
            raise ValueError("collector requires a Problem 3 config path")
        return self.collect_captured_runs(
            scenario_ids, output_dir, self.config_path
        )

    def collect_captured_runs(
        self,
        scenario_paths: Sequence[str | Path],
        output_dir: Path,
        config_path: Path,
    ) -> list[CameraFrame]:
        """Load existing simulator captures and persist a frame manifest."""
        if not scenario_paths:
            raise ValueError("provide at least one captured scenario directory")
        config = load_problem3_config(config_path)
        definitions = camera_definitions(config)
        overhead_ids = [
            camera.camera_id
            for camera in definitions
            if camera.placement == "overhead"
        ]
        wrist_ids = [
            camera.camera_id
            for camera in definitions
            if camera.placement == "robot_wrist"
        ]
        side_ids = [
            camera.camera_id
            for camera in definitions
            if camera.placement == "side"
        ]
        frames = []
        for scenario_path in scenario_paths:
            run = Path(scenario_path).resolve()
            image_path = run / "world_camera.png"
            camera_path = run / "camera.json"
            if not image_path.is_file() or not camera_path.is_file():
                raise FileNotFoundError(
                    f"{run} must contain world_camera.png and camera.json"
                )
            camera_data = json.loads(camera_path.read_text())
            captured_camera = camera_data.get("camera")
            if captured_camera == "wrist":
                candidates = wrist_ids
                camera_index = 0
            elif captured_camera == "side":
                candidates = side_ids
                camera_index = 0
            elif captured_camera == "table":
                candidates = overhead_ids
                camera_index = 0
            else:
                raise ValueError(
                    f"{camera_path} has unsupported camera {captured_camera!r}"
                )
            if not candidates:
                raise ValueError(
                    f"Config {config_path} has no camera for {captured_camera!r}"
                )
            camera_id = candidates[min(camera_index, len(candidates) - 1)]
            calibration_id = hashlib.sha256(camera_path.read_bytes()).hexdigest()
            split_group = run.parent.name
            metadata = {
                "source_run": str(run),
                "captured_camera": captured_camera,
                "split_group": split_group,
                "capture_domain": camera_data.get("capture_domain", {}),
            }
            frames.append(
                self.frame_from_camera_config(
                    config_path,
                    camera_id,
                    image_path,
                    image_path.stat().st_mtime_ns,
                    calibration_id,
                    metadata,
                )
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = [
            {
                "camera_id": frame.camera_id,
                "image_path": str(frame.image_path),
                "timestamp_ns": frame.timestamp_ns,
                "calibration_id": frame.calibration_id,
                "metadata": frame.metadata,
            }
            for frame in frames
        ]
        (output_dir / "frame_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        return frames

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
        frame_types = {
            "overhead": OverheadCameraFrame,
            "side": SideCameraFrame,
            "robot_wrist": ArmCameraFrame,
        }
        try:
            frame_type = frame_types[definition.placement]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported camera placement {definition.placement!r} "
                f"for {camera_id!r}"
            ) from exc
        return frame_type.from_yaml(
            config_path,
            camera_id,
            image_path,
            timestamp_ns,
            calibration_id,
            metadata,
        )
