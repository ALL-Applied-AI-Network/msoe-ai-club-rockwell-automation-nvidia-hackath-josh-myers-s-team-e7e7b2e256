from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CameraDefinition:
    camera_id: str
    role: str
    placement: str


def load_problem3_config(path: Path) -> dict[str, Any]:
    """Load the complete YAML config and validate the camera section exists."""
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    try:
        views = config["components"]["cameras"]["views"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing components.cameras.views in {path}") from exc
    if not isinstance(views, list):
        raise TypeError("components.cameras.views must be a list")
    return config


def camera_definitions(config: dict[str, Any]) -> tuple[CameraDefinition, ...]:
    views = config["components"]["cameras"]["views"]
    return tuple(
        CameraDefinition(
            camera_id=view["id"],
            role=view["role"],
            placement=view["placement"],
        )
        for view in views
    )


def camera_definition_from_yaml(path: Path, camera_id: str) -> CameraDefinition:
    """Find one camera's role and placement in the Problem 3 YAML file."""
    definitions = {
        definition.camera_id: definition
        for definition in camera_definitions(load_problem3_config(path))
    }
    try:
        return definitions[camera_id]
    except KeyError as exc:
        raise ValueError(f"Camera {camera_id!r} is not defined in {path}") from exc
