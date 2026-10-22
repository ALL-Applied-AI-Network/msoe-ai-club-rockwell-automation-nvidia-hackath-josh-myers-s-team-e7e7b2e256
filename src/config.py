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
    """Load base and task YAML settings, then validate the camera definitions."""
    path = path.resolve()

    def load(path: Path, visited: set[Path]) -> dict[str, Any]:
        if path in visited:
            raise ValueError(f"Config inheritance cycle detected at {path}")
        with path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
        if not isinstance(config, dict):
            raise TypeError(f"Config root must be a mapping: {path}")
        base_config = config.pop("base_config", None)
        if base_config is None:
            return config
        if not isinstance(base_config, str) or not base_config:
            raise TypeError(f"base_config must be a non-empty path in {path}")
        base_path = (path.parent / base_config).resolve()
        base = load(base_path, visited | {path})
        return _merge_config(base, config)

    config = load(path, set())
    try:
        views = config["components"]["cameras"]["views"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing components.cameras.views in {path}") from exc
    if not isinstance(views, list):
        raise TypeError("components.cameras.views must be a list")
    return config


def _merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


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
