"""Core domain models, controllers, and generators for Problem 3."""

from .models import (
    Annotation,
    ArmCameraFrame,
    CameraFrame,
    Detection,
    ModelArtifact,
    OverheadCameraFrame,
    Pose,
    SceneSpec,
    SideCameraFrame,
    SpawnedObject,
    TrialResult,
)
from .SceneGenerator import (
    ParsedObjectRequest,
    SceneLayoutSolver,
    ScenePromptParser,
    TextToSceneSpawner,
)
from .PartModelFactory import (
    PartModelFactory,
    TargetPartSpec,
)

__all__ = [
    "Annotation",
    "ArmCameraFrame",
    "CameraFrame",
    "Detection",
    "ModelArtifact",
    "OverheadCameraFrame",
    "ParsedObjectRequest",
    "PartModelFactory",
    "Pose",
    "SceneLayoutSolver",
    "ScenePromptParser",
    "SceneSpec",
    "SideCameraFrame",
    "SpawnedObject",
    "TargetPartSpec",
    "TextToSceneSpawner",
    "TrialResult",
]

