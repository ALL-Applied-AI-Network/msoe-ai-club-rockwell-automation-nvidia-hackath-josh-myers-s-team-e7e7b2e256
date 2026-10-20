"""Class-based architecture stubs for Problem 3 grasping and sorting."""

from .config import (
    CameraDefinition,
    camera_definition_from_yaml,
    camera_definitions,
    load_problem3_config,
)
from .core.DigitalTwinController import DigitalTwinController
from .core.DigitalTwinDataCollector import DigitalTwinDataCollector
from .core.DigitalEvaluator import DigitalEvaluator
from .core.Evaluator import Evaluator
from .core.MaskingModel import MaskingModel
from .core.models import (
    Annotation,
    ArmCameraFrame,
    CameraFrame,
    Detection,
    ModelArtifact,
    OverheadCameraFrame,
    Pose,
    TrialResult,
)
from .core.RealityController import RealityController
from .core.RealityEvaluator import RealityEvaluator
from .core.RobotController import RobotController
from .core.SAMDataLabeler import SAMDataLabeler
from .core.YOLOModel import YOLOModel
from .orchestrator import Problem3Orchestrator, RunReport, build_orchestrator

__all__ = [
    "Annotation",
    "ArmCameraFrame",
    "CameraDefinition",
    "CameraFrame",
    "Detection",
    "DigitalTwinController",
    "DigitalTwinDataCollector",
    "DigitalEvaluator",
    "Evaluator",
    "MaskingModel",
    "ModelArtifact",
    "OverheadCameraFrame",
    "Pose",
    "Problem3Orchestrator",
    "RealityController",
    "RealityEvaluator",
    "RobotController",
    "RunReport",
    "SAMDataLabeler",
    "TrialResult",
    "YOLOModel",
    "build_orchestrator",
    "camera_definition_from_yaml",
    "camera_definitions",
    "load_problem3_config",
]
