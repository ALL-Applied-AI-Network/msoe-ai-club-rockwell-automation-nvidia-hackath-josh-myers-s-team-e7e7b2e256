from typing import Any

from .Evaluator import DigitalEvaluator
from .models import Pose, TrialResult
from .RobotController import RobotController


class DigitalTwinController(DigitalEvaluator):
    """Run grasp trials against the digital twin and score their outcomes."""

    def __init__(self, twin: Any, robot: RobotController) -> None:
        self.twin = twin
        self.robot = robot

    def perform_action(self, final_pose: Pose) -> TrialResult:
        """Command the simulated robot, then capture the simulated outcome."""
        raise NotImplementedError

