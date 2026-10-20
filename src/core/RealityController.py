from .Evaluator import RealityEvaluator
from .MaskingModel import MaskingModel
from .models import ArmCameraFrame, OverheadCameraFrame, TrialResult
from .RobotController import RobotController
from .YOLOModel import YOLOModel


class RealityController:
    """Run the deployed physical-robot loop; no simulation assumptions leak here."""

    def __init__(
        self,
        yolo: YOLOModel,
        masking: MaskingModel,
        robot: RobotController,
        evaluator: RealityEvaluator,
    ) -> None:
        self.yolo = yolo
        self.masking = masking
        self.robot = robot
        self.evaluator = evaluator

    def process_frame(
        self,
        overhead_frame: OverheadCameraFrame,
        arm_frame: ArmCameraFrame,
    ) -> list[TrialResult]:
        """Detect overhead, refine with the arm camera, execute, and record results."""
        raise NotImplementedError
