from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import sleep

from .config import camera_definitions, load_problem3_config
from .core.DigitalTwinController import DigitalTwinController
from .core.DigitalTwinDataCollector import DigitalTwinDataCollector
from .core.Evaluator import DigitalEvaluator, RealityEvaluator
from .core.MaskingModel import MaskingModel
from .core.models import (
    ArmCameraFrame,
    ModelArtifact,
    OverheadCameraFrame,
    Pose,
    TrialResult,
)
from .core.RealityController import RealityController
from .core.RobotController import RobotController
from .core.SAMDataLabeler import SAMDataLabeler
from .core.YOLOModel import YOLOModel


@dataclass
class RunReport:
    completed: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)


class Problem3Orchestrator:
    def __init__(self, config_path: Path, robot_controller: RobotController) -> None:
        self.config_path = config_path
        self.config = load_problem3_config(config_path)
        self.output_dir = Path(self.config["data"].get("directory", "outputs/problem3"))
        self.camera_definitions = camera_definitions(self.config)
        self.class_names = self._class_names()

        self.data_collector = DigitalTwinDataCollector.from_config(None, config_path)
        self.sam_labeler = SAMDataLabeler(None, self.class_names)
        self.yolo_model = YOLOModel(self.class_names)
        self.masking_model = MaskingModel(None)
        self.robot_controller = robot_controller
        self.digital_evaluator = DigitalEvaluator()
        self.reality_evaluator = RealityEvaluator()
        self.digital_twin_controller = DigitalTwinController(None, self.robot_controller)
        self.reality_controller = RealityController(
            self.yolo_model,
            self.masking_model,
            self.robot_controller,
            self.reality_evaluator,
        )

    def _class_names(self) -> tuple[str, ...]:
        names = self.config.get("model", {}).get("class_names")
        if not isinstance(names, list) or not names:
            raise ValueError("Config must define model.class_names as a non-empty list")
        return tuple(str(name) for name in names)

    def run_data_generation(self, scenario_ids: list[str]) -> Path:
        frames = self.data_collector.collect(scenario_ids, self.output_dir / "renders")
        labels = self.sam_labeler.label(frames)
        return self.sam_labeler.export_yolo(frames, labels, self.output_dir / "dataset")

    def train_detector(self, dataset_dir: Path | None = None) -> ModelArtifact:
        return self.yolo_model.train(
            dataset_dir or self.output_dir / "dataset",
            self.output_dir / "models",
        )

    def run_simulation(self, final_pose: Pose) -> TrialResult:
        trial = self.digital_twin_controller.perform_action(final_pose)
        self.digital_evaluator.evaluate(trial)
        return trial

    def run_reality(
        self,
        overhead_frame: OverheadCameraFrame,
        arm_frame: ArmCameraFrame,
    ) -> list[TrialResult]:
        trials = self.reality_controller.process_frame(overhead_frame, arm_frame)
        for trial in trials:
            self.reality_evaluator.evaluate(trial)
        return trials

    def run_loop(
        self,
        frame_source: Callable[[], tuple[OverheadCameraFrame, ArmCameraFrame]],
        max_cycles: int | None = None,
        cycle_delay_s: float = 0.0,
        stop_requested: Callable[[], bool] | None = None,
    ) -> list[TrialResult]:
        """Process synchronized camera frames repeatedly until stopped."""
        results: list[TrialResult] = []
        cycle = 0
        while max_cycles is None or cycle < max_cycles:
            if stop_requested and stop_requested():
                break
            overhead_frame, arm_frame = frame_source()
            results.extend(self.run_reality(overhead_frame, arm_frame))
            cycle += 1
            if cycle_delay_s > 0:
                sleep(cycle_delay_s)
        return results

    def run_all(self, scenario_ids: list[str]) -> RunReport:
        report = RunReport()
        stages = (
            ("data_generation", lambda: self.run_data_generation(scenario_ids)),
            ("yolo_training", self.train_detector),
        )
        for name, stage in stages:
            try:
                stage()
            except NotImplementedError:
                report.blocked.append(name)
                break
            report.completed.append(name)
        return report


def build_orchestrator(
    robot_controller: RobotController,
    config_path: str | Path = "configs/problem3_grasping_sorting.yaml",
) -> Problem3Orchestrator:
    return Problem3Orchestrator(Path(config_path), robot_controller)
