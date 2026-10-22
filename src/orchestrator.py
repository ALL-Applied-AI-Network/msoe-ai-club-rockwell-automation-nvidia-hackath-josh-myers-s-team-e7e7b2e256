from collections.abc import Callable
from dataclasses import dataclass, field
import json
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
from scripts.build_yolo_dataset import build_dataset
from scripts.project_sim_ground_truth import project_ground_truth


@dataclass
class RunReport:
    completed: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)


class Problem3Orchestrator:
    def __init__(
        self,
        config_path: Path,
        robot_controller: RobotController | None = None,
        sam3_model=None,
    ) -> None:
        self.config_path = config_path.resolve()
        self.config = load_problem3_config(config_path)
        self.output_dir = Path(self.config["data"].get("directory", "outputs/problem3"))
        self.camera_definitions = camera_definitions(self.config)
        self.class_names = self._class_names()

        self.data_collector = DigitalTwinDataCollector.from_config(
            None, self.config_path
        )
        self.sam_labeler = SAMDataLabeler(sam3_model, self.class_names)
        model_config = self.config.get("model", {})
        training_config = self.config.get("training", {})
        self.yolo_model = YOLOModel(
            self.class_names,
            weights=model_config.get("pretrained_weights", "yolo26n.pt"),
            train_options={
                "epochs": int(
                    model_config.get("epochs", training_config.get("epochs", 100))
                ),
                "batch": int(
                    model_config.get(
                        "batch_size", training_config.get("batch_size", 4)
                    )
                ),
                "lr0": float(training_config.get("learning_rate", 0.001)),
                "patience": int(training_config.get("early_stopping_patience", 10)),
                "imgsz": int(model_config.get("image_size", 1280)),
            },
        )
        self.masking_model = MaskingModel(None)
        self.robot_controller = robot_controller
        self.digital_evaluator = DigitalEvaluator()
        self.reality_evaluator = RealityEvaluator()
        self.digital_twin_controller = (
            DigitalTwinController(None, robot_controller)
            if robot_controller is not None
            else None
        )
        self.reality_controller = (
            RealityController(
                self.yolo_model,
                self.masking_model,
                robot_controller,
                self.reality_evaluator,
            )
            if robot_controller is not None
            else None
        )

    def _class_names(self) -> tuple[str, ...]:
        names = self.config.get("model", {}).get("class_names")
        if not isinstance(names, list) or not names:
            raise ValueError("Config must define model.class_names as a non-empty list")
        return tuple(str(name) for name in names)

    def _validation_group(self, groups: set[str]) -> str | None:
        configured_group = self.config.get("data", {}).get("validation_group")
        if configured_group in groups:
            return configured_group
        if {"train", "val"} <= groups:
            return "val"
        if configured_group is not None and len(groups) > 1:
            raise ValueError(
                f"configured validation group {configured_group!r} not found among "
                f"capture groups {sorted(groups)}; pass --validation-group to select "
                "an available held-out group"
            )
        return None

    def run_data_generation(self, scenario_ids: list[str]) -> Path:
        frames = self.data_collector.collect(
            scenario_ids, self.output_dir / "renders"
        )
        source = self.config.get("data", {}).get(
            "annotation_source", "simulator_ground_truth"
        )
        dataset_dir = self.output_dir / "dataset"
        if source == "simulator_ground_truth":
            run_paths = [Path(frame.metadata["source_run"]) for frame in frames]
            detections_name = "sim_ground_truth_all_detections.json"
            for run in run_paths:
                labels_path = run / detections_name
                labels = project_ground_truth(run)
                serialized_labels = json.dumps(labels, indent=2) + "\n"
                if labels_path.exists():
                    if labels_path.read_text() != serialized_labels:
                        raise FileExistsError(
                            f"existing simulator labels differ; refusing to overwrite "
                            f"{labels_path}"
                        )
                else:
                    labels_path.write_text(serialized_labels)
            groups = {run.parent.name for run in run_paths}
            validation_group = self._validation_group(groups)
            build_dataset(
                run_paths,
                dataset_dir,
                val_fraction=float(
                    self.config.get("data", {}).get("validation_split", 0.2)
                ),
                seed=int(self.config.get("project", {}).get("seed", 42)),
                detections_name=detections_name,
                validation_group=validation_group,
                class_names=self.class_names,
            )
            return dataset_dir
        if source == "sam3":
            labels = self.sam_labeler.label(frames)
            run_paths = [
                Path(frame.metadata["source_run"]) for frame in frames
            ]
            groups = {run.parent.name for run in run_paths}
            validation_group = self._validation_group(groups)
            return self.sam_labeler.export_yolo(
                frames,
                labels,
                dataset_dir,
                validation_group=validation_group,
            )
        raise ValueError(
            f"Unsupported data.annotation_source {source!r}; expected "
            "'simulator_ground_truth' or 'sam3'"
        )

    def train_detector(self, dataset_dir: Path | None = None) -> ModelArtifact:
        return self.yolo_model.train(
            dataset_dir or self.output_dir / "dataset",
            self.output_dir / "models",
        )

    def run_simulation(self, final_pose: Pose) -> TrialResult:
        if self.digital_twin_controller is None:
            raise RuntimeError("run_simulation requires a configured robot controller")
        trial = self.digital_twin_controller.perform_action(final_pose)
        self.digital_evaluator.evaluate(trial)
        return trial

    def run_reality(
        self,
        overhead_frame: OverheadCameraFrame,
        arm_frame: ArmCameraFrame,
    ) -> list[TrialResult]:
        if self.reality_controller is None:
            raise RuntimeError("run_reality requires a configured robot controller")
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
    robot_controller: RobotController | None = None,
    config_path: str | Path = "configs/problem3_grasping_sorting.yaml",
    sam3_model=None,
) -> Problem3Orchestrator:
    return Problem3Orchestrator(Path(config_path), robot_controller, sam3_model)
