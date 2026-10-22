from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from .models import CameraFrame, Detection, ModelArtifact


class YOLOModel:
    """Train and run the coarse object detector used by the controllers."""

    def __init__(
        self,
        class_names: Sequence[str],
        weights: str | Path = "yolo26n.pt",
        train_options: dict[str, Any] | None = None,
    ) -> None:
        self.class_names = tuple(class_names)
        if not self.class_names or len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names must be non-empty and unique")
        self.weights = str(weights)
        self.train_options = dict(train_options or {})
        self.artifact: ModelArtifact | None = None
        self._model = None

    def train(self, dataset_dir: Path, output_dir: Path) -> ModelArtifact:
        dataset_dir = Path(dataset_dir).resolve()
        data_yaml = dataset_dir / "data.yaml"
        if not data_yaml.is_file():
            raise FileNotFoundError(f"YOLO dataset config not found: {data_yaml}")
        data = yaml.safe_load(data_yaml.read_text())
        if data.get("names") != list(self.class_names):
            raise ValueError(
                "YOLO dataset class names do not match configured model classes: "
                f"{data.get('names')!r} != {list(self.class_names)!r}"
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is required to train the detector; install the "
                "project's optional yolo dependencies"
            ) from exc

        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        model = YOLO(self.weights)
        options = {
            "data": str(data_yaml),
            "project": str(output_dir),
            "name": "yolo26n",
            **self.train_options,
        }
        results = model.train(**options)
        best_weights = Path(model.trainer.best).resolve()
        if not best_weights.is_file():
            raise FileNotFoundError(
                f"Ultralytics training did not produce best weights: {best_weights}"
            )
        metrics = {
            str(key): float(value)
            for key, value in getattr(results, "results_dict", {}).items()
            if isinstance(value, (int, float))
        }
        artifact = ModelArtifact(
            name="YOLO26n",
            version=best_weights.parent.parent.name,
            weights_path=best_weights,
            class_names=self.class_names,
            metrics=metrics,
        )
        self.load(artifact)
        return artifact

    def load(self, artifact: ModelArtifact) -> None:
        if artifact.class_names != self.class_names:
            raise ValueError("model artifact classes do not match this YOLOModel")
        if not artifact.weights_path.is_file():
            raise FileNotFoundError(
                f"YOLO model weights not found: {artifact.weights_path}"
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is required to load the detector; install the "
                "project's optional yolo dependencies"
            ) from exc
        self.artifact = artifact
        self._model = YOLO(str(artifact.weights_path))

    def detect(self, frame: CameraFrame) -> list[Detection]:
        if self.artifact is None or self._model is None:
            raise RuntimeError("YOLOModel must be trained or loaded before detection")
        predictions = self._model.predict(
            source=str(frame.image_path),
            verbose=False,
        )
        detections = []
        for result in predictions:
            boxes = result.boxes
            if boxes is None:
                continue
            xyxy = boxes.xyxy.detach().cpu().tolist()
            confidences = boxes.conf.detach().cpu().tolist()
            class_ids = boxes.cls.detach().cpu().tolist()
            for bbox, confidence, class_id in zip(xyxy, confidences, class_ids):
                class_index = int(class_id)
                if not 0 <= class_index < len(self.class_names):
                    raise ValueError(
                        f"YOLO prediction has unknown class id {class_index}"
                    )
                detections.append(
                    Detection(
                        class_name=self.class_names[class_index],
                        confidence=float(confidence),
                        bbox_xyxy=tuple(float(value) for value in bbox),
                        source_camera=frame.camera_id,
                    )
                )
        return detections
