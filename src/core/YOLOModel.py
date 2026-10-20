from collections.abc import Sequence
from pathlib import Path

from .models import CameraFrame, Detection, ModelArtifact


class YOLOModel:
    """Train and run the coarse object detector used by the controllers."""

    def __init__(self, class_names: Sequence[str]) -> None:
        self.class_names = tuple(class_names)
        self.artifact: ModelArtifact | None = None

    def train(self, dataset_dir: Path, output_dir: Path) -> ModelArtifact:
        raise NotImplementedError

    def load(self, artifact: ModelArtifact) -> None:
        self.artifact = artifact

    def detect(self, frame: CameraFrame) -> list[Detection]:
        if self.artifact is None:
            raise RuntimeError("YOLOModel must be trained or loaded before detection")
        raise NotImplementedError
