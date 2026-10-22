"""Automated pipeline to generate focused lightweight student models for specific parts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Sequence

from .models import ModelArtifact, SceneSpec, SpawnedObject
from .SceneGenerator import (
    ParsedObjectRequest,
    ScenePromptParser,
    TextToSceneSpawner,
)
from .YOLOModel import YOLOModel
from scripts.build_yolo_dataset import build_dataset
from scripts.project_sim_ground_truth import project_ground_truth


@dataclass(frozen=True)
class TargetPartSpec:
    prompt: str
    category: str
    color: str | None
    state_aware: bool
    class_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "category": self.category,
            "color": self.color,
            "state_aware": self.state_aware,
            "class_names": list(self.class_names),
        }


class PartModelFactory:
    """End-to-end factory: Text Prompt -> Scenes -> Distilled Labels -> YOLO Student."""

    def __init__(
        self,
        base_weights: str | Path = "weights/yolo26n.pt",
        spawner: TextToSceneSpawner | None = None,
    ) -> None:
        weights_path = Path(base_weights)
        if not weights_path.is_file():
            for candidate in ("weights/yolo26n.pt", "models/yolo26n.pt", "yolo26n.pt"):
                if Path(candidate).is_file():
                    weights_path = Path(candidate)
                    break
        self.base_weights = str(weights_path)
        self.spawner = spawner or TextToSceneSpawner()

    def parse_target_part(self, prompt: str, state_aware: bool = False) -> TargetPartSpec:
        """Resolve the target part schema and class labels from a text prompt."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty")

        clean = prompt.lower().strip()
        category = None
        if re.search(r"\b(cube|cubes|block|blocks)\b", clean):
            category = "cube"
        elif re.search(r"\b(vial|vials|tube|tubes|bottle|bottles)\b", clean):
            category = "vial"
        elif re.search(r"\b(cap|caps|puck|pucks)\b", clean):
            category = "cap"
        else:
            category = re.sub(r"[^a-zA-Z0-9_]+", "_", clean).strip("_")

        color = None
        for candidate in ("yellow", "blue", "red", "green", "white", "clear"):
            if re.search(r"\b" + candidate + r"\b", clean):
                color = candidate
                break

        if category == "cap" and not color:
            color = "yellow"
        elif category == "vial" and not color:
            color = "clear"

        # Determine class names for student training
        if state_aware and category == "cap":
            class_names = ("cap_narrow_end_up", "cap_wide_end_up")
        elif category == "cube":
            if color:
                class_names = (f"{color}_cube",)
            else:
                class_names = ("yellow_cube", "blue_cube", "red_cube")
        elif category == "vial":
            class_names = ("vial",)
        elif category == "cap":
            class_names = ("cap",)
        else:
            class_names = (category,)

        return TargetPartSpec(
            prompt=prompt,
            category=category,
            color=color,
            state_aware=state_aware,
            class_names=class_names,
        )

    def generate_scenes(
        self,
        target_part: TargetPartSpec,
        num_scenes: int = 10,
        seed: int = 42,
        add_distractors: bool = True,
    ) -> list[SceneSpec]:
        """Generate diverse digital twin scene layouts focused on the target part."""
        rng = random.Random(seed)
        scenes: list[SceneSpec] = []

        for scene_idx in range(num_scenes):
            scene_seed = seed + scene_idx * 17
            scene_rng = random.Random(scene_seed)
            requests: list[ParsedObjectRequest] = []

            # 1 to 4 instances of target part
            target_count = scene_rng.randint(1, 4)
            cat = target_part.category

            if cat == "cap":
                for _ in range(target_count):
                    orientation = (
                        "narrow_end_up" if scene_rng.random() > 0.4 else "narrow_end_down"
                    )
                    requests.append(
                        ParsedObjectRequest(
                            category="cap",
                            count=1,
                            color=target_part.color or "yellow",
                            region="scattered",
                            orientation=orientation,
                        )
                    )
            elif cat == "cube":
                requests.append(
                    ParsedObjectRequest(
                        category="cube",
                        count=target_count,
                        color=target_part.color or "yellow",
                        region="scattered",
                    )
                )
            elif cat == "vial":
                requests.append(
                    ParsedObjectRequest(
                        category="vial",
                        count=target_count,
                        region="scattered",
                    )
                )
            else:
                requests.append(
                    ParsedObjectRequest(
                        category=cat,
                        count=target_count,
                        color=target_part.color,
                        region="scattered",
                    )
                )

            # Optional distractors (other categories)
            if add_distractors:
                distractor_cats = [c for c in ("cube", "vial", "cap") if c != cat]
                if distractor_cats:
                    d_cat = scene_rng.choice(distractor_cats)
                    d_color = "red" if d_cat == "cube" else None
                    requests.append(
                        ParsedObjectRequest(
                            category=d_cat,
                            count=scene_rng.randint(1, 2),
                            color=d_color,
                            region="scattered",
                        )
                    )

            objects = self.spawner.solver.solve(requests, seed=scene_seed)
            scenes.append(
                SceneSpec(
                    prompt=f"{target_part.prompt} variation {scene_idx + 1}",
                    objects=objects,
                    seed=scene_seed,
                    metadata={"target_part": target_part.to_dict()},
                )
            )

        return scenes

    def distill_labels(
        self,
        run_paths: Sequence[Path | str],
        target_part: TargetPartSpec,
        output_name: str = "part_detections.json",
        overwrite: bool = True,
    ) -> list[Path]:
        """Extract and map teacher detections strictly to target part classes."""
        saved_paths: list[Path] = []

        for raw_run in run_paths:
            run_path = Path(raw_run).resolve()
            dest_file = run_path / output_name
            if dest_file.exists() and not overwrite:
                raise FileExistsError(f"Output label file already exists: {dest_file}")

            # Try loading projected ground truth or Cosmos detections
            detections_data = None
            sim_gt = run_path / "sim_ground_truth_all_detections.json"
            cosmos_det = run_path / "detections.json"

            if sim_gt.is_file():
                try:
                    detections_data = json.loads(sim_gt.read_text())
                except Exception:
                    pass
            elif cosmos_det.is_file():
                try:
                    detections_data = json.loads(cosmos_det.read_text())
                except Exception:
                    pass

            if detections_data is None:
                # Compute projection dynamically from scene & camera
                detections_data = project_ground_truth(run_path)

            raw_objects = detections_data.get("objects", [])

            # Also check ground_truth.json for extra orientation metadata if available
            gt_file = run_path / "ground_truth.json"
            gt_objects_by_name = {}
            if gt_file.is_file():
                try:
                    gt_info = json.loads(gt_file.read_text())
                    for obj in gt_info.get("objects", []):
                        if "name" in obj:
                            gt_objects_by_name[obj["name"]] = obj
                except Exception:
                    pass

            filtered_objects = []
            for obj in raw_objects:
                obj_cat = obj.get("category")
                if obj_cat != target_part.category:
                    continue

                obj_colour = str(obj.get("colour", "")).lower()
                if (
                    target_part.category == "cube"
                    and target_part.color
                    and obj_colour != target_part.color
                ):
                    continue

                # State-aware class mapping (distilling Cosmos reasoning into YOLO classes)
                if target_part.state_aware and target_part.category == "cap":
                    obj_name = obj.get("name", "")
                    gt_match = gt_objects_by_name.get(obj_name, {})
                    orientation = obj.get(
                        "orientation", gt_match.get("orientation", "narrow_end_up")
                    )
                    if orientation in ("narrow_end_down", "wide_end_up"):
                        mapped_class = "cap_wide_end_up"
                    else:
                        mapped_class = "cap_narrow_end_up"
                else:
                    if target_part.category == "cube" and target_part.color:
                        mapped_class = f"{target_part.color}_cube"
                    elif target_part.category == "cube":
                        mapped_class = f"{obj_colour}_cube" if obj_colour else "cube"
                    else:
                        mapped_class = target_part.category

                if mapped_class not in target_part.class_names:
                    continue

                item = dict(obj)
                item["class_name"] = mapped_class
                filtered_objects.append(item)

            distilled = {
                "coordinate_convention": "unit",
                "model": f"PartModelFactory teacher for {target_part.prompt}",
                "task": "object_detection",
                "target_part": target_part.to_dict(),
                "objects": filtered_objects,
            }

            dest_file.write_text(json.dumps(distilled, indent=2) + "\n")
            saved_paths.append(dest_file)

        return saved_paths

    def build_part_dataset(
        self,
        run_paths: Sequence[Path | str],
        target_part: TargetPartSpec,
        output_dir: Path | str,
        detections_name: str = "part_detections.json",
        val_fraction: float = 0.2,
        validation_group: str | None = None,
        seed: int = 42,
    ) -> Path:
        """Assemble a versioned YOLO dataset filtered to the target part."""
        resolved_runs = [Path(p).resolve() for p in run_paths]
        resolved_output = Path(output_dir).resolve()
        build_dataset(
            run_paths=resolved_runs,
            output=resolved_output,
            detections_name=detections_name,
            class_names=target_part.class_names,
            val_fraction=val_fraction,
            validation_group=validation_group,
            seed=seed,
        )
        return resolved_output

    def train_student(
        self,
        dataset_dir: Path | str,
        target_part: TargetPartSpec,
        output_dir: Path | str,
        train_options: dict[str, Any] | None = None,
    ) -> ModelArtifact:
        """Train a lightweight YOLO student model on the distilled dataset."""
        model = YOLOModel(
            class_names=target_part.class_names,
            weights=self.base_weights,
            train_options=train_options or {},
        )
        return model.train(Path(dataset_dir), Path(output_dir))

    def _find_candidate_frames(self, run_paths: Sequence[Path | str]) -> Optional[Path]:
        """Locate an image frame directory among the run paths for video rendering."""
        for p in run_paths:
            path = Path(p).resolve()
            frames_sub = path / "frames"
            if frames_sub.is_dir() and any(frames_sub.glob("*.png")):
                return frames_sub
            if path.is_dir() and any(path.glob("*.png")):
                return path
        return None

    def create_model(
        self,
        prompt: str,
        run_paths: Sequence[Path | str],
        output_dir: Path | str,
        state_aware: bool = False,
        skip_training: bool = False,
        train_options: dict[str, Any] | None = None,
        val_fraction: float = 0.2,
        validation_group: str | None = None,
        seed: int = 42,
        render_video: bool = True,
    ) -> tuple[TargetPartSpec, Path, ModelArtifact | None]:
        """End-to-end: Prompt + Captures -> Distilled Dataset -> Trained Student Model."""
        target_part = self.parse_target_part(prompt, state_aware=state_aware)
        resolved_out = Path(output_dir).resolve()
        dataset_dir = resolved_out / "dataset"
        models_dir = resolved_out / "models"

        # 1. Distill teacher detections for the target part
        self.distill_labels(run_paths, target_part, output_name="part_detections.json")

        # 2. Build the focused YOLO dataset
        self.build_part_dataset(
            run_paths=run_paths,
            target_part=target_part,
            output_dir=dataset_dir,
            detections_name="part_detections.json",
            val_fraction=val_fraction,
            validation_group=validation_group,
            seed=seed,
        )

        # 3. Train student model
        artifact = None
        if not skip_training:
            artifact = self.train_student(
                dataset_dir=dataset_dir,
                target_part=target_part,
                output_dir=models_dir,
                train_options=train_options,
            )

        # 4. Generate demonstration video (enabled by default)
        if render_video and artifact and artifact.weights_path:
            candidate_frames = self._find_candidate_frames(run_paths)
            if candidate_frames:
                try:
                    from scripts.render_student_video import StudentVideoRenderer
                    device = train_options.get("device") if train_options else None
                    renderer = StudentVideoRenderer(
                        weights_path=Path(artifact.weights_path),
                        device=device,
                    )
                    video_out = resolved_out / "demonstration.mp4"
                    renderer.render_video(
                        input_source=candidate_frames,
                        output_path=video_out,
                        show_hud=True,
                    )
                    if artifact.metrics is not None:
                        artifact.metrics["demonstration_video"] = str(video_out)
                except Exception as exc:
                    print(f"[PartModelFactory] Notice: Video creation skipped or failed: {exc}")

        return target_part, dataset_dir, artifact

