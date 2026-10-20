"""Generate reviewable cube labels with the official Meta SAM 3.1 package."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


CLASS_PROMPTS = (
    ("yellow", "yellow cube", (0, 220, 255)),
    ("blue", "blue cube", (255, 120, 0)),
    ("red", "red cube", (40, 40, 240)),
)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def detections_from_masks(
    masks: Any,
    scores: Any,
    *,
    colour: str,
    width: int,
    height: int,
    threshold: float,
) -> list[dict[str, Any]]:
    """Convert SAM instance masks to the normalized boxes used by the exporter."""
    mask_array = _as_numpy(masks)
    if mask_array.ndim == 4 and mask_array.shape[1] == 1:
        mask_array = mask_array[:, 0]
    if mask_array.ndim != 3:
        raise ValueError(f"SAM masks must have shape [N,H,W], got {mask_array.shape}")
    if mask_array.shape[1:] != (height, width):
        raise ValueError(
            "SAM mask dimensions do not match the source image: "
            f"{mask_array.shape[1:]} != {(height, width)}"
        )

    score_array = _as_numpy(scores).reshape(-1)
    if len(score_array) != len(mask_array):
        raise ValueError("SAM returned a different number of masks and scores")

    detections = []
    for mask_index, (mask, score) in enumerate(zip(mask_array, score_array)):
        confidence = float(score)
        if not np.isfinite(confidence) or confidence < threshold:
            continue
        ys, xs = np.where(mask > 0)
        if not len(xs):
            continue

        x_min, x_max = int(xs.min()), int(xs.max()) + 1
        y_min, y_max = int(ys.min()), int(ys.max()) + 1
        x1, y1, x2, y2 = (
            x_min / width,
            y_min / height,
            x_max / width,
            y_max / height,
        )
        detections.append(
            {
                "label": f"{colour} cube",
                "category": "cube",
                "colour": colour,
                "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                "bbox": [x1, y1, x2, y2],
                "score": confidence,
                "mask_area_px": int(np.count_nonzero(mask)),
                "_mask_index": mask_index,
            }
        )
    return detections


def _draw_preview(
    image: np.ndarray,
    masks: list[np.ndarray],
    detections: list[dict[str, Any]],
    output_path: Path,
) -> None:
    overlay = image.copy()
    for mask, detection in zip(masks, detections):
        colour = next(
            color for name, _, color in CLASS_PROMPTS if name == detection["colour"]
        )
        binary_mask = mask.astype(bool)
        tint = np.zeros_like(overlay)
        tint[binary_mask] = colour
        overlay = cv2.addWeighted(overlay, 1.0, tint, 0.35, 0)

        x1, y1, x2, y2 = detection["bbox"]
        height, width = image.shape[:2]
        left, top = round(x1 * width), round(y1 * height)
        right, bottom = round(x2 * width), round(y2 * height)
        cv2.rectangle(overlay, (left, top), (right, bottom), colour, 2)
        label = f'{detection["label"]} {detection["score"]:.2f}'
        cv2.putText(
            overlay,
            label,
            (left, max(18, top - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            2,
            cv2.LINE_AA,
        )

    if not cv2.imwrite(str(output_path), overlay):
        raise OSError(f"Could not write SAM preview: {output_path}")


def _load_sam3(checkpoint: Path, device: str, threshold: float):
    try:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except ImportError as exc:
        raise SystemExit(
            "The official Meta SAM 3 package is not installed in this Python "
            "environment. Follow the SAM 3.1 setup instructions in "
            "fanuc_er4ia_sim/SETUP.md."
        ) from exc

    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        device=device,
    )
    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=threshold,
    )
    return processor


def _set_text_prompt(processor, prompt: str, state, amp_dtype: Any):
    if amp_dtype is None:
        return processor.set_text_prompt(prompt=prompt, state=state)

    import torch

    if amp_dtype is not torch.float16:
        return processor.set_text_prompt(prompt=prompt, state=state)

    with torch.autocast(device_type="cuda", enabled=False):
        text_outputs = processor.model.backbone.forward_text(
            [prompt], device=processor.device
        )
    state["backbone_out"].update(text_outputs)
    if "geometric_prompt" not in state:
        state["geometric_prompt"] = processor.model._get_dummy_prompt()
    with torch.autocast(device_type="cuda", enabled=False):
        state["backbone_out"] = _cast_floating_tensors(
            state["backbone_out"], torch.float32
        )
        return processor._forward_grounding(state)


def _cast_floating_tensors(value: Any, dtype: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype) if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: _cast_floating_tensors(item, dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [_cast_floating_tensors(item, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_cast_floating_tensors(item, dtype) for item in value)
    return value


def label_run(
    run: Path,
    processor,
    *,
    image_name: str,
    output_name: str,
    preview_name: str,
    threshold: float,
    overwrite: bool,
    amp_dtype: Any = None,
) -> int:
    image_path = run / image_name
    output_path = run / output_name
    preview_path = run / preview_name
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not overwrite and (output_path.exists() or preview_path.exists()):
        raise FileExistsError(
            f"Refusing to overwrite SAM output in {run}; choose a different "
            "filename or pass --overwrite."
        )

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    height, width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    objects = []
    preview_masks = []
    if amp_dtype is None:
        amp_context = nullcontext()
    else:
        import torch

        amp_context = torch.autocast(device_type="cuda", dtype=amp_dtype)
    with amp_context:
        state = processor.set_image(Image.fromarray(image_rgb))
        for colour, prompt, _ in CLASS_PROMPTS:
            result = _set_text_prompt(processor, prompt, state, amp_dtype)
            records = detections_from_masks(
                result["masks"],
                result["scores"],
                colour=colour,
                width=width,
                height=height,
                threshold=threshold,
            )
            masks = _as_numpy(result["masks"])
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            for detection in records:
                preview_masks.append(masks[detection.pop("_mask_index")])
                objects.append(detection)

    payload = {
        "model": "facebook/sam3.1",
        "task": "cube_detection",
        "coordinate_convention": "unit",
        "label_status": "sam3_pseudo_labels_require_human_review",
        "score_threshold": threshold,
        "prompts": {colour: prompt for colour, prompt, _ in CLASS_PROMPTS},
        "objects": objects,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    _draw_preview(image_bgr, preview_masks, objects, preview_path)
    print(f"{run}: wrote {len(objects)} SAM detections to {output_path}")
    print(f"{run}: preview saved to {preview_path}")
    return len(objects)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        type=Path,
        help="captured view folders containing the source image",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="downloaded SAM 3.1 checkpoint, usually sam3.1_multiplex.pt",
    )
    parser.add_argument("--image-name", default="world_camera.png")
    parser.add_argument("--output-name", default="sam3_detections.json")
    parser.add_argument("--preview-name", default="sam3_preview.png")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="inference device; SAM 3.1 is intended for a CUDA GPU",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing SAM output JSON and preview files",
    )
    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if Path(args.output_name).name != args.output_name:
        parser.error("--output-name must be a filename within each run directory")
    if Path(args.preview_name).name != args.preview_name:
        parser.error("--preview-name must be a filename within each run directory")
    if args.output_name == args.preview_name:
        parser.error("--output-name and --preview-name must be different")
    if not args.checkpoint.is_file():
        parser.error(f"SAM 3.1 checkpoint not found: {args.checkpoint}")
    if not args.runs:
        parser.error("provide at least one run directory")

    runs = [run.resolve() for run in args.runs]
    if len(set(runs)) != len(runs):
        parser.error("--runs must contain unique directories")
    for run in runs:
        if not run.is_dir():
            parser.error(f"Run directory not found: {run}")
        if not (run / args.image_name).is_file():
            parser.error(f"Image not found: {run / args.image_name}")
        if not args.overwrite and (
            (run / args.output_name).exists() or (run / args.preview_name).exists()
        ):
            parser.error(
                f"SAM output already exists in {run}; use different filenames "
                "or pass --overwrite"
            )

    try:
        import torch
    except ImportError as exc:
        parser.error(f"PyTorch is not installed in this environment: {exc}")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was selected, but CUDA is unavailable")
    amp_dtype = None
    if device == "cuda":
        gpu_major = torch.cuda.get_device_properties(0).major
        amp_dtype = torch.bfloat16 if gpu_major >= 8 else torch.float16

    processor = _load_sam3(args.checkpoint.resolve(), device, args.threshold)
    total = 0
    for run in runs:
        total += label_run(
            run,
            processor,
            image_name=args.image_name,
            output_name=args.output_name,
            preview_name=args.preview_name,
            threshold=args.threshold,
            overwrite=args.overwrite,
            amp_dtype=amp_dtype,
        )
    print(f"Wrote {total} SAM pseudo-labels across {len(runs)} image(s).")


if __name__ == "__main__":
    main()
