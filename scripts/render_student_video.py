#!/usr/bin/env python3
"""Run student YOLO model inference on Isaac Sim simulation frames and generate an annotated video."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# Predefined distinct colors (BGR) for part classes
CLASS_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "yellow_cube": (20, 215, 255),    # Vibrant Gold/Yellow
    "blue_cube": (245, 130, 20),      # Bright Azure
    "red_cube": (40, 40, 235),        # Deep Crimson
    "cap": (50, 220, 80),             # Vivid Green
    "cap_narrow_end_up": (60, 240, 120),
    "cap_wide_end_up": (30, 190, 210),
    "vial": (220, 60, 190),           # Bright Magenta
}

# Standard pick-and-place segment timing in Isaac Sim cell.py / sort_scene.py
SEGMENTS_DEF = [
    ("approach", 1.2),
    ("descend", 0.8),
    ("close", 0.6),
    ("dwell", 0.5),
    ("lift", 0.8),
    ("carry", 1.4),
    ("lower", 0.8),
    ("release", 0.6),
    ("settle", 0.6),
    ("retreat", 0.8),
    ("home", 1.1),
]
FRAMES_PER_PICK = sum(int(round(dur * 30)) for _, dur in SEGMENTS_DEF)  # 276 frames


def natural_sort_key(path: Path) -> Tuple[Any, ...]:
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", path.name)
    )


def find_ffmpeg_binary() -> Optional[str]:
    """Locate an available ffmpeg executable (system or bundled in virtualenv)."""
    # 1. Bundled static binary from imageio_ffmpeg
    workspace_root = Path(__file__).resolve().parent.parent
    bundled = (
        workspace_root
        / "fanuc_er4ia_sim/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
    )
    if bundled.is_file() and os.access(bundled, os.X_OK):
        return str(bundled)

    # 2. Check system PATH
    import shutil
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg

    return None


def get_color_for_class(cls_name: str) -> Tuple[int, int, int]:
    """Return a consistent, high-contrast BGR color for any class name."""
    if cls_name in CLASS_PALETTE:
        return CLASS_PALETTE[cls_name]
    # Deterministic hash color for unseen arbitrary classes
    h = hash(cls_name) % 180
    bgr = cv2.cvtColor(np.uint8([[[h, 220, 240]]]), cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


class RobotPhaseResolver:
    """Computes the robot's active subtask and phase from frame index."""

    def __init__(self, events_path: Optional[Path] = None, fps: int = 30):
        self.fps = fps
        self.events: List[Dict[str, Any]] = []
        if events_path and events_path.is_file():
            try:
                data = json.loads(events_path.read_text())
                if isinstance(data, list):
                    self.events = data
                elif isinstance(data, dict) and "picks" in data:
                    self.events = data["picks"]
            except Exception:
                self.events = []

        # Cumulative segment frame bounds
        self.seg_bounds = []
        cum = 0
        for name, dur in SEGMENTS_DEF:
            count = int(round(dur * fps))
            self.seg_bounds.append((name, cum, cum + count))
            cum += count
        self.frames_per_pick = cum

    def get_phase_info(self, frame_idx: int) -> Dict[str, Any]:
        pick_idx = frame_idx // self.frames_per_pick
        rel_frame = frame_idx % self.frames_per_pick

        current_phase = "in_motion"
        for name, start, end in self.seg_bounds:
            if start <= rel_frame < end:
                current_phase = name.upper()
                break

        target_str = f"Pick #{pick_idx + 1}"
        target_obj = ""
        stack_level = ""
        if pick_idx < len(self.events):
            ev = self.events[pick_idx]
            col = ev.get("colour", "")
            cat = ev.get("category", "")
            target_obj = f"{col} {cat}".strip()
            if "stack_level" in ev:
                stack_level = f"Stack Level {ev['stack_level']}"

        return {
            "pick_idx": pick_idx,
            "target_obj": target_obj,
            "phase": current_phase,
            "stack_level": stack_level,
            "total_picks": len(self.events),
        }


class StudentVideoRenderer:
    """Performs YOLO student inference and renders annotated frames into a high-quality video."""

    def __init__(
        self,
        weights_path: Path,
        device: Optional[str] = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 1280,
    ):
        self.weights_path = Path(weights_path).resolve()
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"Model weights not found: {self.weights_path}")

        print(f"[Renderer] Loading student detector: {self.weights_path.name}")
        self.model = YOLO(str(self.weights_path))
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

        # Auto-detect device
        if device:
            self.device = device
        else:
            import torch
            self.device = "0" if torch.cuda.is_available() else "cpu"
        print(f"[Renderer] Using device: {self.device}")

        # Model classes
        self.names = self.model.names
        print(f"[Renderer] Model classes ({len(self.names)}): {self.names}")

    def annotate_frame(
        self,
        frame: np.ndarray,
        results: Any,
        frame_idx: int,
        total_frames: int,
        fps: int,
        latency_ms: float,
        phase_info: Optional[Dict[str, Any]] = None,
        show_hud: bool = True,
    ) -> np.ndarray:
        """Render clean bounding boxes, center reticles, pills, and HUD overlay."""
        h, w = frame.shape[:2]
        annotated = frame.copy()
        overlay = frame.copy()

        counts: Dict[str, int] = {}
        boxes = []

        if results and hasattr(results, "boxes") and results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                conf_val = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy()
                cls_name = self.names.get(cls_id, f"cls_{cls_id}")
                counts[cls_name] = counts.get(cls_name, 0) + 1
                boxes.append((xyxy, cls_name, conf_val))

        # Sort boxes by area descending so smaller boxes/labels render on top
        boxes.sort(key=lambda b: (b[0][2] - b[0][0]) * (b[0][3] - b[0][1]), reverse=True)

        # 1. Draw subtle translucent fills inside boxes
        for xyxy, cls_name, conf_val in boxes:
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            color = get_color_for_class(cls_name)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)

        cv2.addWeighted(overlay, 0.14, annotated, 0.86, 0, annotated)

        # 2. Draw solid boundary, corner accents, and center reticles
        for xyxy, cls_name, conf_val in boxes:
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            bw = x2 - x1
            bh = y2 - y1
            color = get_color_for_class(cls_name)

            # Main outline
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Corner bracket accents for sharp visual appearance
            corner_len = min(12, bw // 4, bh // 4)
            if corner_len > 3:
                thick = 3
                # Top-left
                cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), thick, cv2.LINE_AA)
                cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), thick, cv2.LINE_AA)
                # Top-right
                cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), (255, 255, 255), thick, cv2.LINE_AA)
                cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), (255, 255, 255), thick, cv2.LINE_AA)
                # Bottom-left
                cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), (255, 255, 255), thick, cv2.LINE_AA)
                cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), (255, 255, 255), thick, cv2.LINE_AA)
                # Bottom-right
                cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), thick, cv2.LINE_AA)
                cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), thick, cv2.LINE_AA)

            # Center reticle (+)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            reticle_r = 5
            cv2.line(annotated, (cx - reticle_r, cy), (cx + reticle_r, cy), (0, 0, 0), 2, cv2.LINE_AA)
            cv2.line(annotated, (cx, cy - reticle_r), (cx, cy + reticle_r), (0, 0, 0), 2, cv2.LINE_AA)
            cv2.line(annotated, (cx - reticle_r, cy), (cx + reticle_r, cy), color, 1, cv2.LINE_AA)
            cv2.line(annotated, (cx, cy - reticle_r), (cx, cy + reticle_r), color, 1, cv2.LINE_AA)
            cv2.circle(annotated, (cx, cy), 1, (255, 255, 255), -1, cv2.LINE_AA)

            # Text label badge
            label_text = f"{cls_name} {int(conf_val * 100)}%"
            font = cv2.FONT_HERSHEY_DUPLEX
            scale = 0.52
            thick = 1
            (tw, th), baseline = cv2.getTextSize(label_text, font, scale, thick)

            # Badge position above box (or inside if near top)
            badge_y2 = y1 - 4 if y1 - th - 8 > 60 else y1 + th + 12
            badge_y1 = badge_y2 - th - 8
            badge_x1 = max(0, x1)
            badge_x2 = badge_x1 + tw + 14

            # Dark badge outline and colored header tab
            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), (20, 22, 26), -1)
            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 1, cv2.LINE_AA)
            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x1 + 4, badge_y2), color, -1)
            cv2.putText(
                annotated,
                label_text,
                (badge_x1 + 8, badge_y2 - 5),
                font,
                scale,
                (245, 245, 245),
                thick,
                cv2.LINE_AA,
            )

        # 3. Heads-Up Display (HUD) Banner
        if show_hud:
            hud_height = 82
            hud_bg = annotated[:hud_height, :].copy()
            hud_overlay = np.zeros_like(hud_bg)
            cv2.rectangle(hud_overlay, (0, 0), (w, hud_height), (12, 14, 18), -1)
            cv2.addWeighted(hud_overlay, 0.85, hud_bg, 0.15, 0, annotated[:hud_height, :])
            # Accent bottom border for HUD
            cv2.line(annotated, (0, hud_height), (w, hud_height), (50, 180, 240), 2, cv2.LINE_AA)

            # Top Line: Title & Telemetry
            time_sec = frame_idx / fps
            total_sec = total_frames / fps if total_frames > 0 else 0
            title = "YOLO26n Student Edge Detector"
            sim_info = "Isaac Sim FANUC ER-4iA Physical AI Sorting"
            cv2.putText(
                annotated,
                f"{title} | {sim_info}",
                (20, 28),
                cv2.FONT_HERSHEY_DUPLEX,
                0.62,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Right side: Frame, time, latency
            timing_str = f"Frame {frame_idx:04d}/{total_frames:04d} ({time_sec:4.1f}s/{total_sec:4.1f}s) | Latency: {latency_ms:.1f}ms ({self.device})"
            (rw, _), _ = cv2.getTextSize(timing_str, cv2.FONT_HERSHEY_DUPLEX, 0.50, 1)
            cv2.putText(
                annotated,
                timing_str,
                (w - rw - 20, 28),
                cv2.FONT_HERSHEY_DUPLEX,
                0.50,
                (200, 215, 230),
                1,
                cv2.LINE_AA,
            )

            # Second Line: Live Detection breakdown + Robot phase
            sub_y = 62
            total_dets = sum(counts.values())
            det_label = f"Detections: {total_dets}"
            cv2.putText(
                annotated,
                det_label,
                (20, sub_y),
                cv2.FONT_HERSHEY_DUPLEX,
                0.54,
                (50, 220, 255),
                1,
                cv2.LINE_AA,
            )

            # Class count pills
            offset_x = 180
            for cls_name, c_cnt in sorted(counts.items()):
                color = get_color_for_class(cls_name)
                pill_str = f"{c_cnt} {cls_name}"
                (pw, _), _ = cv2.getTextSize(pill_str, cv2.FONT_HERSHEY_DUPLEX, 0.44, 1)
                cv2.rectangle(
                    annotated,
                    (offset_x, sub_y - 17),
                    (offset_x + pw + 16, sub_y + 5),
                    (28, 32, 40),
                    -1,
                )
                cv2.rectangle(
                    annotated,
                    (offset_x, sub_y - 17),
                    (offset_x + pw + 16, sub_y + 5),
                    color,
                    1,
                    cv2.LINE_AA,
                )
                cv2.circle(annotated, (offset_x + 8, sub_y - 6), 3, color, -1, cv2.LINE_AA)
                cv2.putText(
                    annotated,
                    pill_str,
                    (offset_x + 16, sub_y - 2),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.44,
                    (230, 235, 240),
                    1,
                    cv2.LINE_AA,
                )
                offset_x += pw + 26

            # Right side of sub-line: Robot Phase Status
            if phase_info:
                p_text = f"Robot Action: {phase_info['target_obj'] or 'Homing'} | Phase: {phase_info['phase']}"
                if phase_info.get("stack_level"):
                    p_text += f" | {phase_info['stack_level']}"
                (pw2, _), _ = cv2.getTextSize(p_text, cv2.FONT_HERSHEY_DUPLEX, 0.50, 1)
                cv2.putText(
                    annotated,
                    p_text,
                    (w - pw2 - 20, sub_y),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.50,
                    (100, 240, 160),
                    1,
                    cv2.LINE_AA,
                )

        return annotated

    def render_video(
        self,
        input_source: Path,
        output_path: Path,
        fps: int = 30,
        batch_size: int = 16,
        max_frames: Optional[int] = None,
        side_by_side: bool = False,
        show_hud: bool = True,
        events_path: Optional[Path] = None,
    ) -> Path:
        """Process simulation frames, perform batched YOLO inference, and encode MP4."""
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        input_source = Path(input_source).resolve()
        frame_files: List[Path] = []
        is_video_input = False

        if input_source.is_dir():
            supported = {".png", ".jpg", ".jpeg"}
            frame_files = sorted(
                (p for p in input_source.iterdir() if p.suffix.lower() in supported),
                key=natural_sort_key,
            )
            if not frame_files:
                raise ValueError(f"No image frames found in directory: {input_source}")
        elif input_source.is_file() and input_source.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            is_video_input = True
        else:
            raise FileNotFoundError(f"Input path not found or unsupported: {input_source}")

        if max_frames and frame_files:
            frame_files = frame_files[:max_frames]

        total_frames = len(frame_files) if frame_files else 0

        # Auto-discover execution events if in same run directory
        if not events_path:
            candidate_events = input_source.parent / "execution_events.json"
            if candidate_events.is_file():
                events_path = candidate_events
        phase_resolver = RobotPhaseResolver(events_path, fps=fps)

        # Determine dimensions from first frame
        if frame_files:
            first_img = cv2.imread(str(frame_files[0]))
            if first_img is None:
                raise ValueError(f"Could not read first frame: {frame_files[0]}")
            h, w = first_img.shape[:2]
        else:
            cap = cv2.VideoCapture(str(input_source))
            ret, first_img = cap.read()
            cap.release()
            if not ret or first_img is None:
                raise ValueError(f"Could not read from input video: {input_source}")
            h, w = first_img.shape[:2]

        out_w = w * 2 if side_by_side else w
        out_h = h

        print(
            f"[Renderer] Encoding {total_frames} frames ({out_w}x{out_h} @ {fps} fps) -> {output_path}"
        )

        ffmpeg_bin = find_ffmpeg_binary()
        proc: Optional[subprocess.Popen] = None
        cv_writer: Optional[cv2.VideoWriter] = None

        if ffmpeg_bin:
            print(f"[Renderer] Using hardware-optimized H.264 encoder: {ffmpeg_bin}")
            ffmpeg_cmd = [
                ffmpeg_bin,
                "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{out_w}x{out_h}",
                "-pix_fmt", "bgr24",
                "-r", str(fps),
                "-i", "-",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                str(output_path),
            ]
            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        else:
            print("[Renderer] FFmpeg not found, falling back to cv2.VideoWriter (mp4v)")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            cv_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

        def write_frame_to_stream(img_bgr: np.ndarray) -> None:
            if proc and proc.stdin:
                proc.stdin.write(img_bgr.tobytes())
            elif cv_writer:
                cv_writer.write(img_bgr)

        # Batch processing loop
        start_time = time.time()
        processed_count = 0

        try:
            if frame_files:
                for batch_start in range(0, len(frame_files), batch_size):
                    batch_paths = frame_files[batch_start : batch_start + batch_size]
                    batch_imgs = [cv2.imread(str(p)) for p in batch_paths]

                    t0 = time.time()
                    predictions = self.model.predict(
                        source=batch_imgs,
                        conf=self.conf,
                        iou=self.iou,
                        imgsz=self.imgsz,
                        device=self.device,
                        verbose=False,
                    )
                    batch_latency = ((time.time() - t0) * 1000) / len(batch_imgs)

                    for b_i, (raw_frame, res) in enumerate(zip(batch_imgs, predictions)):
                        idx = batch_start + b_i
                        phase_info = phase_resolver.get_phase_info(idx)
                        annotated = self.annotate_frame(
                            frame=raw_frame,
                            results=res,
                            frame_idx=idx,
                            total_frames=total_frames,
                            fps=fps,
                            latency_ms=batch_latency,
                            phase_info=phase_info,
                            show_hud=show_hud,
                        )

                        if side_by_side:
                            final_frame = np.hstack([raw_frame, annotated])
                        else:
                            final_frame = annotated

                        write_frame_to_stream(final_frame)
                        processed_count += 1

                    if (batch_start // batch_size) % 5 == 0 or processed_count == total_frames:
                        elapsed = time.time() - start_time
                        fps_rendered = processed_count / max(0.001, elapsed)
                        print(
                            f"[Renderer] Progress: {processed_count}/{total_frames} frames "
                            f"({(processed_count / total_frames) * 100:.1f}%) "
                            f"- {fps_rendered:.1f} fps - {batch_latency:.1f}ms latency"
                        )
            else:
                # Video input stream
                cap = cv2.VideoCapture(str(input_source))
                idx = 0
                while True:
                    ret, raw_frame = cap.read()
                    if not ret or raw_frame is None:
                        break
                    t0 = time.time()
                    res = self.model.predict(
                        raw_frame,
                        conf=self.conf,
                        iou=self.iou,
                        imgsz=self.imgsz,
                        device=self.device,
                        verbose=False,
                    )[0]
                    latency = (time.time() - t0) * 1000

                    phase_info = phase_resolver.get_phase_info(idx)
                    annotated = self.annotate_frame(
                        frame=raw_frame,
                        results=res,
                        frame_idx=idx,
                        total_frames=total_frames or idx + 1,
                        fps=fps,
                        latency_ms=latency,
                        phase_info=phase_info,
                        show_hud=show_hud,
                    )

                    if side_by_side:
                        final_frame = np.hstack([raw_frame, annotated])
                    else:
                        final_frame = annotated

                    write_frame_to_stream(final_frame)
                    processed_count += 1
                    idx += 1
                    if max_frames and idx >= max_frames:
                        break
                cap.release()

        finally:
            if proc:
                _, stderr_data = proc.communicate()
                if proc.returncode != 0:
                    print(
                        f"[Renderer] FFmpeg error ({proc.returncode}):\n{stderr_data.decode('utf-8', errors='ignore')}"
                    )
            elif cv_writer:
                cv_writer.release()

        elapsed_total = time.time() - start_time
        file_size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.is_file() else 0.0
        print(
            f"[Renderer] Completed {processed_count} frames in {elapsed_total:.1f}s "
            f"({processed_count / max(0.001, elapsed_total):.1f} fps) -> {output_path} ({file_size_mb:.2f} MB)"
        )
        return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run student detector inference on Isaac Sim simulation frames and generate an annotated video."
    )
    parser.add_argument(
        "--frames",
        type=Path,
        default=Path("output/runs/slurm_stack_20261014/frames"),
        help="Input frames directory (or video file)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("data/problem3_full_run_20261021_1359/models/yolo26n/weights/best.pt"),
        help="YOLO student model weights (.pt)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/student_model_demo.mp4"),
        help="Output video destination path (.mp4)",
    )
    parser.add_argument("--fps", type=int, default=30, help="Video framerate")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU NMS threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default=None, help="Device ('0', 'cpu', etc.)")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames to process")
    parser.add_argument(
        "--side-by-side",
        action="store_true",
        help="Render side-by-side (Raw Sim vs Student Detections)",
    )
    parser.add_argument(
        "--hide-hud",
        action="store_true",
        help="Omit telemetry HUD overlay",
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=None,
        help="Path to execution_events.json or sort_plan.json",
    )
    args = parser.parse_args()

    renderer = StudentVideoRenderer(
        weights_path=args.weights,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )
    renderer.render_video(
        input_source=args.frames,
        output_path=args.out,
        fps=args.fps,
        batch_size=args.batch,
        max_frames=args.max_frames,
        side_by_side=args.side_by_side,
        show_hud=not args.hide_hud,
        events_path=args.events,
    )


if __name__ == "__main__":
    main()
