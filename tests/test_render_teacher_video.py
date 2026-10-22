import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from scripts.render_teacher_video import TeacherVideoRenderer


def test_teacher_renderer_initialization(tmp_path: Path):
    real_cam = Path("output/runs/cap_sort_demo/camera.json")
    if not real_cam.is_file():
        pytest.skip("Test requires real camera.json")

    cam_data = json.loads(real_cam.read_text())
    cam_file = tmp_path / "camera.json"
    cam_file.write_text(json.dumps(cam_data))

    gt_file = tmp_path / "ground_truth.json"
    gt_data = {
        "objects": [
            {
                "name": "SortCap0",
                "category": "cap",
                "colour": "yellow",
                "xyz": [0.461, 0.085, 0.00785],
                "orientation": "narrow_end_up",
            }
        ]
    }
    gt_file.write_text(json.dumps(gt_data))

    renderer = TeacherVideoRenderer(
        camera_path=cam_file,
        ground_truth_path=gt_file,
    )
    assert renderer.W == cam_data["resolution"][0]
    assert renderer.H == cam_data["resolution"][1]

    # Test point projection for table coordinate
    pt = renderer.project_point([0.461, 0.085, 0.902])
    assert pt is not None
    assert isinstance(pt, tuple)
    assert len(pt) == 2
    assert 0 <= pt[0] < renderer.W
    assert 0 <= pt[1] < renderer.H


def test_teacher_frame_annotation(tmp_path: Path):
    cam_file = Path("output/runs/cap_sort_demo/camera.json")
    gt_file = Path("output/runs/cap_sort_demo/ground_truth.json")
    events_file = Path("output/runs/cap_sort_demo/execution_report.json")
    if not (cam_file.is_file() and gt_file.is_file()):
        pytest.skip("Test requires cap_sort_demo scene files")

    renderer = TeacherVideoRenderer(
        camera_path=cam_file,
        ground_truth_path=gt_file,
        execution_report_path=events_file,
    )

    dummy_frame = np.full((1080, 1920, 3), 100, dtype=np.uint8)
    annotated = renderer.annotate_teacher_frame(
        frame=dummy_frame,
        frame_idx=50,
        total_frames=552,
        fps=30,
    )

    assert annotated.shape == (1080, 1920, 3)
    # Check that HUD was rendered on top
    assert not np.all(annotated[:90, :] == 100)

