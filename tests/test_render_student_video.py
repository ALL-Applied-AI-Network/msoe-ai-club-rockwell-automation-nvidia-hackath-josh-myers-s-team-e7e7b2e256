import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from scripts.render_student_video import (
    natural_sort_key,
    find_ffmpeg_binary,
    get_color_for_class,
    RobotPhaseResolver,
    StudentVideoRenderer,
    CLASS_PALETTE,
    SEGMENTS_DEF,
    FRAMES_PER_PICK,
)


def test_natural_sort_key():
    names = ["f00010.png", "f00001.png", "f00100.png", "f00002.png"]
    paths = [Path(n) for n in names]
    sorted_paths = sorted(paths, key=natural_sort_key)
    assert [p.name for p in sorted_paths] == [
        "f00001.png",
        "f00002.png",
        "f00010.png",
        "f00100.png",
    ]


def test_find_ffmpeg_binary():
    ffmpeg = find_ffmpeg_binary()
    assert ffmpeg is not None
    assert Path(ffmpeg).is_file()


def test_get_color_for_class():
    for name in ["yellow_cube", "blue_cube", "red_cube", "cap", "vial"]:
        color = get_color_for_class(name)
        assert isinstance(color, tuple)
        assert len(color) == 3
        assert color == CLASS_PALETTE[name]

    # Test unknown arbitrary class returns a valid BGR tuple
    unk_color = get_color_for_class("arbitrary_unknown_part")
    assert isinstance(unk_color, tuple)
    assert len(unk_color) == 3
    assert all(0 <= c <= 255 for c in unk_color)


def test_robot_phase_resolver(tmp_path: Path):
    events_file = tmp_path / "events.json"
    events_data = [
        {"colour": "yellow", "category": "cube", "stack_level": 0},
        {"colour": "blue", "category": "cube", "stack_level": 1},
    ]
    events_file.write_text(json.dumps(events_data))

    resolver = RobotPhaseResolver(events_file, fps=30)
    assert resolver.frames_per_pick == 276

    # Test frame 0 (approach for pick 0)
    p0 = resolver.get_phase_info(0)
    assert p0["pick_idx"] == 0
    assert p0["target_obj"] == "yellow cube"
    assert p0["phase"] == "APPROACH"
    assert p0["stack_level"] == "Stack Level 0"

    # Test frame 40 (descend for pick 0)
    p40 = resolver.get_phase_info(40)
    assert p40["phase"] == "DESCEND"

    # Test frame 300 (pick 1)
    p300 = resolver.get_phase_info(300)
    assert p300["pick_idx"] == 1
    assert p300["target_obj"] == "blue cube"
    assert p300["stack_level"] == "Stack Level 1"


def test_student_video_renderer_annotation(tmp_path: Path):
    weights = Path("data/problem3_full_run_20261021_1359/models/yolo26n/weights/best.pt")
    if not weights.is_file():
        pytest.skip("Test requires student model checkpoint")

    renderer = StudentVideoRenderer(weights_path=weights, device="cpu")
    dummy_img = np.full((720, 1280, 3), 120, dtype=np.uint8)

    # Annotate empty results
    annotated = renderer.annotate_frame(
        frame=dummy_img,
        results=None,
        frame_idx=0,
        total_frames=100,
        fps=30,
        latency_ms=12.5,
        phase_info={"pick_idx": 0, "target_obj": "yellow cube", "phase": "APPROACH", "stack_level": ""},
        show_hud=True,
    )
    assert annotated.shape == dummy_img.shape
    # Check that HUD was rendered on top (not all 120)
    assert not np.all(annotated[:80, :] == 120)


def test_student_video_renderer_video_generation(tmp_path: Path):
    weights = Path("data/problem3_full_run_20261021_1359/models/yolo26n/weights/best.pt")
    if not weights.is_file():
        pytest.skip("Test requires student model checkpoint")

    # Create 5 dummy frames
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(5):
        frame = np.full((360, 640, 3), 50 + i * 20, dtype=np.uint8)
        cv2.imwrite(str(frames_dir / f"f{i:05d}.png"), frame)

    renderer = StudentVideoRenderer(weights_path=weights, device="cpu", imgsz=640)
    out_video = tmp_path / "output.mp4"

    result_path = renderer.render_video(
        input_source=frames_dir,
        output_path=out_video,
        fps=10,
        batch_size=2,
        show_hud=True,
    )

    assert result_path.is_file()
    assert result_path.stat().st_size > 0

    cap = cv2.VideoCapture(str(result_path))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    cap.release()

