"""Encode an ordered directory of PNG/JPEG images into an MP4 video."""

import argparse
import re
from pathlib import Path

import cv2
import imageio_ffmpeg


def natural_sort_key(path):
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", path.name)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Combine PNG/JPEG images in a directory into an MP4."
    )
    parser.add_argument("--frames", default="frames", help="Directory of input images")
    parser.add_argument("--out", default="demo.mp4", help="Output MP4 path")
    parser.add_argument("--fps", type=int, default=30, help="Video frames per second")
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be a positive integer")

    frames_dir = Path(args.frames)
    if not frames_dir.is_dir():
        raise SystemExit(f"image directory not found: {frames_dir}")

    supported_formats = {".png", ".jpg", ".jpeg"}
    files = sorted(
        (
            path
            for path in frames_dir.iterdir()
            if path.is_file() and path.suffix.lower() in supported_formats
        ),
        key=natural_sort_key,
    )
    if not files:
        raise SystemExit(f"no PNG or JPEG images in {frames_dir}")

    first = cv2.imread(str(files[0]))
    if first is None:
        raise SystemExit(f"could not read image: {files[0]}")
    height, width = first.shape[:2]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"{len(files)} images, {width}x{height} @ {args.fps} fps -> {output}")
    writer = imageio_ffmpeg.write_frames(
        str(output), (width, height), fps=args.fps, quality=8, macro_block_size=1
    )
    try:
        writer.send(None)
        for path in files:
            image = first if path == files[0] else cv2.imread(str(path))
            if image is None:
                raise SystemExit(f"could not read image: {path}")
            if image.shape[:2] != (height, width):
                raise SystemExit(
                    f"image size mismatch in {path}: "
                    f"expected {width}x{height}, got {image.shape[1]}x{image.shape[0]}"
                )
            writer.send(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).tobytes())
    finally:
        writer.close()

    print(
        f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB, "
        f"{len(files) / args.fps:.1f} s)"
    )


if __name__ == "__main__":
    main()
