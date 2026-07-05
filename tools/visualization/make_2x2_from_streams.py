#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def find_stream(root: Path, patterns):
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def open_cap(path):
    if path is None:
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    return cap


def read_panel(cap, size, label):
    if cap is None:
        img = np.zeros((size, size, 3), dtype=np.uint8)
        cv2.putText(img, label, (18, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA)
        cv2.putText(img, "missing", (18, size // 2 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (110, 110, 110), 2, cv2.LINE_AA)
        return img, True
    ok, img = cap.read()
    if not ok or img is None:
        return None, False
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = img
    cv2.rectangle(canvas, (0, 0), (size, 34), (0, 0, 0), -1)
    cv2.putText(canvas, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas, True


def transcode_with_ffmpeg(src: Path, dst: Path, fps: float) -> None:
    ffmpeg_exe = os.environ.get("FFMPEG_EXE") or shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    codec = os.environ.get("SPATIAL_SOFT_PREVIEW_CODEC", os.environ.get("SPATIAL_FFMPEG_CODEC", "libx264"))
    crf = os.environ.get("SPATIAL_SOFT_PREVIEW_CRF", os.environ.get("SPATIAL_FFMPEG_CRF", "18"))
    preset = os.environ.get("SPATIAL_SOFT_PREVIEW_PRESET", os.environ.get("SPATIAL_FFMPEG_PRESET", "medium"))
    pix_fmt = os.environ.get("SPATIAL_SOFT_PREVIEW_PIX_FMT", os.environ.get("SPATIAL_FFMPEG_PIX_FMT", "yuv420p"))
    cmd = [
        ffmpeg_exe,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-c:v",
        codec,
        "-pix_fmt",
        pix_fmt,
        "-crf",
        str(crf),
        "-preset",
        preset,
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--panel", type=int, default=int(os.environ.get("SPATIAL_SOFT_PREVIEW_PANEL", "512")))
    parser.add_argument("--max-frames", type=int, default=900)
    args = parser.parse_args()

    root = Path(args.run_root)
    agent = find_stream(root, ["video_datasets/*/videos/*agentview_rgb.mp4", "**/*agentview_rgb.mp4"])
    wrist = find_stream(root, ["video_datasets/*/videos/*eye_in_hand_rgb.mp4", "**/*eye_in_hand_rgb.mp4"])
    left = find_stream(root, ["video_datasets/*/tactile_outputs/*left_markers_rgb.mp4", "**/*left_markers_rgb.mp4"])
    right = find_stream(root, ["video_datasets/*/tactile_outputs/*right_markers_rgb.mp4", "**/*right_markers_rgb.mp4"])

    caps = [open_cap(agent), open_cap(wrist), open_cap(left), open_cap(right)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp_out = out.with_name(out.name + ".tmp.mp4")
    if temp_out.exists():
        temp_out.unlink()
    size = args.panel
    writer = cv2.VideoWriter(str(temp_out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (size * 2, size * 2))
    if not writer.isOpened():
        raise SystemExit(f"failed to open writer: {temp_out}")

    labels = ["third-person", "wrist", "left marker", "right marker"]
    written = 0
    for idx in range(args.max_frames):
        panels = []
        alive_any = False
        for cap, label in zip(caps, labels):
            panel, ok = read_panel(cap, size, label)
            if ok and panel is not None:
                alive_any = True if cap is not None else alive_any
                panels.append(panel)
            elif cap is None:
                panels.append(read_panel(None, size, label)[0])
            else:
                panels.append(None)
        if not alive_any:
            break
        for i, panel in enumerate(panels):
            if panel is None:
                panels[i] = read_panel(None, size, labels[i])[0]
        frame = np.concatenate([
            np.concatenate([panels[0], panels[1]], axis=1),
            np.concatenate([panels[2], panels[3]], axis=1),
        ], axis=0)
        cv2.putText(frame, f"{root.name}  frame {idx:04d}", (18, size * 2 - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        written += 1
    writer.release()
    for cap in caps:
        if cap is not None:
            cap.release()
    if written == 0:
        raise SystemExit(f"no frames written for {root}")

    transcode_with_ffmpeg(temp_out, out, args.fps)
    temp_out.unlink(missing_ok=True)
    print(out)


if __name__ == "__main__":
    main()
