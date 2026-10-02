# SPDX-License-Identifier: AGPL-3.0-or-later
"""Writes picked frames as full-resolution PNGs, optionally also through an
upscaling model. Shared by the app and the command line."""
from pathlib import Path

import cv2

from .video import VideoFile


def frame_filename(stem, index, label=""):
    """<video>_S014_f00431.png, or <video>_f00431.png without a label."""
    return f"{stem}_{label + '_' if label else ''}f{index:05d}.png"


def upscale_tag(model_path, scale):
    return f"{Path(model_path).stem[:32]}_{scale:g}x"


def write_png(path, rgb):
    ok, data = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError(f"cannot encode {path}")
    Path(path).write_bytes(data.tobytes())


def export_frames(video_path, jobs, up=None, progress=None, cancelled=lambda: False):
    """Decodes the frame at each (pts, out_path) job and writes it as a PNG.

    up = {"client", "model", "scale", "keep", "tag"} also writes an upscaled
    copy named <out>_<tag>.png, and the plain frame only when keep is true.
    progress(frames_done, total, stage) gets fractional frames_done.
    Returns the paths written."""
    jobs = sorted(jobs)
    n = len(jobs)
    written = []
    video = VideoFile(video_path)
    try:
        for k, (pts, out) in enumerate(jobs):
            if cancelled():
                break
            f = video.frame_at(pts)
            if f is None:
                continue
            rgb = video.planes(f).to_rgb()
            out = Path(out)
            if not up or up["keep"]:
                write_png(out, rgb)
                written.append(out)
            if progress:
                progress(k + (0.02 if up else 1), n, "Exporting")
            if up:
                dest = out.with_name(f"{out.stem}_{up['tag']}.png")
                tiles = None
                if progress:
                    def tiles(d, t, k=k):
                        progress(k + 0.02 + 0.98 * d / t, n, "Upscaling")
                up["client"].upscale_to_file(rgb, up["model"], dest, up["scale"], on_progress=tiles)
                written.append(dest)
                if progress:
                    progress(k + 1, n, "Upscaling")
    finally:
        video.close()
    return written
