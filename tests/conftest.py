# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared fixtures: a synthetic test video made with PyAV, and private cache
and model folders for the whole test run.

The video has four shots of simple generated patterns. Every frame carries
its frame number as a barcode in the top-left corner, so tests can check
which frame they got back. Frames 216 to 240 are blurred on purpose.
"""
import os

import av
import cv2
import numpy as np
import pytest

W, H, FPS = 640, 272, 24
SHOTS = [(0, 71), (72, 143), (144, 191), (192, 263)]   # first and last frame, inclusive
N_FRAMES = 264
BLUR = range(216, 241)
BLOCKS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 128, 0)]   # shot 2, left to right
BITS, CELL = 10, 12   # barcode: 10 cells of 12x12 px


def rgb_to_yuv420(rgb):
    """BT.709 limited range, chroma averaged over 2x2 blocks."""
    f = rgb.astype(np.float64) / 255
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cb = (b - y) / 1.8556
    cr = (r - y) / 1.5748
    Y = np.clip(np.round(16 + 219 * y), 0, 255).astype(np.uint8)
    half = lambda c: c.reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3))
    U = np.clip(np.round(128 + 224 * half(cb)), 0, 255).astype(np.uint8)
    V = np.clip(np.round(128 + 224 * half(cr)), 0, 255).astype(np.uint8)
    return Y, U, V


def frame_rgb(n):
    yy, xx = np.mgrid[0:H, 0:W]
    if n <= 71:   # moving diagonal color stripes
        t = (xx + yy + 4 * n) % 96 / 96
        img = np.stack([t, 1 - t, 0.5 + 0.5 * np.sin(6.28 * t)], -1)
    elif n <= 143:   # rings around a moving center
        d = np.hypot(xx - 200 - 3 * (n - 72), yy - 136) / 18
        img = np.stack([0.5 + 0.5 * np.sin(d), 0.3 + 0.2 * np.cos(d), np.full_like(d, 0.6)], -1)
    elif n <= 191:   # flat color blocks
        img = np.zeros((H, W, 3))
        for k, c in enumerate(BLOCKS):
            img[:, k * W // 4:(k + 1) * W // 4] = np.array(c) / 255
    else:   # wide stripes with a sliding bar
        t = (xx // 40) % 2
        img = np.stack([0.2 + 0.6 * t, 0.7 - 0.4 * t, 0.4 + 0.2 * t], -1)
        bar = (np.abs(xx - (60 + 6 * (n - 192))) < 12)
        img[bar] = (0.95, 0.95, 0.2)
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if n in BLUR:
        img = cv2.GaussianBlur(img, (0, 0), 4)
    for bit in range(BITS):   # barcode last, so it stays sharp and readable
        on = (n >> bit) & 1
        img[4:4 + CELL, 4 + bit * CELL:4 + (bit + 1) * CELL] = 255 if on else 0
    return img


def read_barcode(rgb):
    """Frame number from a full-size RGB frame."""
    n = 0
    for bit in range(BITS):
        y, x = 4 + CELL // 2, 4 + bit * CELL + CELL // 2
        if rgb[y - 2:y + 3, x - 2:x + 3].mean() > 128:
            n |= 1 << bit
    return n


def make_video(path):
    with av.open(str(path), "w", format="webm") as out:
        s = out.add_stream("libvpx-vp9", rate=FPS)
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        s.options = {"crf": "18", "b": "0", "deadline": "realtime", "cpu-used": "8", "g": "48"}
        for attr in ("colorspace", "color_range", "color_primaries", "color_trc"):
            setattr(s.codec_context, attr, 1)   # BT.709, limited range
        for n in range(N_FRAMES):
            y, u, v = rgb_to_yuv420(frame_rgb(n))
            frame = av.VideoFrame.from_ndarray(np.concatenate([y, u.reshape(-1, W), v.reshape(-1, W)]),
                                               format="yuv420p")
            frame.pts = n
            for packet in s.encode(frame):
                out.mux(packet)
        for packet in s.encode():
            out.mux(packet)


@pytest.fixture(scope="session", autouse=True)
def private_dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("wallpaper-frame-picker")
    old = {k: os.environ.get(k) for k in ("FRAME_PICKER_CACHE", "FRAME_PICKER_MODELS", "XDG_CONFIG_HOME")}
    os.environ["FRAME_PICKER_CACHE"] = str(root / "cache")
    os.environ["FRAME_PICKER_MODELS"] = str(root / "models")
    os.environ["XDG_CONFIG_HOME"] = str(root / "config")   # QSettings
    yield root
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(scope="session")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "synthetic.webm"
    make_video(path)
    return path


@pytest.fixture(scope="session")
def full_analysis(video):
    from wallpaper_frame_picker import analysis as A
    from wallpaper_frame_picker.video import VideoFile
    v = VideoFile(video)
    folder = A.analyze(video, 0.0, round(v.duration, 3))
    v.close()
    return folder
