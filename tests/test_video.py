# SPDX-License-Identifier: AGPL-3.0-or-later
import random

import numpy as np

from conftest import BLOCKS, H, N_FRAMES, W, read_barcode
from wallpaper_frame_picker.video import VideoFile


def test_metadata(video):
    v = VideoFile(video)
    assert (v.width, v.height, v.fps) == (W, H, 24.0)
    assert v.n_frames == N_FRAMES
    assert v.kr_kb == (0.2126, 0.0722) and not v.full_range


def test_frames_in_order(video):
    v = VideoFile(video)
    assert [read_barcode(v.planes(f).to_rgb()) for f in v.frames(0, v.duration)] == list(range(N_FRAMES))
    part = [v.index_of(f.pts) for f in v.frames(2.0, 6.5)]
    assert part == list(range(48, 156))


def test_exact_seeks_in_any_order(video):
    v = VideoFile(video)
    order = list(range(N_FRAMES))
    random.Random(1).shuffle(order)
    for n in order[:60] + [0, N_FRAMES - 1, 215, 216]:
        f = v.frame_at(v.pts_of_index(n))
        assert v.index_of(f.pts) == n
        assert read_barcode(v.planes(f).to_rgb()) == n


def test_colors_match_bt709(video):
    v = VideoFile(video)
    rgb = v.planes(v.frame_at(v.pts_of_index(170))).to_rgb()
    for k, want in enumerate(BLOCKS):
        x = (2 * k + 1) * W // 8
        got = rgb[100:200, x - 30:x + 30].reshape(-1, 3).mean(0)
        assert np.abs(got - want).max() <= 3, (want, got)


def test_scaled_conversion(video):
    v = VideoFile(video)
    p = v.planes(v.frame_at(0))
    assert p.to_rgb(320).shape == (136, 320, 3)
    assert p.to_rgb(order="bgr")[..., ::-1].tobytes() == p.to_rgb().tobytes()
