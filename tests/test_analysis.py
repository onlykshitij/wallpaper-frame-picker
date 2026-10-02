# SPDX-License-Identifier: AGPL-3.0-or-later
import shutil

import numpy as np

from conftest import BLUR, SHOTS
from wallpaper_frame_picker import analysis as A


def test_finds_the_four_shots(full_analysis):
    an = A.Analysis(full_analysis)
    assert an.shots == SHOTS


def test_blurred_frames_rank_low(full_analysis):
    an = A.Analysis(full_analysis)
    first, _ = an.shots[3]
    _, pct = an.scores(3)
    blurred = [pct[i - first] for i in BLUR]
    assert np.mean(blurred) < 30
    assert an.best(3) not in BLUR


def test_candidates_cover_the_shot(full_analysis):
    an = A.Analysis(full_analysis)
    cands = [i for i, _, _ in an.candidates(3, 12)]
    assert len(cands) == 12 and cands == sorted(cands)
    assert cands[0] < 198 and cands[-1] > 257   # one from each end of frames 192..263


def test_shot_edits_persist(full_analysis):
    (full_analysis / "edits.json").unlink(missing_ok=True)
    try:
        an = A.Analysis(full_analysis)
        an.merge_with_next(0)
        assert an.shots[0] == (0, 143)
        an.split_at(100)
        assert A.Analysis(full_analysis).shots == [(0, 99), (100, 143), (144, 191), (192, 263)]
        an.set_sensitivity(100)
        assert len(an.shots) >= len(SHOTS)
    finally:
        (full_analysis / "edits.json").unlink(missing_ok=True)


def test_part_of_the_video(video):
    folder = A.analyze(video, 2.0, 6.5)
    an = A.Analysis(folder)
    assert an.n == 108
    # cuts at frames 72 and 144 fall at local frames 24 and 96
    assert an.shots == [(0, 23), (24, 95), (96, 107)]


def test_cache_is_reused_and_survives_renames(video, full_analysis, tmp_path):
    meta = A.Analysis(full_analysis).meta
    assert A.analyze(video, meta["t0"], meta["t1"]) == full_analysis
    copy = tmp_path / "renamed copy.webm"
    shutil.copy(video, copy)
    assert A.video_key(copy) == A.video_key(video)
    assert A.video_dir(copy) == A.video_dir(video)
