# SPDX-License-Identifier: AGPL-3.0-or-later
import json

import cv2
import numpy as np

from conftest import H, W, read_barcode
from wallpaper_frame_picker import analysis as A
from wallpaper_frame_picker import cli


def read_png(path):
    return cv2.cvtColor(cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def test_analyze_and_shots(video, full_analysis, capsys):
    cli.main(["analyze", str(video)])
    assert "4 shots" in capsys.readouterr().out
    cli.main(["shots", str(video)])
    lines = capsys.readouterr().out.splitlines()
    assert [ln.split()[0] for ln in lines[1:]] == ["S000", "S001", "S002", "S003"]


def test_sheets_write_a_draft_picks_file(video, full_analysis, tmp_path):
    cli.main(["sheets", str(video), str(tmp_path)])
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["S000.jpg", "S001.jpg", "S002.jpg", "S003.jpg", "overview_0.jpg", "picks.txt"]
    picks = cli.read_picks(tmp_path / "picks.txt")
    assert [n for n, _ in picks] == ["S000", "S001", "S002", "S003"]
    cli.main(["sheets", str(video), str(tmp_path)])   # a second run keeps existing lines
    assert len(cli.read_picks(tmp_path / "picks.txt")) == 4


def test_export_frames(video, full_analysis, tmp_path):
    cli.main(["export", str(video), str(tmp_path), "--frames", "5", "100", "230"])
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["synthetic_S000_f00005.png", "synthetic_S001_f00100.png", "synthetic_S003_f00230.png"]
    for name, n in zip(files, (5, 100, 230)):
        img = read_png(tmp_path / name)
        assert img.shape == (H, W, 3) and read_barcode(img) == n


def test_export_picks_and_selection(video, full_analysis, tmp_path):
    picks = tmp_path / "picks.txt"
    picks.write_text("S003 250  # comment\nhero 12\n")
    cli.main(["export", str(video), str(tmp_path / "a"), "--picks", str(picks)])
    assert sorted(p.name for p in (tmp_path / "a").iterdir()) == ["synthetic_S003_f00250.png",
                                                                 "synthetic_hero_f00012.png"]
    an = A.Analysis(full_analysis)
    sel = A.video_dir(video) / "selections.json"
    sel.parent.mkdir(parents=True, exist_ok=True)
    sel.write_text(json.dumps([{"pts": int(an.pts[150])}]))
    try:
        cli.main(["export", str(video), str(tmp_path / "b"), "--selected"])
        out = list((tmp_path / "b").iterdir())
        assert len(out) == 1 and read_barcode(read_png(out[0])) == 150
    finally:
        sel.unlink()
