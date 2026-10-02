# SPDX-License-Identifier: AGPL-3.0-or-later
"""Drives the app offscreen on the synthetic video."""
import os
import time

import cv2
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from conftest import SHOTS, read_barcode   # noqa: E402
from wallpaper_frame_picker import app as G   # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wait(qapp, cond, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        qapp.processEvents()
        if cond():
            return
        time.sleep(0.01)
    raise TimeoutError


@pytest.fixture
def win(qapp, video):
    w = G.MainWindow()
    w.resize(1500, 900)
    w.show()
    w.open_video(str(video))
    wait(qapp, lambda: w.cur_exact)
    yield w
    w.close()


def test_find_shots_select_and_export(qapp, win, tmp_path, monkeypatch):
    win.whole_range()
    win.start_analysis()
    wait(qapp, lambda: win.an is not None and not win.worker.isRunning())
    assert win.an.shots == SHOTS
    assert win.shot_list.count() == 4

    win.selected.clear()
    win._selection_changed()
    best = win.an.best(3)
    win.canvas.toggled.emit(best)
    assert len(win.selected) == 1 and win.sel_list.count() == 1

    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(tmp_path)))
    win.export()
    wait(qapp, lambda: not win.exporter.isRunning())
    qapp.processEvents()
    files = list(tmp_path.glob("*.png"))
    assert len(files) == 1 and files[0].name.startswith("synthetic_S003_")
    assert read_barcode(cv2.imread(str(files[0]))[..., ::-1]) == best


def test_viewer_steps_frame_by_frame(qapp, win):
    win.go_time(100 / 24)
    wait(qapp, lambda: win.cur_exact and win.info.index_of(win.cur_pts) == 100)
    for n, want in ((1, 101), (1, 102), (-1, 101), (-1, 100), (-1, 99)):
        win.step(n)
        wait(qapp, lambda: win.cur_exact and win.info.index_of(win.cur_pts) == want, 20)
    win.timeline.scrubbed.emit(3.0, False)
    wait(qapp, lambda: win.cur_exact and win.info.index_of(win.cur_pts) == 72, 20)
