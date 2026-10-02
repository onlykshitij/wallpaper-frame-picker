# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-frame sharpness and motion scores, shot detection, candidate frames.

An analysis covers one time range of one video and lives in its own cache
folder: meta.json, data.npz (per-frame arrays) and thumbs/ (one JPEG per
frame). Frames inside an analysis are addressed by their local index
0..n-1; data.npz maps each to its pts.
"""
import hashlib
import json
import os
import queue
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .paths import cache_dir
from .video import VideoFile

VERSION = 1
GRID_W = 16           # tiles across the frame for sharpness and motion
THUMB_W = 640         # thumbnail width
FLOW_W = 960          # width the optical flow runs at
TOP_TILES = 8         # a frame's score is the mean of its sharpest tiles

# A frame starts a new shot when its color change from the previous frame is
# above CUT_MIN and at least CUT_RATIO times the average change of the
# CUT_WIN frames on each side, or above CUT_HARD outright (the rule
# PySceneDetect's AdaptiveDetector uses). Sensitivity scales all three.
CUT_RATIO, CUT_MIN, CUT_HARD, CUT_WIN = 3.0, 12.0, 45.0, 2


def video_key(path):
    """Identifies a video by its size and its first and last MB, so the
    analysis survives moving or renaming the file."""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha1(str(size).encode())
    with open(p, "rb") as f:
        h.update(f.read(1 << 20))
        if size > 2 << 20:
            f.seek(-(1 << 20), os.SEEK_END)
            h.update(f.read(1 << 20))
    return h.hexdigest()[:12]


def video_dir(path):
    """Cache folder for one video file."""
    key = video_key(path)
    root = cache_dir()
    if root.exists():
        for d in root.glob(f"*-{key}"):
            return d
    return root / f"{re.sub(r'[^A-Za-z0-9._-]+', '_', Path(path).stem)[:40]}-{key}"


def _write_jpeg(path, img):
    ok, data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if ok:
        Path(path).write_bytes(data.tobytes())


def range_dir(path, t0, t1):
    return video_dir(path) / f"range_{t0:.3f}-{t1:.3f}"


def saved_ranges(path):
    """(t0, t1, folder) of finished analyses of this video, newest first."""
    out = []
    for d in video_dir(path).glob("range_*"):
        m = re.fullmatch(r"range_([\d.]+)-([\d.]+)", d.name)
        if m and (d / "meta.json").exists():
            out.append((float(m[1]), float(m[2]), d))
    return sorted(out, key=lambda r: r[2].stat().st_mtime, reverse=True)


def analyze(path, t0, t1, progress=None, cancelled=lambda: False):
    """Scores every frame with t0 <= time < t1. Returns the cache folder, or
    None if cancelled. Reuses a finished analysis of the same range."""
    out = range_dir(path, t0, t1)
    if (out / "meta.json").exists():
        return out
    tmp = out.with_name(out.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "thumbs").mkdir(parents=True)

    video = VideoFile(path)
    w, h = video.width, video.height
    gh = max(4, round(GRID_W * h / w))
    fw = min(FLOW_W, w)
    fh = max(1, round(fw * h / w))
    flow_scale = w / fw
    thumb_w = min(THUMB_W, w)
    expected = max(1, round((t1 - t0) * video.fps))

    # decode on one thread, score on this one, write JPEGs on two more
    stop = threading.Event()
    frames = queue.Queue(maxsize=6)

    def put(item):
        while not stop.is_set():
            try:
                frames.put(item, timeout=0.2)
                return
            except queue.Full:
                pass

    def decode():
        try:
            for f in video.frames(t0, t1):
                if stop.is_set():
                    break
                put(video.planes(f))
        except Exception as e:  # handed to the scoring loop
            put(e)
        put(None)

    decoder = threading.Thread(target=decode, daemon=True)
    decoder.start()
    writer = ThreadPoolExecutor(2)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    cols = {k: [] for k in ("pts", "sharp_half", "sharp_quarter", "flow", "luma", "cut")}
    prev_small = prev_hsv = None
    t_start = time.time()
    try:
        while True:
            p = frames.get()
            if p is None:
                break
            if isinstance(p, Exception):
                raise p
            if cancelled():
                return None
            i = len(cols["pts"])
            y = p.y if p.y is not None else cv2.cvtColor(p.rgb, cv2.COLOR_RGB2GRAY)

            half = cv2.resize(y.astype(np.float32), (max(1, w // 2), max(1, h // 2)),
                              interpolation=cv2.INTER_AREA)
            lap = cv2.Laplacian(half, cv2.CV_32F, ksize=3)
            cols["sharp_half"].append(cv2.resize(lap * lap, (GRID_W, gh), interpolation=cv2.INTER_AREA))
            small = cv2.resize(y, (fw, fh), interpolation=cv2.INTER_AREA)
            lap = cv2.Laplacian(small.astype(np.float32), cv2.CV_32F, ksize=3)
            cols["sharp_quarter"].append(cv2.resize(lap * lap, (GRID_W, gh), interpolation=cv2.INTER_AREA))

            if prev_small is None:
                speed = np.zeros((fh, fw), np.float32)
            else:
                flow = dis.calc(prev_small, small, None)
                speed = np.hypot(flow[..., 0], flow[..., 1]) * flow_scale
            prev_small = small
            cols["flow"].append(cv2.resize(speed, (GRID_W, gh), interpolation=cv2.INTER_AREA))

            thumb = p.to_rgb(thumb_w, order="bgr")
            writer.submit(_write_jpeg, tmp / "thumbs" / f"{i:05d}.jpg", thumb)
            hsv = cv2.cvtColor(cv2.resize(thumb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2HSV).astype(np.int16)
            if prev_hsv is None:
                cols["cut"].append(0.0)
            else:
                d = np.abs(hsv - prev_hsv)
                d[..., 0] = np.minimum(d[..., 0], 180 - d[..., 0])  # hue wraps at 180
                cols["cut"].append(float(d.mean()))
            prev_hsv = hsv
            cols["luma"].append(float(small.mean()))
            cols["pts"].append(p.pts)

            if progress and (i % 4 == 0):
                progress(i + 1, max(expected, i + 1), (i + 1) / (time.time() - t_start + 1e-9))
    finally:
        stop.set()
        while decoder.is_alive():  # unblock the decoder if it waits on a full queue
            try:
                frames.get(timeout=0.1)
            except queue.Empty:
                pass
        writer.shutdown(wait=True)
        video.close()
        if cancelled() or not cols["pts"]:
            shutil.rmtree(tmp, ignore_errors=True)

    if not cols["pts"]:
        raise RuntimeError("no frames decoded in that range")
    n = len(cols["pts"])
    if progress:
        progress(n, n, n / (time.time() - t_start + 1e-9))
    np.savez(tmp / "data.npz", pts=np.array(cols["pts"], np.int64),
             sharp_half=np.array(cols["sharp_half"], np.float32),
             sharp_quarter=np.array(cols["sharp_quarter"], np.float32),
             flow=np.array(cols["flow"], np.float32),
             luma=np.array(cols["luma"], np.float32), cut=np.array(cols["cut"], np.float32))
    (tmp / "meta.json").write_text(json.dumps({
        "version": VERSION, "video": str(Path(path).resolve()), "t0": t0, "t1": t1,
        "fps": video.fps, "width": w, "height": h, "frames": n,
        "seconds": round(time.time() - t_start, 1)}, indent=1))
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)
    return out


class Analysis:
    """A finished analysis, its shots, and the user's shot edits."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / "meta.json").read_text())
        d = np.load(self.folder / "data.npz")
        self.pts = d["pts"]
        self.sharp_half, self.sharp_quarter = d["sharp_half"], d["sharp_quarter"]
        self.flow, self.luma, self.cut = d["flow"], d["luma"], d["cut"]
        self.n = len(self.pts)
        self.fps = self.meta["fps"]
        self.aspect = self.meta["height"] / self.meta["width"]
        edits_file = self.folder / "edits.json"
        self.edits = {"sensitivity": 50, "removed": [], "added": []}
        if edits_file.exists():
            self.edits.update(json.loads(edits_file.read_text()))
        self._scores = {}
        self.detect()

    # frames ------------------------------------------------------------------

    def thumb_path(self, i):
        return self.folder / "thumbs" / f"{i:05d}.jpg"

    def local(self, pts):
        """Local index of the frame with this pts, or None."""
        i = int(np.searchsorted(self.pts, pts))
        return i if i < self.n and self.pts[i] == pts else None

    def nearest(self, pts):
        """Local index of the frame showing at pts, or None outside the range."""
        if pts < self.pts[0] - 1 or pts > self.pts[-1] + (self.pts[-1] - self.pts[max(0, self.n - 2)]):
            return None
        return max(0, int(np.searchsorted(self.pts, pts, side="right")) - 1)

    # shots -------------------------------------------------------------------

    def detect(self):
        """Rebuilds self.shots, a list of (first, last) local indices."""
        f = 2 ** ((50 - self.edits["sensitivity"]) / 25)
        ratio, lo_score, hard = CUT_RATIO * f, CUT_MIN * f, CUT_HARD * f
        s = self.cut
        cuts = {0}
        for i in range(1, self.n):
            nb = np.concatenate([s[max(1, i - CUT_WIN):i], s[i + 1:i + CUT_WIN + 1]])
            r = s[i] / (nb.mean() + 1e-6) if len(nb) else 0.0
            if s[i] > hard or (s[i] > lo_score and r > ratio):
                cuts.add(i)
        removed = {self.local(p) for p in self.edits["removed"]}
        added = {self.local(p) for p in self.edits["added"]}
        cuts = sorted(((cuts - removed) | added) - {None} | {0})
        self.shots = [(a, b - 1) for a, b in zip(cuts, cuts[1:] + [self.n])]
        self._starts = np.array([a for a, _ in self.shots])
        self._scores.clear()

    def _save_edits(self):
        (self.folder / "edits.json").write_text(json.dumps(self.edits))

    def set_sensitivity(self, value):
        self.edits["sensitivity"] = int(value)
        self._save_edits()
        self.detect()

    def merge_with_next(self, shot):
        if shot + 1 >= len(self.shots):
            return
        p = int(self.pts[self.shots[shot + 1][0]])
        if p in self.edits["added"]:
            self.edits["added"].remove(p)
        else:
            self.edits["removed"].append(p)
        self._save_edits()
        self.detect()

    def split_at(self, i):
        """Starts a new shot at local frame i."""
        if i <= 0 or i in self._starts:
            return
        p = int(self.pts[i])
        if p in self.edits["removed"]:
            self.edits["removed"].remove(p)
        else:
            self.edits["added"].append(p)
        self._save_edits()
        self.detect()

    def shot_of(self, i):
        return int(np.searchsorted(self._starts, i, side="right")) - 1

    # scoring -----------------------------------------------------------------

    def scores(self, shot):
        """(z-score, percentile) arrays for the frames of a shot. The score is
        the mean log energy of a frame's TOP_TILES sharpest tiles, averaged
        over the half- and quarter-resolution measures."""
        a, b = self.shots[shot]
        if (a, b) not in self._scores:
            def top(arr):
                flat = np.log(arr[a:b + 1].reshape(b - a + 1, -1) + 1e-3)
                t = np.sort(flat, axis=1)[:, -TOP_TILES:].mean(axis=1)
                return (t - t.mean()) / (t.std() + 1e-6)
            z = 0.5 * top(self.sharp_quarter) + 0.5 * top(self.sharp_half)
            pct = (np.argsort(np.argsort(z)) + 0.5) / len(z) * 100
            self._scores[(a, b)] = (z, pct)
        return self._scores[(a, b)]

    def speed(self, i):
        """90th-percentile tile speed around frame i (mean of the flow into
        and out of it, staying inside its shot), in full-res px per frame."""
        a, b = self.shots[self.shot_of(i)]
        sides = ([self.flow[i]] if i > a else []) + ([self.flow[i + 1]] if i < b else [])
        return float(np.percentile(np.mean(sides, axis=0), 90)) if sides else 0.0

    def candidates(self, shot, k):
        """Splits the shot into up to k equal parts and returns
        (local index, sharpness percentile, speed) of the sharpest frame in
        each part."""
        a, b = self.shots[shot]
        z, pct = self.scores(shot)
        out = []
        for part in np.array_split(np.arange(b - a + 1), min(k, b - a + 1)):
            j = part[np.argmax(z[part])]
            out.append((a + int(j), int(pct[j]), self.speed(a + int(j))))
        return out

    def best(self, shot):
        a, _ = self.shots[shot]
        return a + int(np.argmax(self.scores(shot)[0]))

    def percentile(self, i):
        shot = self.shot_of(i)
        return float(self.scores(shot)[1][i - self.shots[shot][0]])
