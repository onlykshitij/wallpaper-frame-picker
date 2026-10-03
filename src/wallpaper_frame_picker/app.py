# SPDX-License-Identifier: AGPL-3.0-or-later
"""Qt interface for picking sharp frames from a video."""
import json
import shutil
import sys
import threading
import traceback
from collections import OrderedDict, deque
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import QObject, QPointF, QRect, QRectF, QSettings, QSize, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import (QColor, QDesktopServices, QFont, QIcon, QImage, QKeySequence, QPainter,
                           QPainterPath, QPalette, QPen, QPixmap, QShortcut)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QFileDialog,
                               QFormLayout, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton,
                               QScrollArea, QSizePolicy, QSlider, QSpinBox, QSplitter, QTabWidget, QToolTip,
                               QVBoxLayout, QWidget)

from . import NOTICE, SOURCE_URL
from . import analysis as A
from . import upscaler as U
from .export import export_frames, frame_filename, upscale_tag
from .timefmt import fmt_time, parse_time
from .video import VideoFile

BG = QColor("#16181d")
PANEL = QColor("#1d2027")
TILE = QColor("#252932")
TEXT = QColor("#e6e8ee")
DIM = QColor("#9aa1ad")
ACCENT = QColor("#4c8dff")
GOOD, MID, BAD = QColor("#3fb950"), QColor("#d6a12a"), QColor("#f0626a")
VIDEO_FILTER = "Videos (*.mp4 *.mkv *.webm *.mov *.avi *.m4v *.ts *.mts *.m2ts *.wmv *.flv);;All files (*)"


def score_color(pct):
    return GOOD if pct >= 75 else MID if pct >= 40 else BAD


def to_qimage(rgb):
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def qimage_crop(img, rect):
    """RGB numpy copy of `rect` (QRect, image pixels) from an RGB888 QImage."""
    img = img.convertToFormat(QImage.Format_RGB888) if img.format() != QImage.Format_RGB888 else img
    arr = np.frombuffer(img.constBits(), np.uint8, img.bytesPerLine() * img.height())
    arr = arr.reshape(img.height(), img.bytesPerLine())[:, :img.width() * 3].reshape(img.height(), img.width(), 3)
    return arr[rect.top():rect.bottom() + 1, rect.left():rect.right() + 1].copy()


class Task(QThread):
    """Runs fn() on a thread."""
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.succeeded.emit(self.fn())
        except U.SetupError as e:   # already a readable message
            self.failed.emit(str(e))
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))


# background work ---------------------------------------------------------------

class AnalysisWorker(QThread):
    progress = Signal(int, int, float)   # frames done, frames expected, fps
    succeeded = Signal(str)              # analysis folder
    failed = Signal(str)

    def __init__(self, path, t0, t1):
        super().__init__()
        self.path, self.t0, self.t1 = path, t0, t1

    def run(self):
        try:
            out = A.analyze(self.path, self.t0, self.t1, progress=self.progress.emit,
                            cancelled=self.isInterruptionRequested)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))
            return
        if out is None:
            self.failed.emit("")
        else:
            self.succeeded.emit(str(out))


class FrameServer(QThread):
    """Decodes frames for the viewer on its own thread. Only the newest
    request is served. Recently decoded frames are kept, so stepping back a
    few frames does not need another seek."""
    ready = Signal(object, QImage, bool, int)   # pts, image, exact, request id
    CACHE_BYTES = 400 * 2 ** 20
    KEEP_VISITED = 40

    def __init__(self, path):
        super().__init__()
        self.path = path
        self._cond = threading.Condition()
        self._req = None
        self._quit = False
        self._cache = OrderedDict()   # pts -> Planes
        self._bytes = 0

    def request(self, kind, pts, rid):
        with self._cond:
            self._req = (kind, pts, rid)
            self._cond.notify()

    def stop(self):
        with self._cond:
            self._quit = True
            self._cond.notify()
        self.wait()

    def _store(self, planes):
        if planes.pts in self._cache:
            self._cache.move_to_end(planes.pts)
            return
        self._cache[planes.pts] = planes
        self._bytes += planes.nbytes
        while self._bytes > self.CACHE_BYTES and len(self._cache) > 1:
            _, old = self._cache.popitem(last=False)
            self._bytes -= old.nbytes

    def _lookup(self, pts, tol):
        for k in self._cache:
            if abs(k - pts) <= tol:
                self._cache.move_to_end(k)
                return self._cache[k]
        return None

    def run(self):
        video = VideoFile(self.path)
        try:
            while True:
                with self._cond:
                    while self._req is None and not self._quit:
                        self._cond.wait()
                    if self._quit:
                        return
                    kind, pts, rid = self._req
                    self._req = None
                try:
                    if kind == "fast":
                        f = video.keyframe_near(pts)
                        if f is not None:
                            p = video.planes(f)
                            self._store(p)
                            self.ready.emit(p.pts, to_qimage(p.to_rgb(1600)), False, rid)
                        continue
                    p = self._lookup(pts, video.frame_pts // 2)
                    if p is None:
                        visited = deque(maxlen=self.KEEP_VISITED)
                        f = video.frame_at(pts, visit=visited.append)
                        for v in visited:
                            self._store(video.planes(v))
                        if f is None:
                            continue
                        p = video.planes(f)
                        self._store(p)
                    self.ready.emit(p.pts, to_qimage(p.to_rgb()), True, rid)
                except Exception:
                    traceback.print_exc()
        finally:
            video.close()


class ExportWorker(QThread):
    """Writes full-resolution PNGs. With `up` set, also runs each frame
    through the upscaler: up = {"client", "model", "scale", "keep", "tag"}."""
    progress = Signal(float, int, str)   # frames done (fractional), total, stage
    succeeded = Signal(list)
    failed = Signal(str)

    def __init__(self, path, jobs, up=None):
        super().__init__()
        self.path, self.jobs, self.up = path, sorted(jobs), up

    def run(self):
        try:
            written = export_frames(self.path, self.jobs, self.up, progress=self.progress.emit,
                                    cancelled=self.isInterruptionRequested)
            self.succeeded.emit([str(p) for p in written])
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))


class PreviewUpscaler(QThread):
    """Upscales the small area under the loupe. Only the newest request is
    served."""
    ready = Signal(object, QRect, QImage)   # pts, source rect it covers, upscaled image

    def __init__(self, client):
        super().__init__()
        self.client = client
        self._cond = threading.Condition()
        self._req = None
        self._quit = False

    def request(self, pts, rect, crop, pad, model, scale):
        with self._cond:
            self._req = (pts, rect, crop, pad, model, scale)
            self._cond.notify()

    def stop(self):
        with self._cond:
            self._quit = True
            self._cond.notify()
        self.wait()

    def run(self):
        while True:
            with self._cond:
                while self._req is None and not self._quit:
                    self._cond.wait()
                if self._quit:
                    return
                pts, rect, crop, pad, model, scale = self._req
                self._req = None
            try:
                out = self.client.upscale_array(crop, model, scale)
                s = out.shape[1] / crop.shape[1]
                (l, t), w, h = pad, rect.width(), rect.height()
                core = out[round(t * s):round((t + h) * s), round(l * s):round((l + w) * s)]
                self.ready.emit(pts, rect, to_qimage(core))
            except Exception:
                traceback.print_exc()


class ModelDownload(QThread):
    progress = Signal(int, int)
    succeeded = Signal(str)
    failed = Signal(str)

    def __init__(self, url, dest):
        super().__init__()
        self.url, self.dest = url, Path(dest)

    def run(self):
        try:
            U.download_model(self.url, self.dest, self.progress.emit)
            self.succeeded.emit(str(self.dest))
        except Exception as e:
            self.failed.emit(str(e))


# widgets -------------------------------------------------------------------

class PreviewWidget(QWidget):
    """Shows a frame scaled to fit, with a 1:1 loupe under the cursor. With
    AI compare on, the loupe shows plain resizing next to the AI model's
    output for the same area."""
    clicked = Signal()
    ai_wanted = Signal(QRect)   # source-pixel area the AI panel needs

    PANEL = 300

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 160)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._img = None
        self._pix = None
        self._exact = False
        self._note = ""
        self._mouse = None
        self.loupe = True
        self.zoom = 1
        self.ai_enabled = False
        self.ai_scale = 4.0      # output size relative to the source
        self.ai_label = ""
        self.ai = None           # (source QRect, QImage) from the model
        self._rest = QTimer(self, singleShot=True, interval=180)
        self._rest.timeout.connect(self._want_ai)

    def set_image(self, img, exact, note=""):
        self._img, self._exact, self._note, self._pix = img, exact, note, None
        self.ai = None
        self.update()
        if self.ai_enabled and exact and self._mouse is not None:
            self._rest.start()

    def set_ai_mode(self, enabled, scale=4.0, label=""):
        self.ai_enabled, self.ai_scale, self.ai_label = enabled, scale, label
        self.ai = None
        self.update()

    def set_ai(self, rect, img):
        self.ai = (rect, img)
        self.update()

    def _loupe_area(self):
        """Source-pixel square the loupe shows, centered on the cursor."""
        r = self._target()
        ix = (self._mouse.x() - r.x()) / r.width() * self._img.width()
        iy = (self._mouse.y() - r.y()) / r.height() * self._img.height()
        panel = self.PANEL if self.ai_enabled else 360
        side = panel * self.devicePixelRatioF() / self.zoom / (self.ai_scale if self.ai_enabled else 1)
        return QRectF(ix - side / 2, iy - side / 2, side, side)

    def _want_ai(self):
        if not (self.ai_enabled and self._exact and self._img is not None and self._mouse is not None):
            return
        if not self._target().contains(self._mouse):
            return
        a = self._loupe_area()
        if self.ai and QRectF(self.ai[0]).contains(a.intersected(QRectF(self._img.rect()))):
            return
        # ask for twice the visible area so small mouse moves stay covered
        want = QRect(round(a.center().x() - a.width()), round(a.center().y() - a.height()),
                     round(a.width() * 2), round(a.height() * 2)).intersected(self._img.rect())
        if not want.isEmpty():
            self.ai_wanted.emit(want)

    def image(self):
        return self._img

    def _target(self):
        iw, ih = self._img.width(), self._img.height()
        s = min(self.width() / iw, self.height() / ih)
        w, h = iw * s, ih * s
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0d0e11"))
        if self._img is None:
            p.setPen(DIM)
            p.drawText(self.rect(), Qt.AlignCenter, "Open a video (Ctrl+O) or drop one on the window")
            return
        r = self._target()
        dpr = self.devicePixelRatioF()
        size = QSize(round(r.width() * dpr), round(r.height() * dpr))
        if self._pix is None or self._pix.size() != size:
            self._pix = QPixmap.fromImage(self._img.scaled(size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
            self._pix.setDevicePixelRatio(dpr)
        p.drawPixmap(r.topLeft(), self._pix)
        if self._note:
            p.setFont(QFont(p.font().family(), 9))
            w = p.fontMetrics().horizontalAdvance(self._note) + 16
            badge = QRectF(r.x() + 10, r.y() + 10, w, 24)
            p.setRenderHint(QPainter.Antialiasing)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 170))
            p.drawRoundedRect(badge, 6, 6)
            p.setPen(MID)
            p.drawText(badge, Qt.AlignCenter, self._note)
        if self.loupe and self._exact and self._mouse is not None and r.contains(self._mouse):
            self._draw_loupe(p, r, dpr)

    def _panel(self, p, dst, label):
        p.save()
        path = QPainterPath()
        path.addRoundedRect(dst, 10, 10)
        p.setClipPath(path)
        p.fillRect(dst, Qt.black)
        return path

    def _end_panel(self, p, dst, label):
        p.restore()
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(ACCENT, 2))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(dst, 10, 10)
        p.setFont(QFont(p.font().family(), 9))
        w = p.fontMetrics().horizontalAdvance(label) + 14
        badge = QRectF(dst.x() + 8, dst.bottom() - 30, w, 22)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 175))
        p.drawRoundedRect(badge, 5, 5)
        p.setPen(TEXT)
        p.drawText(badge, Qt.AlignCenter, label)

    def _draw_loupe(self, p, r, dpr):
        src = self._loupe_area()
        size = self.PANEL if self.ai_enabled else 360
        width = 2 * size + 8 if self.ai_enabled else size
        x = self._mouse.x() + 24
        y = self._mouse.y() + 24
        if x + width > self.width():
            x = self._mouse.x() - 24 - width
        if y + size > self.height():
            y = self._mouse.y() - 24 - size
        x, y = max(0, x), max(0, y)
        mag = self.zoom * (self.ai_scale if self.ai_enabled else 1)
        dst = QRectF(x, y, size, size)
        self._panel(p, dst, "")
        # real pixels at 100% and up; a smooth resize stands in for plain upscaling
        p.setRenderHint(QPainter.SmoothPixmapTransform, self.ai_enabled and mag > 1)
        p.drawImage(dst, self._img, src)
        left = f"{self.zoom * 100}%, scroll to zoom" if not self.ai_enabled else f"Plain resize {mag:g}x"
        self._end_panel(p, dst, left)
        if not self.ai_enabled:
            return
        dst = QRectF(x + size + 8, y, size, size)
        self._panel(p, dst, "")
        covered = self.ai and QRectF(self.ai[0]).contains(src.intersected(QRectF(self._img.rect())))
        if covered:
            rect, img = self.ai
            s = img.width() / rect.width()
            sub = QRectF((src.x() - rect.x()) * s, (src.y() - rect.y()) * s, src.width() * s, src.height() * s)
            p.setRenderHint(QPainter.SmoothPixmapTransform, self.zoom < 1)
            p.drawImage(dst, img, sub)
        else:
            p.setPen(DIM)
            p.drawText(dst, Qt.AlignCenter, "AI preview loading…\nhold the mouse still")
        self._end_panel(p, dst, f"{self.ai_label} {mag:g}x")

    def mouseMoveEvent(self, e):
        self._mouse = e.position()
        if self.loupe:
            self.update()
            if self.ai_enabled:
                self._rest.start()

    def leaveEvent(self, e):
        self._mouse = None
        self.update()

    def mousePressEvent(self, e):
        self.clicked.emit()

    def wheelEvent(self, e):
        if not self.loupe:
            return
        steps = [1, 2, 4]
        i = steps.index(self.zoom)
        i = min(i + 1, 2) if e.angleDelta().y() > 0 else max(i - 1, 0)
        self.zoom = steps[i]
        self.update()

    def resizeEvent(self, e):
        self._pix = None


class Timeline(QWidget):
    """The whole video: analyzed range, shot cuts, selected frames and the
    playhead. Drag to scrub."""
    scrubbed = Signal(float, bool)   # time, still dragging

    M = 12

    def __init__(self):
        super().__init__()
        self.setFixedHeight(44)
        self.setMouseTracking(True)
        self.duration = 0.0
        self.t = 0.0
        self.range = None
        self.cuts = []
        self.marks = []
        self._drag = False
        self._hover = None

    def _x(self, t):
        return self.M + (self.width() - 2 * self.M) * t / max(self.duration, 1e-9)

    def _t(self, x):
        return min(max((x - self.M) / max(1, self.width() - 2 * self.M), 0.0), 1.0) * self.duration

    def set_time(self, t):
        self.t = t
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        mid = self.height() / 2 + 4
        track = QRectF(self.M, mid - 5, self.width() - 2 * self.M, 10)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#2c313b"))
        p.drawRoundedRect(track, 5, 5)
        if self.duration <= 0:
            return
        if self.range:
            x0, x1 = self._x(self.range[0]), self._x(self.range[1])
            p.setBrush(QColor(76, 141, 255, 90))
            p.drawRoundedRect(QRectF(x0, mid - 5, max(2, x1 - x0), 10), 4, 4)
            p.setPen(QPen(QColor(230, 232, 238, 110), 1))
            for t in self.cuts:
                x = self._x(t)
                p.drawLine(QPointF(x, mid - 5), QPointF(x, mid + 5))
        p.setPen(Qt.NoPen)
        p.setBrush(ACCENT)
        for t in self.marks:
            p.drawEllipse(QPointF(self._x(t), mid - 12), 3, 3)
        x = self._x(self.t)
        p.setPen(QPen(TEXT, 2))
        p.drawLine(QPointF(x, mid - 9), QPointF(x, mid + 9))
        p.setBrush(TEXT)
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(x, mid), 5, 5)
        if self._hover is not None:
            text = fmt_time(self._t(self._hover))
            p.setPen(DIM)
            w = p.fontMetrics().horizontalAdvance(text)
            tx = min(max(self._hover - w / 2, 0), self.width() - w)
            p.drawText(QPointF(tx, 12), text)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self.duration > 0:
            self._drag = True
            self.t = self._t(e.position().x())
            self.scrubbed.emit(self.t, True)
            self.update()

    def mouseMoveEvent(self, e):
        self._hover = e.position().x()
        if self._drag:
            self.t = self._t(self._hover)
            self.scrubbed.emit(self.t, True)
        self.update()

    def mouseReleaseEvent(self, e):
        if self._drag:
            self._drag = False
            self.t = self._t(e.position().x())
            self.scrubbed.emit(self.t, False)
            self.update()

    def leaveEvent(self, e):
        self._hover = None
        self.update()


class SharpnessStrip(QWidget):
    """One bar per frame of the current shot: height is sharpness, color is
    its rank in the shot. Click a bar to jump to that frame."""
    jump = Signal(int)   # local frame index

    def __init__(self):
        super().__init__()
        self.setFixedHeight(76)
        self.setMouseTracking(True)
        self.an = None
        self.shot = None
        self.current = None
        self.selected = set()
        self.cands = set()
        self.label = lambda i: ""

    def set_shot(self, an, shot, current, selected=(), cands=()):
        self.an, self.shot, self.current = an, shot, current
        self.selected, self.cands = set(selected), set(cands)
        self.update()

    def _geom(self):
        a, b = self.an.shots[self.shot]
        n = b - a + 1
        bw = (self.width() - 24) / n
        return a, n, bw

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), PANEL)
        if self.an is None or self.shot is None:
            p.setPen(DIM)
            p.drawText(self.rect(), Qt.AlignCenter, "Sharpness bars appear here for frames inside the analyzed range")
            return
        a, n, bw = self._geom()
        z, pct = self.an.scores(self.shot)
        lo, hi = float(z.min()), float(z.max())
        top, base = 26, self.height() - 10
        p.setPen(DIM)
        p.setFont(QFont(p.font().family(), 8))
        p.drawText(QRectF(12, 3, self.width() - 24, 16), Qt.AlignLeft | Qt.AlignVCenter,
                   f"Sharpness across S{self.shot:03d} ({n} frames). Taller is sharper, gray ticks mark "
                   "the frames on its sheet, click a bar to jump.")
        for j in range(n):
            frac = (z[j] - lo) / (hi - lo) if hi > lo else 1.0
            h = 4 + frac * (base - top - 4)
            x = 12 + j * bw
            r = QRectF(x + bw * 0.12, base - h, max(1.0, bw * 0.76), h)
            p.fillRect(r, score_color(pct[j]))
            i = a + j
            if i in self.cands:
                p.fillRect(QRectF(x + bw * 0.12, top - 2, max(1.0, bw * 0.76), 2), DIM)
            if i in self.selected:
                p.fillRect(QRectF(x, base + 3, max(2.0, bw), 4), ACCENT)
            if i == self.current:
                p.setPen(QPen(TEXT, 2))
                p.setBrush(Qt.NoBrush)
                p.drawRect(QRectF(x, top, max(2.0, bw), base - top))

    def _index_at(self, x):
        a, n, bw = self._geom()
        j = int((x - 12) // bw)
        return a + j if 0 <= j < n else None

    def mousePressEvent(self, e):
        if self.an is not None and self.shot is not None:
            i = self._index_at(e.position().x())
            if i is not None:
                self.jump.emit(i)

    def mouseMoveEvent(self, e):
        if self.an is not None and self.shot is not None:
            i = self._index_at(e.position().x())
            if i is not None:
                QToolTip.showText(e.globalPosition().toPoint(), self.label(i), self)


class SheetCanvas(QWidget):
    """Candidate sheets for every shot, drawn as one tall canvas inside a
    scroll area. Thumbnails load as they scroll into view."""
    toggled = Signal(int)        # local frame index
    opened = Signal(int)         # local frame index
    shot_clicked = Signal(int)   # shot index

    M, G, HEADER, LABEL = 14, 10, 30, 24

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.an = None
        self.shots = []
        self.k = 12
        self.tile_w = 320
        self.selected = set()
        self.label = lambda i: ("", "")
        self.sections = []   # (shot, header rect, bottom, [(local, pct, speed, rect, best)])
        self._cands = {}
        self._pix = OrderedDict()
        self._hover = None
        self._width = -1

    def set_analysis(self, an, shots, k, tile_w):
        if an is not self.an:
            self._cands.clear()
            self._pix.clear()
        self.an, self.shots, self.k, self.tile_w = an, shots, k, tile_w
        self.relayout(force=True)

    def set_selected(self, locals_):
        self.selected = set(locals_)
        self.update()

    def candidates(self, shot):
        key = (self.an.shots[shot], self.k)
        if key not in self._cands:
            self._cands[key] = self.an.candidates(shot, self.k)
        return self._cands[key]

    def relayout(self, force=False):
        w = self.width()
        if not force and w == self._width:
            return
        self._width = w
        self.sections = []
        if self.an is None or not self.shots:
            self.setFixedHeight(max(200, self.parentWidget().height() if self.parentWidget() else 200))
            self.update()
            return
        M, G = self.M, self.G
        cols = max(1, (w - 2 * M + G) // (self.tile_w + G))
        tw = max(80, (w - 2 * M - (cols - 1) * G) // cols)
        th = round(tw * self.an.aspect)
        y = M
        for shot in self.shots:
            cands = self.candidates(shot)
            best = max(cands, key=lambda c: c[1])[0]
            header = QRect(M, y, w - 2 * M, self.HEADER)
            y += self.HEADER + 4
            tiles = []
            for j, (i, pct, v) in enumerate(cands):
                r, c = divmod(j, cols)
                rect = QRect(M + c * (tw + G), y + r * (th + self.LABEL + G), tw, th + self.LABEL)
                tiles.append((i, pct, v, rect, i == best))
            y += -(-len(cands) // cols) * (th + self.LABEL + G) + 18
            self.sections.append((shot, header, y, tiles))
        self.setFixedHeight(y)
        self.update()

    def section_top(self, shot):
        for s, header, _, _ in self.sections:
            if s == shot:
                return header.top()
        return None

    def resizeEvent(self, e):
        self.relayout()

    def _pixmap(self, i, size):
        dpr = self.devicePixelRatioF()
        key = (i, size.width(), size.height())
        pix = self._pix.get(key)
        if pix is None:
            img = QImage(str(self.an.thumb_path(i)))
            pix = QPixmap.fromImage(img.scaled(size * dpr, Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
            pix.setDevicePixelRatio(dpr)
            self._pix[key] = pix
            if len(self._pix) > 800:
                self._pix.popitem(last=False)
        else:
            self._pix.move_to_end(key)
        return pix

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(e.rect(), BG)
        if not self.sections:
            p.setPen(DIM)
            msg = ("Open a video, choose the part to use, and press Find shots."
                   if self.an is None else "No shots to show. Lower 'Hide shots under' to see short shots.")
            p.drawText(QRect(0, 0, self.width(), 200), Qt.AlignCenter, msg)
            return
        clip = e.rect()
        p.setRenderHint(QPainter.Antialiasing)
        base_font = p.font()
        bold = QFont(base_font)
        bold.setBold(True)
        small = QFont(base_font)
        small.setPointSizeF(max(7.0, base_font.pointSizeF() * 0.9))
        for shot, header, bottom, tiles in self.sections:
            if bottom < clip.top():
                continue
            if header.top() > clip.bottom():
                break
            a, b = self.an.shots[shot]
            n_sel = sum(1 for i in self.selected if a <= i <= b)
            p.setFont(bold)
            p.setPen(TEXT)
            p.drawText(header.adjusted(0, 0, 0, -6), Qt.AlignLeft | Qt.AlignBottom, f"S{shot:03d}")
            p.setFont(base_font)
            p.setPen(DIM)
            t0, t1 = self.label(a)[1], self.label(b)[1]
            sel = f"   ·   {n_sel} selected" if n_sel else ""
            p.drawText(header.adjusted(54, 0, 0, -6), Qt.AlignLeft | Qt.AlignBottom,
                       f"{t0} – {t1}   ·   {b - a + 1} frames{sel}")
            p.setPen(QPen(QColor("#2c313b"), 1))
            p.drawLine(header.bottomLeft(), header.bottomRight())
            for i, pct, v, rect, best in tiles:
                if rect.intersects(clip):
                    self._draw_tile(p, i, pct, v, rect, best, small)

    def _draw_tile(self, p, i, pct, v, rect, best, small):
        img_r = QRect(rect.x(), rect.y(), rect.width(), rect.height() - self.LABEL)
        p.drawPixmap(img_r, self._pixmap(i, img_r.size()))
        lab = QRect(rect.x(), img_r.bottom() + 1, rect.width(), self.LABEL)
        p.fillRect(lab, TILE)
        p.setFont(small)
        frame, t = self.label(i)
        p.setPen(TEXT)
        p.drawText(lab.adjusted(8, 0, -8, 0), Qt.AlignLeft | Qt.AlignVCenter,
                   ("★ " if best else "") + f"{frame}  {t}")
        stats = f"sharp {pct}%   motion {v:.1f}"
        sw = p.fontMetrics().horizontalAdvance(stats)
        if sw + 150 < rect.width():
            p.setPen(DIM)
            p.drawText(lab.adjusted(8, 0, -8, 0), Qt.AlignRight | Qt.AlignVCenter, stats)
            dot_x = lab.right() - 8 - sw - 10
        else:
            dot_x = lab.right() - 14
        p.setPen(Qt.NoPen)
        p.setBrush(score_color(pct))
        p.drawEllipse(QPointF(dot_x, lab.center().y() + 1), 4, 4)
        if i in self.selected:
            p.setPen(QPen(ACCENT, 3))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(QRectF(rect).adjusted(1.5, 1.5, -1.5, -1.5), 4, 4)
            c = QPointF(img_r.right() - 16, img_r.top() + 16)
            p.setPen(Qt.NoPen)
            p.setBrush(ACCENT)
            p.drawEllipse(c, 11, 11)
            p.setPen(QPen(Qt.white, 2.4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawPolyline([c + QPointF(-5, 0), c + QPointF(-1.5, 4), c + QPointF(5, -4)])
        elif i == self._hover:
            p.setPen(QPen(QColor(230, 232, 238, 150), 2))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(QRectF(rect).adjusted(1, 1, -1, -1), 4, 4)

    def _hit(self, pos):
        for shot, header, bottom, tiles in self.sections:
            if pos.y() > bottom:
                continue
            if header.contains(pos):
                return "header", shot
            for i, _, _, rect, _ in tiles:
                if rect.contains(pos):
                    return "tile", i
            return None
        return None

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        hit = self._hit(e.position().toPoint())
        if hit and hit[0] == "tile":
            self.toggled.emit(hit[1])
        elif hit:
            self.shot_clicked.emit(hit[1])

    def mouseDoubleClickEvent(self, e):
        hit = self._hit(e.position().toPoint())
        if hit and hit[0] == "tile":
            self.toggled.emit(hit[1])   # undo the toggle from the first click
            self.opened.emit(hit[1])

    def mouseMoveEvent(self, e):
        hit = self._hit(e.position().toPoint())
        hover = hit[1] if hit and hit[0] == "tile" else None
        if hover != self._hover:
            self._hover = hover
            self.setCursor(Qt.PointingHandCursor if hover is not None else Qt.ArrowCursor)
            self.update()

    def leaveEvent(self, e):
        self._hover = None
        self.update()

    def contextMenuEvent(self, e):
        hit = self._hit(e.pos())
        if not hit or hit[0] != "tile":
            return
        i = hit[1]
        menu = QMenu(self)
        open_act = menu.addAction("Open in viewer")
        sel_act = menu.addAction("Deselect" if i in self.selected else "Select")
        chosen = menu.exec(e.globalPos())
        if chosen == open_act:
            self.opened.emit(i)
        elif chosen == sel_act:
            self.toggled.emit(i)


class ViewerPage(QWidget):
    """Holds keyboard focus for frame stepping."""

    def __init__(self, window):
        super().__init__()
        self.win = window
        self.setFocusPolicy(Qt.StrongFocus)

    def keyPressEvent(self, e):
        k, mods = e.key(), e.modifiers()
        shift = bool(mods & Qt.ShiftModifier)
        if k == Qt.Key_Right:
            self.win.step_seconds(1) if shift else self.win.step(1)
        elif k == Qt.Key_Left:
            self.win.step_seconds(-1) if shift else self.win.step(-1)
        elif k == Qt.Key_PageDown:
            self.win.step_shot(1)
        elif k == Qt.Key_PageUp:
            self.win.step_shot(-1)
        elif k == Qt.Key_Home:
            self.win.go_time(0)
        elif k == Qt.Key_End:
            self.win.go_time(1e12)
        elif k in (Qt.Key_Space, Qt.Key_S):
            self.win.toggle_current()
        else:
            super().keyPressEvent(e)


# main window ----------------------------------------------------------------

def _button(text, slot, tip=None, focus=False):
    b = QPushButton(text)
    b.clicked.connect(slot)
    if tip:
        b.setToolTip(tip)
    if not focus:
        b.setFocusPolicy(Qt.NoFocus)
    return b


class Bridge(QObject):
    """Carries upscaler status from its reader threads to the GUI thread."""
    status = Signal(str)


class ModelsDialog(QDialog):
    """Lists the official Real-ESRGAN models with download buttons."""
    changed = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Get upscaling models")
        self.setMinimumWidth(640)
        self.downloads = []
        v = QVBoxLayout(self)
        intro = QLabel("Official Real-ESRGAN models (BSD-3-Clause license). They download from GitHub "
                       f"into <b>{U.models_dir()}</b>. Any other spandrel-compatible model works too: "
                       "use Add model file… for files you downloaded yourself.")
        intro.setWordWrap(True)
        v.addWidget(intro)
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        for row, (name, mb, desc, url) in enumerate(U.SUGGESTED):
            grid.addWidget(QLabel(f"<b>{name}</b><br><span style='color:{DIM.name()}'>{desc}</span>"), row, 0)
            grid.addWidget(QLabel(f"{mb:g} MB"), row, 1)
            btn = QPushButton()
            btn.clicked.connect(lambda _=False, n=name, u=url, b=btn: self._download(n, u, b))
            self._mark(btn, name)
            grid.addWidget(btn, row, 2)
        v.addLayout(grid)
        more = QPushButton("Browse more models on OpenModelDB")
        more.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://openmodeldb.info")))
        v.addWidget(more)
        note = QLabel("Use models from sources you trust. Prefer .safetensors files; spandrel loads .pth "
                      "files with a restricted unpickler.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {DIM.name()}")
        v.addWidget(note)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        v.addWidget(close, alignment=Qt.AlignRight)

    def _mark(self, btn, name):
        have = (U.models_dir() / name).exists()
        btn.setText("Installed" if have else "Download")
        btn.setEnabled(not have)

    def _download(self, name, url, btn):
        btn.setEnabled(False)
        job = ModelDownload(url, U.models_dir() / name)
        job.progress.connect(lambda d, t: btn.setText(f"{100 * d // t}%" if t else f"{d >> 20} MB"))
        job.succeeded.connect(lambda _: (self._mark(btn, name), self.changed.emit()))
        job.failed.connect(lambda m: (btn.setText("Retry"), btn.setEnabled(True),
                                      QMessageBox.warning(self, "Download failed", f"{name}\n\n{m}")))
        self.downloads.append(job)
        job.start()


class MainWindow(QMainWindow):
    def __init__(self, path=None):
        super().__init__()
        self.setWindowTitle("Wallpaper Frame Picker")
        self.settings = QSettings("wallpaper-frame-picker", "wallpaper-frame-picker")
        self.bridge = Bridge()
        self.bridge.status.connect(self._set_up_status)
        self.upscaler = U.UpscaleClient(on_status=self.bridge.status.emit)
        self.ai_preview = PreviewUpscaler(self.upscaler)
        self.ai_preview.ready.connect(self._on_ai_ready)
        self.ai_preview.start()
        self.model_info = None     # reply to "load" for the chosen model
        self.export_dir = None
        self._tasks = set()
        self.path = None
        self.info = None          # VideoFile kept open for metadata and time math
        self.vdir = None
        self.server = None
        self.worker = None
        self.exporter = None
        self.an = None
        self.selected = {}        # pts -> {"index", "t"}
        self.cur_pts = None
        self.cur_exact = False
        self.want_t = 0.0
        self.rid = 0
        self.shown_rid = -1
        self._build()
        self.setAcceptDrops(True)
        geo = self.settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        else:
            self.resize(1680, 1000)
        if path:
            QTimer.singleShot(0, lambda: self.open_video(path))

    # layout ------------------------------------------------------------------

    def _build(self):
        s = self.settings
        top = QWidget()
        bar = QHBoxLayout(top)
        bar.setContentsMargins(10, 8, 10, 8)
        bar.addWidget(_button("Open video…", self.choose_video, "Ctrl+O"))
        self.video_label = QLabel("No video")
        self.video_label.setStyleSheet(f"color: {DIM.name()}")
        bar.addWidget(self.video_label, 1)
        bar.addWidget(QLabel("Use the video from"))
        self.start_edit = QLineEdit()
        self.end_edit = QLineEdit()
        for ed in (self.start_edit, self.end_edit):
            ed.setFixedWidth(96)
            ed.setPlaceholderText("m:ss.sss")
            ed.setToolTip("A time like 83.5, 1:23.5 or 0:01:23.5")
            ed.returnPressed.connect(self.start_analysis)
        bar.addWidget(self.start_edit)
        bar.addWidget(QLabel("to"))
        bar.addWidget(self.end_edit)
        bar.addWidget(_button("Whole video", self.whole_range))
        self.analyze_btn = _button("Find shots", self.start_analysis,
                                   "Score every frame in the range and split it into shots")
        self.analyze_btn.setDefault(True)
        bar.addWidget(self.analyze_btn)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(330)
        self.progress.hide()
        bar.addWidget(self.progress)
        self.cancel_btn = _button("Cancel", self.cancel_analysis)
        self.cancel_btn.hide()
        bar.addWidget(self.cancel_btn)
        bar.addWidget(_button("About", self.about, "Version, license and source code (F1)"))

        # left: shots
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(10, 4, 4, 10)
        lv.addWidget(self._heading("Shots"))
        self.shot_list = QListWidget()
        self.shot_list.setIconSize(QSize(112, 48))
        self.shot_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.shot_list.setTextElideMode(Qt.ElideRight)
        self.shot_list.currentItemChanged.connect(self._shot_chosen)
        self.shot_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.shot_list.customContextMenuRequested.connect(self._shot_menu)
        lv.addWidget(self.shot_list, 1)
        lv.addWidget(_button("Merge with next shot", self.merge_current,
                             "Join the chosen shot and the one after it"))
        form = QFormLayout()
        form.setContentsMargins(0, 8, 0, 0)
        self.sens = QSlider(Qt.Horizontal)
        self.sens.setRange(0, 100)
        self.sens.setValue(50)
        self.sens.setToolTip("Higher finds more cuts; lower merges more")
        self.sens_label = QLabel("50")
        self._sens_timer = QTimer(singleShot=True, interval=300)
        self._sens_timer.timeout.connect(self._apply_sensitivity)
        self.sens.valueChanged.connect(lambda v: (self.sens_label.setText(str(v)), self._sens_timer.start()))
        row = QHBoxLayout()
        row.addWidget(self.sens, 1)
        row.addWidget(self.sens_label)
        form.addRow("Cut sensitivity", row)
        self.min_frames = QSpinBox()
        self.min_frames.setRange(1, 240)
        self.min_frames.setValue(int(s.value("min_frames", 6)))
        self.min_frames.setSuffix(" frames")
        self.min_frames.setToolTip("Leaves very short shots, such as flash frames, out of the list and sheets")
        self.min_frames.valueChanged.connect(lambda _: self._refresh_shots())
        form.addRow("Hide shots under", self.min_frames)
        self.per_sheet = QSpinBox()
        self.per_sheet.setRange(1, 48)
        self.per_sheet.setValue(int(s.value("per_sheet", 12)))
        self.per_sheet.valueChanged.connect(lambda _: self._refresh_canvas())
        form.addRow("Frames per sheet", self.per_sheet)
        self.tile_size = QSlider(Qt.Horizontal)
        self.tile_size.setRange(160, 720)
        self.tile_size.setValue(int(s.value("tile_size", 320)))
        self.tile_size.valueChanged.connect(lambda _: self._refresh_canvas())
        form.addRow("Tile size", self.tile_size)
        lv.addLayout(form)

        # center: sheets and viewer
        self.canvas = SheetCanvas()
        self.canvas.label = self._frame_label_local
        self.canvas.toggled.connect(lambda i: self.toggle_pts(int(self.an.pts[i])))
        self.canvas.opened.connect(self.open_local_in_viewer)
        self.canvas.shot_clicked.connect(self._select_shot_in_list)
        self.sheet_scroll = QScrollArea()
        self.sheet_scroll.setWidgetResizable(True)
        self.sheet_scroll.setWidget(self.canvas)
        self.sheet_scroll.setFrameShape(QScrollArea.NoFrame)

        self.viewer = ViewerPage(self)
        vv = QVBoxLayout(self.viewer)
        vv.setContentsMargins(0, 0, 0, 0)
        vv.setSpacing(6)
        self.preview = PreviewWidget()
        self.preview.clicked.connect(self.viewer.setFocus)
        self.preview.ai_wanted.connect(self._on_ai_wanted)
        vv.addWidget(self.preview, 1)
        self.strip = SharpnessStrip()
        self.strip.jump.connect(self.go_local)
        self.strip.label = self._strip_tip
        vv.addWidget(self.strip)
        self.timeline = Timeline()
        self.timeline.scrubbed.connect(lambda t, drag: self.go_time(t, drag=drag))
        vv.addWidget(self.timeline)
        ctl = QHBoxLayout()
        ctl.setContentsMargins(8, 0, 8, 0)
        for text, slot, tip in (("⏮ Shot", lambda: self.step_shot(-1), "Page Up"),
                                ("−1 s", lambda: self.step_seconds(-1), "Shift+Left"),
                                ("◀ Frame", lambda: self.step(-1), "Left"),
                                ("Frame ▶", lambda: self.step(1), "Right"),
                                ("+1 s", lambda: self.step_seconds(1), "Shift+Right"),
                                ("Shot ⏭", lambda: self.step_shot(1), "Page Down")):
            ctl.addWidget(_button(text, slot, tip))
        ctl.addStretch(1)
        self.loupe_box = QCheckBox("1:1 loupe")
        self.loupe_box.setChecked(True)
        self.loupe_box.setFocusPolicy(Qt.NoFocus)
        self.loupe_box.setToolTip("Shows real pixels under the cursor; scroll to zoom")
        self.loupe_box.toggled.connect(self._toggle_loupe)
        ctl.addWidget(self.loupe_box)
        self.select_btn = _button("Select frame", self.toggle_current, "Space")
        self.select_btn.setObjectName("primary")
        ctl.addWidget(self.select_btn)
        vv.addLayout(ctl)
        row2 = QHBoxLayout()
        row2.setContentsMargins(8, 0, 8, 8)
        self.info_label = QLabel(" ")
        self.info_label.setContentsMargins(2, 0, 10, 0)
        self.info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.info_label.setWordWrap(True)
        self.info_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)   # wraps instead of widening the window
        row2.addWidget(self.info_label, 1)
        row2.addWidget(_button("Range starts here", lambda: self._range_from_viewer(0),
                               "Use this frame as the start of the part to analyze"))
        row2.addWidget(_button("Range ends here", lambda: self._range_from_viewer(1),
                               "Use this frame as the end of the part to analyze"))
        row2.addWidget(_button("Split shot here", self.split_here, "Start a new shot at this frame"))
        vv.addLayout(row2)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.sheet_scroll, "Sheets")
        self.tabs.addTab(self.viewer, "Viewer")
        self.tabs.currentChanged.connect(lambda i: self.viewer.setFocus() if i == 1 else None)

        # right: selection and export
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 4, 10, 10)
        rv.addWidget(self._heading("Selected frames"))
        self.sel_list = QListWidget()
        self.sel_list.setIconSize(QSize(128, 56))
        self.sel_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.sel_list.itemDoubleClicked.connect(lambda it: self.open_pts_in_viewer(it.data(Qt.UserRole)))
        QShortcut(QKeySequence.Delete, self.sel_list, self.remove_chosen, context=Qt.WidgetShortcut)
        rv.addWidget(self.sel_list, 1)
        row = QHBoxLayout()
        row.addWidget(_button("Remove", self.remove_chosen, "Delete"))
        row.addWidget(_button("Clear all", self.clear_selection))
        rv.addLayout(row)

        rv.addSpacing(10)
        rv.addWidget(self._heading("AI upscaling"))
        self.model_box = QComboBox()
        self.model_box.setToolTip(f"Model files in {U.models_dir()}")
        self.model_box.currentIndexChanged.connect(self._model_changed)
        rv.addWidget(self.model_box)
        row = QHBoxLayout()
        row.addWidget(_button("Add model file…", self._add_model))
        row.addWidget(_button("Get models…", self._get_models))
        rv.addLayout(row)
        self.size_box = QComboBox()
        for text, value in (("Output: the model's own scale", None), ("Output: 2x the video", 2.0),
                            ("Output: same size as the video (clean-up only)", 1.0)):
            self.size_box.addItem(text, value)
        self.size_box.setCurrentIndex(int(s.value("up_size", 0)))
        self.size_box.currentIndexChanged.connect(self._ai_settings_changed)
        rv.addWidget(self.size_box)
        self.keep_orig = QCheckBox("Also save the original frame")
        self.keep_orig.setChecked(s.value("up_keep", "true") == "true")
        rv.addWidget(self.keep_orig)
        self.ai_loupe = QCheckBox("Compare in the Viewer loupe")
        self.ai_loupe.setToolTip("The loupe shows plain resizing next to the model's output")
        self.ai_loupe.setChecked(s.value("up_loupe", "true") == "true")
        self.ai_loupe.toggled.connect(self._ai_settings_changed)
        rv.addWidget(self.ai_loupe)
        self.up_status = QLabel()
        self.up_status.setWordWrap(True)
        self.up_status.setStyleSheet(f"color: {DIM.name()}")
        rv.addWidget(self.up_status)
        rv.addSpacing(6)

        self.export_btn = _button("Export frames…", self.export, "Ctrl+E")
        self.export_btn.setObjectName("primary")
        rv.addWidget(self.export_btn)
        self.export_progress = QProgressBar()
        self.export_progress.hide()
        rv.addWidget(self.export_progress)
        self.open_out_btn = _button("Open export folder", self._open_export_dir)
        self.open_out_btn.hide()
        rv.addWidget(self.open_out_btn)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(self.tabs)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([290, 1100, 300])
        self.splitter = split
        state = s.value("splitter")
        if state is not None:
            split.restoreState(state)

        central = QWidget()
        cv_ = QVBoxLayout(central)
        cv_.setContentsMargins(0, 0, 0, 0)
        cv_.setSpacing(0)
        cv_.addWidget(top)
        cv_.addWidget(split, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Open a video to begin.")

        QShortcut(QKeySequence.Open, self, self.choose_video)
        QShortcut(QKeySequence("Ctrl+E"), self, self.export)
        QShortcut(QKeySequence("Ctrl+1"), self, lambda: self.tabs.setCurrentIndex(0))
        QShortcut(QKeySequence("Ctrl+2"), self, lambda: self.tabs.setCurrentIndex(1))
        QShortcut(QKeySequence("F1"), self, self.about)
        self._populate_models(s.value("up_model", ""), initial=True)
        self._update_export_button()

    def _heading(self, text):
        lab = QLabel(text)
        f = lab.font()
        f.setBold(True)
        f.setPointSizeF(f.pointSizeF() * 1.1)
        lab.setFont(f)
        return lab

    def about(self):
        text = NOTICE.replace("\n\n", "<br><br>").replace("\n", " ")
        QMessageBox.about(self, "About Wallpaper Frame Picker",
                          text.replace(SOURCE_URL, f'<a href="{SOURCE_URL}">{SOURCE_URL}</a>'))

    # opening a video -----------------------------------------------------------

    def choose_video(self):
        start = self.settings.value("video_dir", str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(self, "Open video", start, VIDEO_FILTER,
                                              options=QFileDialog.DontUseNativeDialog)
        if path:
            self.open_video(path)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            if url.isLocalFile():
                self.open_video(url.toLocalFile())
                return

    def open_video(self, path):
        if self.exporter and self.exporter.isRunning():
            QMessageBox.information(self, "Export running", "Wait for the export to finish first.")
            return
        try:
            info = VideoFile(path)
        except Exception as e:
            QMessageBox.critical(self, "Cannot open video", f"{path}\n\n{e}")
            return
        self._stop_threads()
        if self.info:
            self.info.close()
        self.info = info
        self.path = str(Path(path).resolve())
        self.settings.setValue("video_dir", str(Path(self.path).parent))
        self.vdir = A.video_dir(self.path)
        (self.vdir / "selected").mkdir(parents=True, exist_ok=True)
        self.server = FrameServer(self.path)
        self.server.ready.connect(self._on_frame)
        self.server.start()
        self.setWindowTitle(f"{Path(self.path).name} · Wallpaper Frame Picker")
        self.video_label.setText(f"{Path(self.path).name}   ·   {info.width}×{info.height}   ·   "
                                 f"{info.fps:g} fps   ·   {fmt_time(info.duration)}   ·   {info.codec}")
        self.video_label.setToolTip(self.path)
        self.an = None
        self.timeline.duration = info.duration
        self.timeline.range = None
        self.timeline.cuts = []
        self._load_selection()
        saved = A.saved_ranges(self.path)
        if saved:
            t0, t1, folder = saved[0]
            self.start_edit.setText(fmt_time(t0))
            self.end_edit.setText(fmt_time(t1))
            self._load_analysis(folder)
            self.statusBar().showMessage(f"Loaded the saved analysis of {fmt_time(t0)} to {fmt_time(t1)}.")
            self.tabs.setCurrentIndex(0)
        else:
            self.whole_range()
            self._refresh_shots()
            self.statusBar().showMessage("Choose the part of the video to use, then press Find shots. "
                                         "You can also scrub the video in the Viewer and select frames by hand.")
            self.tabs.setCurrentIndex(1)
        self.go_time(0)

    def _stop_threads(self):
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait()
        if self.server:
            self.server.stop()
            self.server = None

    def closeEvent(self, e):
        if self.exporter and self.exporter.isRunning():
            self.exporter.requestInterruption()
            self.exporter.wait()
        self._stop_threads()
        self.ai_preview.stop()
        for task in list(self._tasks):
            task.wait(2000)
        self.upscaler.close()
        s = self.settings
        s.setValue("geometry", self.saveGeometry())
        s.setValue("splitter", self.splitter.saveState())
        s.setValue("min_frames", self.min_frames.value())
        s.setValue("per_sheet", self.per_sheet.value())
        s.setValue("tile_size", self.tile_size.value())
        super().closeEvent(e)

    # analysis ------------------------------------------------------------------

    def whole_range(self):
        if self.info:
            self.start_edit.setText(fmt_time(0))
            self.end_edit.setText(fmt_time(self.info.duration))

    def _range_from_viewer(self, which):
        if self.info is None or self.cur_pts is None:
            return
        t = self.info.t_of(self.cur_pts)
        (self.start_edit if which == 0 else self.end_edit).setText(fmt_time(t))
        self.statusBar().showMessage(f"Range {'start' if which == 0 else 'end'} set to {fmt_time(t)}. "
                                     "Press Find shots to analyze it.")

    def start_analysis(self):
        if self.info is None:
            self.choose_video()
            return
        if self.worker and self.worker.isRunning():
            return
        t0, t1 = parse_time(self.start_edit.text()), parse_time(self.end_edit.text())
        if t0 is None or t1 is None:
            QMessageBox.warning(self, "Range", "Times look like 83.5, 1:23.5 or 0:01:23.5.")
            return
        t1 = min(t1, self.info.duration)
        if t1 - t0 < 2 / self.info.fps:
            QMessageBox.warning(self, "Range", "The end has to come after the start.")
            return
        self.worker = AnalysisWorker(self.path, round(t0, 3), round(t1, 3))
        self.worker.progress.connect(self._on_progress)
        self.worker.succeeded.connect(self._on_analysis_done)
        self.worker.failed.connect(self._on_analysis_failed)
        self.analyze_btn.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.setFormat("Starting…")
        self.progress.show()
        self.cancel_btn.show()
        self.worker.start()

    def cancel_analysis(self):
        if self.worker:
            self.worker.requestInterruption()

    def _on_progress(self, done, expected, fps):
        self.progress.setRange(0, expected)
        self.progress.setValue(done)
        left = (expected - done) / fps if fps > 0 else 0
        self.progress.setFormat(f"{done} / {expected} frames   ·   {fps:.0f} fps   ·   {fmt_time(left)[:-4]} left")

    def _analysis_idle(self):
        self.analyze_btn.setEnabled(True)
        self.progress.hide()
        self.cancel_btn.hide()

    def _on_analysis_done(self, folder):
        self._analysis_idle()
        self._load_analysis(folder)
        m = self.an.meta
        took = f" in {m['seconds']:.0f} s" if "seconds" in m else ""
        self.statusBar().showMessage(f"Found {len(self.an.shots)} shots in {m['frames']} frames{took}.")
        self.tabs.setCurrentIndex(0)

    def _on_analysis_failed(self, msg):
        self._analysis_idle()
        if msg:
            QMessageBox.critical(self, "Analysis failed", msg)
        else:
            self.statusBar().showMessage("Analysis cancelled.")

    def _load_analysis(self, folder):
        self.an = A.Analysis(folder)
        self.sens.blockSignals(True)
        self.sens.setValue(self.an.edits["sensitivity"])
        self.sens_label.setText(str(self.an.edits["sensitivity"]))
        self.sens.blockSignals(False)
        self.timeline.range = (self.an.meta["t0"], self.an.meta["t1"])
        self._refresh_shots()

    def _apply_sensitivity(self):
        if self.an:
            self.an.set_sensitivity(self.sens.value())
            self._refresh_shots()
            self.statusBar().showMessage(f"{len(self.an.shots)} shots at sensitivity {self.sens.value()}.")

    def visible_shots(self):
        if not self.an:
            return []
        m = self.min_frames.value()
        return [s for s, (a, b) in enumerate(self.an.shots) if b - a + 1 >= m]

    def _refresh_shots(self):
        """Rebuilds everything that depends on the shot list."""
        self.shot_list.blockSignals(True)
        self.shot_list.clear()
        if self.an:
            for s in self.visible_shots():
                it = QListWidgetItem()
                it.setData(Qt.UserRole, s)
                img = QImage(str(self.an.thumb_path(self.an.best(s))))
                it.setIcon(QIcon(QPixmap.fromImage(img.scaledToWidth(112, Qt.SmoothTransformation))))
                self.shot_list.addItem(it)
            self._update_shot_texts()
            self.timeline.cuts = [self.info.t_of(int(self.an.pts[a])) for a, _ in self.an.shots[1:]]
        self.shot_list.blockSignals(False)
        self._refresh_canvas()
        self._update_marks()
        self._update_info()

    def _update_shot_texts(self):
        sel = self._selected_locals()
        for row in range(self.shot_list.count()):
            it = self.shot_list.item(row)
            s = it.data(Qt.UserRole)
            a, b = self.an.shots[s]
            n_sel = sum(1 for i in sel if a <= i <= b)
            extra = f"  ✓ {n_sel}" if n_sel else ""
            it.setText(f"S{s:03d}  {self._frame_label_local(a)[1][:-2]}\n{b - a + 1} frames{extra}")
            it.setToolTip(f"S{s:03d}: {b - a + 1} frames from {self._frame_label_local(a)[1]}"
                          + (f", {n_sel} selected" if n_sel else ""))

    def _refresh_canvas(self):
        self.canvas.set_analysis(self.an, self.visible_shots(), self.per_sheet.value(), self.tile_size.value())
        self.canvas.set_selected(self._selected_locals())

    def _shot_chosen(self, item, _prev=None):
        if item is None or not self.an:
            return
        s = item.data(Qt.UserRole)
        top = self.canvas.section_top(s)
        if top is not None:
            self.sheet_scroll.verticalScrollBar().setValue(top - 6)
        if self.tabs.currentIndex() == 1:
            self.go_local(self.an.best(s))

    def _select_shot_in_list(self, shot):
        for row in range(self.shot_list.count()):
            if self.shot_list.item(row).data(Qt.UserRole) == shot:
                self.shot_list.setCurrentRow(row)
                return

    def _current_shot(self):
        it = self.shot_list.currentItem()
        return it.data(Qt.UserRole) if it else None

    def _shot_menu(self, pos):
        it = self.shot_list.itemAt(pos)
        if not it:
            return
        s = it.data(Qt.UserRole)
        menu = QMenu(self)
        view = menu.addAction("Open sharpest frame in viewer")
        merge = menu.addAction("Merge with next shot")
        chosen = menu.exec(self.shot_list.mapToGlobal(pos))
        if chosen == view:
            self.open_local_in_viewer(self.an.best(s))
        elif chosen == merge:
            self._merge(s)

    def merge_current(self):
        s = self._current_shot()
        if s is None:
            self.statusBar().showMessage("Choose a shot in the list first.")
            return
        self._merge(s)

    def _merge(self, s):
        if s + 1 >= len(self.an.shots):
            return
        self.an.merge_with_next(s)
        self._refresh_shots()
        self._select_shot_in_list(s)
        self.statusBar().showMessage(f"Merged S{s:03d} with the next shot.")

    def split_here(self):
        i = self._current_local()
        if i is None:
            self.statusBar().showMessage("Splitting works on frames inside the analyzed range.")
            return
        self.an.split_at(i)
        self._refresh_shots()
        self.statusBar().showMessage(f"New shot starts at frame {self.info.index_of(int(self.an.pts[i]))}.")

    # labels --------------------------------------------------------------------

    def _frame_label_local(self, i):
        pts = int(self.an.pts[i])
        return f"f{self.info.index_of(pts)}", fmt_time(self.info.t_of(pts))

    def _strip_tip(self, i):
        f, t = self._frame_label_local(i)
        return f"{f}  {t}\n{self._sharpness_text(i)}\nmotion {self.an.speed(i):.1f} px/frame"

    def _sharpness_text(self, i):
        if i == self.an.best(self.an.shot_of(i)):
            return "sharpest in shot"
        return f"sharper than {int(self.an.percentile(i))}% of shot"

    def _current_local(self):
        if self.an is None or self.cur_pts is None:
            return None
        return self.an.local(self.cur_pts)

    # viewer --------------------------------------------------------------------

    def _toggle_loupe(self, on):
        self.preview.loupe = on
        self.preview.update()

    def go_time(self, t, exact=True, drag=False):
        if self.info is None or self.server is None:
            return
        t = min(max(t, 0.0), max(0.0, self.info.duration - 1 / self.info.fps))
        self.want_t = t
        pts = self.info.pts_of(t)
        self.rid += 1
        self.timeline.set_time(t)
        if drag:
            i = self.an.nearest(pts) if self.an else None
            if i is not None:   # thumbnails make scrubbing instant inside the analyzed range
                self._show(int(self.an.pts[i]), QImage(str(self.an.thumb_path(i))), False, self.rid,
                           "Preview, release to load the full frame")
            else:
                self.server.request("fast", pts, self.rid)
        else:
            self.server.request("exact", pts, self.rid)

    def go_pts(self, pts):
        self.go_time(self.info.t_of(pts))

    def go_local(self, i):
        self.go_pts(int(self.an.pts[i]))

    def open_local_in_viewer(self, i):
        self.tabs.setCurrentIndex(1)
        self.go_local(i)

    def open_pts_in_viewer(self, pts):
        self.tabs.setCurrentIndex(1)
        self.go_pts(pts)

    def step(self, n):
        if self.info is None:
            return
        if self.an:
            i = self.an.nearest(self.info.pts_of(self.want_t))
            if i is not None and 0 <= i + n < self.an.n:
                self.go_local(i + n)
                return
        self.go_time(self.want_t + n / self.info.fps)

    def step_seconds(self, n):
        self.go_time(self.want_t + n)

    def step_shot(self, n):
        shots = self.visible_shots()
        if not shots:
            return
        i = self.an.nearest(self.info.pts_of(self.want_t))
        if i is None:
            target = shots[0] if self.info.pts_of(self.want_t) < self.an.pts[0] or n > 0 else shots[-1]
        else:
            cur = self.an.shot_of(i)
            later = [s for s in shots if s > cur]
            earlier = [s for s in shots if s < cur]
            target = (later[0] if later else None) if n > 0 else (earlier[-1] if earlier else None)
            if target is None:
                return
        self._select_shot_in_list(target)
        self.go_local(self.an.best(target))

    def _on_frame(self, pts, img, exact, rid):
        if rid < self.shown_rid:
            return
        note = "" if exact else "Nearest keyframe, release to load the exact frame"
        self._show(pts, img, exact, rid, note)

    def _show(self, pts, img, exact, rid, note=""):
        self.shown_rid = rid
        self.cur_pts, self.cur_exact = int(pts), exact
        if exact:
            self.timeline.set_time(self.info.t_of(self.cur_pts))
        self.preview.set_image(img, exact, note)
        self._update_info()

    def _update_info(self):
        if self.info is None or self.cur_pts is None:
            self.info_label.setText(" ")
            self.strip.set_shot(None, None, None)
            return
        pts = self.cur_pts
        parts = [f"Frame {self.info.index_of(pts)}", fmt_time(self.info.t_of(pts))]
        i = self._current_local()
        if i is not None:
            s = self.an.shot_of(i)
            a, b = self.an.shots[s]
            parts.append(f"S{s:03d}, {i - a + 1} of {b - a + 1}")
            parts.append(self._sharpness_text(i))
            parts.append(f"motion {self.an.speed(i):.1f} px")
            cands = [c[0] for c in self.canvas.candidates(s)] if s in self.visible_shots() else []
            self.strip.set_shot(self.an, s, i, [j for j in self._selected_locals() if a <= j <= b], cands)
        else:
            self.strip.set_shot(None, None, None)
            if self.an:
                parts.append("not analyzed")
        if pts in self.selected:
            parts.append("selected")
        if not self.cur_exact:
            parts.append("approximate")
        self.info_label.setText("  ·  ".join(parts))
        self.select_btn.setText("Deselect frame" if pts in self.selected else "Select frame")
        self.select_btn.setEnabled(self.cur_exact)

    # selection ------------------------------------------------------------------

    def _selection_file(self):
        return self.vdir / "selections.json"

    def _load_selection(self):
        self.selected = {}
        f = self._selection_file()
        if f.exists():
            for item in json.loads(f.read_text()):
                pts = int(item["pts"])
                self.selected[pts] = {"index": self.info.index_of(pts), "t": self.info.t_of(pts)}
        self._selection_changed(save=False)

    def _selected_locals(self):
        if not self.an:
            return []
        out = []
        for pts in self.selected:
            i = self.an.local(pts)
            if i is not None:
                out.append(i)
        return out

    def toggle_pts(self, pts):
        pts = int(pts)
        if pts in self.selected:
            del self.selected[pts]
        else:
            self.selected[pts] = {"index": self.info.index_of(pts), "t": self.info.t_of(pts)}
            if (self.an is None or self.an.local(pts) is None) and self.preview.image() is not None \
                    and pts == self.cur_pts:
                self.preview.image().scaledToWidth(320, Qt.SmoothTransformation).save(
                    str(self.vdir / "selected" / f"{pts}.jpg"), quality=88)
        self._selection_changed()

    def toggle_current(self):
        if self.cur_pts is None or not self.cur_exact:
            return
        self.toggle_pts(self.cur_pts)

    def remove_chosen(self):
        for it in self.sel_list.selectedItems():
            self.selected.pop(int(it.data(Qt.UserRole)), None)
        self._selection_changed()

    def clear_selection(self):
        if self.selected and QMessageBox.question(
                self, "Clear selection", f"Remove all {len(self.selected)} selected frames?") == QMessageBox.Yes:
            self.selected.clear()
            self._selection_changed()

    def _shot_label(self, pts):
        i = self.an.local(pts) if self.an else None
        return f"S{self.an.shot_of(i):03d}" if i is not None else ""

    def _selection_changed(self, save=True):
        if save and self.vdir:
            self._selection_file().write_text(json.dumps([{"pts": p} for p in sorted(self.selected)]))
        self.sel_list.clear()
        for pts in sorted(self.selected):
            s = self.selected[pts]
            it = QListWidgetItem(f"f{s['index']:05d}   {fmt_time(s['t'])}\n{self._shot_label(pts) or 'outside analysis'}")
            it.setData(Qt.UserRole, pts)
            i = self.an.local(pts) if self.an else None
            src = self.an.thumb_path(i) if i is not None else self.vdir / "selected" / f"{pts}.jpg"
            if Path(src).exists():
                it.setIcon(QIcon(QPixmap.fromImage(QImage(str(src)).scaledToWidth(128, Qt.SmoothTransformation))))
            self.sel_list.addItem(it)
        self.canvas.set_selected(self._selected_locals())
        if self.an:
            self._update_shot_texts()
        self._update_marks()
        self._update_info()
        self._update_export_button()

    def _update_marks(self):
        self.timeline.marks = [s["t"] for s in self.selected.values()]
        self.timeline.update()

    def _update_export_button(self):
        n = len(self.selected)
        self.export_btn.setText(f"Export {n} frame{'s' if n != 1 else ''}…")
        self.export_btn.setEnabled(n > 0 and not (self.exporter and self.exporter.isRunning()))

    # AI upscaling ----------------------------------------------------------------

    def _run(self, fn, ok, fail):
        task = Task(fn)
        task.succeeded.connect(ok)
        task.failed.connect(fail)
        task.finished.connect(lambda: self._tasks.discard(task))
        self._tasks.add(task)
        task.start()

    def _populate_models(self, choose=None, initial=False):
        current = choose if choose is not None else (self.model_box.currentData() or "")
        if initial and not U.upscaler_installed():
            current = ""   # ask before the 3 GB download only when someone picks a model
        self.model_box.blockSignals(True)
        self.model_box.clear()
        self.model_box.addItem("Off: export frames as decoded", "")
        for p in U.list_models():
            self.model_box.addItem(p.stem, str(p))
        i = max(0, self.model_box.findData(current))
        self.model_box.setCurrentIndex(i)
        self.model_box.blockSignals(False)
        self._model_changed()

    def _model_path(self):
        return self.model_box.currentData() or None

    def _set_up_status(self, text):
        self.up_status.setText(text)

    def _model_changed(self, *_):
        self.model_info = None
        self.preview.set_ai_mode(False)
        path = self._model_path()
        self.settings.setValue("up_model", path or "")
        if not path:
            n = len(U.list_models())
            self._set_up_status("Exports are the frames exactly as decoded." if n else
                                "No models yet. Get models… downloads the official Real-ESRGAN ones.")
            self._update_export_button()
            return
        if not U.upscaler_installed() and not self._confirm_pytorch():
            self.model_box.setCurrentIndex(0)   # back to Off
            return
        self._set_up_status(f"Loading {Path(path).name}…")

        def loaded(info):
            if self._model_path() != path:
                return
            self.model_info = info
            dev = info["device"] + ("" if info["cuda"] else " (no CUDA GPU, this will be slow)")
            self._set_up_status(f"{info['scale']}x {info['arch']} model on {dev}, {info['dtype']}.")
            self._ai_settings_changed()

        def failed(msg):
            if self._model_path() == path:
                self._set_up_status(f"<span style='color:{BAD.name()}'>Could not load {Path(path).name}:</span> "
                                    f"{msg.splitlines()[-1] if msg else ''}")
        self._run(lambda: self.upscaler.load(path), loaded, failed)
        self._update_export_button()

    def _confirm_pytorch(self):
        extra = "" if U.find_uv() else " and the uv package manager (about 20 MB)"
        return QMessageBox.question(
            self, "Install AI upscaling",
            f"AI upscaling runs on PyTorch, which is not installed yet. Wallpaper Frame Picker will download "
            f"PyTorch (about 3 GB){extra} once, into a separate environment. This can take a few "
            f"minutes.\n\nDownload it now?") == QMessageBox.Yes

    def _effective_scale(self):
        """Output size relative to the video, or None before a model loads."""
        if not self.model_info:
            return None
        chosen = self.size_box.currentData()
        return float(self.model_info["scale"]) if chosen is None else chosen

    def _ai_settings_changed(self, *_):
        self.settings.setValue("up_size", self.size_box.currentIndex())
        self.settings.setValue("up_keep", "true" if self.keep_orig.isChecked() else "false")
        self.settings.setValue("up_loupe", "true" if self.ai_loupe.isChecked() else "false")
        on = bool(self.model_info) and self.ai_loupe.isChecked()
        self.preview.set_ai_mode(on, self._effective_scale() or 1.0, Path(self._model_path() or "").stem[:28])

    def _add_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Add upscaling model", str(Path.home()),
                                              "Models (*.pth *.pt *.ckpt *.safetensors)",
                                              options=QFileDialog.DontUseNativeDialog)
        if not path:
            return
        dest = U.models_dir() / Path(path).name
        if Path(path).resolve() != dest.resolve():
            shutil.copy2(path, dest)
        self._populate_models(str(dest))

    def _get_models(self):
        dlg = ModelsDialog(self)
        dlg.changed.connect(lambda: self._populate_models())
        dlg.exec()

    def _on_ai_wanted(self, rect):
        if not (self.model_info and self.cur_exact and self.preview.image() is not None):
            return
        img = self.preview.image()
        pad = 16
        padded = rect.adjusted(-pad, -pad, pad, pad).intersected(img.rect())
        crop = qimage_crop(img, padded)
        self.ai_preview.request(self.cur_pts, rect, crop, (rect.left() - padded.left(), rect.top() - padded.top()),
                                self._model_path(), self.size_box.currentData())

    def _on_ai_ready(self, pts, rect, img):
        if pts == self.cur_pts and self.preview.ai_enabled:
            self.preview.set_ai(rect, img)

    # export --------------------------------------------------------------------

    def export(self):
        if not self.selected or (self.exporter and self.exporter.isRunning()):
            return
        if self._model_path() and not self.model_info:
            QMessageBox.information(self, "Upscaling model", "The model is still loading or failed to load. "
                                    "Wait for it, or set AI upscaling to Off.")
            return
        start = self.settings.value("export_dir", str(Path(self.path).parent))
        out = QFileDialog.getExistingDirectory(self, "Export frames to", start,
                                               QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog)
        if not out:
            return
        self.settings.setValue("export_dir", out)
        self.export_dir = out
        stem = Path(self.path).stem
        jobs = []
        for pts, s in sorted(self.selected.items()):
            shot = self._shot_label(pts)
            jobs.append((pts, str(Path(out) / frame_filename(stem, s["index"], shot))))
        up = None
        if self.model_info:
            scale = self._effective_scale()
            up = {"client": self.upscaler, "model": self._model_path(), "scale": self.size_box.currentData(),
                  "keep": self.keep_orig.isChecked(), "tag": upscale_tag(self._model_path(), scale)}
        self.exporter = ExportWorker(self.path, jobs, up)
        self.exporter.progress.connect(self._on_export_progress)
        self.exporter.succeeded.connect(self._on_export_done)
        self.exporter.failed.connect(lambda m: (self._export_idle(), QMessageBox.critical(self, "Export failed", m)))
        self.export_progress.setRange(0, len(jobs) * 100)
        self.export_progress.setValue(0)
        self.export_progress.setFormat("Starting…")
        self.export_progress.show()
        self.open_out_btn.hide()
        self.exporter.start()
        self._update_export_button()

    def _on_export_progress(self, done, total, stage):
        self.export_progress.setRange(0, total * 100)
        self.export_progress.setValue(round(done * 100))
        self.export_progress.setFormat(f"{stage} {min(int(done) + 1, total)} / {total}")

    def _export_idle(self):
        self.export_progress.hide()
        self._update_export_button()

    def _on_export_done(self, written):
        self._export_idle()
        self.open_out_btn.show()
        self.statusBar().showMessage(f"Wrote {len(written)} PNGs to {self.export_dir}.")

    def _open_export_dir(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.export_dir))


# app -----------------------------------------------------------------------

BORDER = QColor("#353b48")

STYLE = f"""
QWidget {{ font-size: 10pt; }}
QToolTip {{ background: {PANEL.name()}; color: {TEXT.name()}; border: 1px solid {BORDER.name()}; padding: 4px 6px; }}
QMenu {{ background: {PANEL.name()}; border: 1px solid {BORDER.name()}; padding: 4px; }}
QMenu::item {{ padding: 5px 18px; border-radius: 4px; }}
QMenu::item:selected {{ background: #2f3a52; }}
QListWidget {{ background: {PANEL.name()}; border: none; border-radius: 6px; padding: 4px; }}
QListWidget::item {{ padding: 6px; border-radius: 6px; }}
QListWidget::item:selected {{ background: #2f3a52; color: {TEXT.name()}; }}
QPushButton {{ background: #2a2f3a; border: 1px solid {BORDER.name()}; border-radius: 6px; padding: 5px 12px; }}
QPushButton:hover {{ background: #323846; }}
QPushButton:disabled {{ color: #6b717c; }}
QPushButton#primary {{ background: {ACCENT.name()}; border-color: {ACCENT.name()}; color: white; font-weight: 600; }}
QPushButton#primary:hover {{ background: #5d99ff; }}
QPushButton#primary:disabled {{ background: #2a3550; border-color: #2a3550; color: #8c94a3; }}
QLineEdit, QSpinBox {{ background: {PANEL.name()}; border: 1px solid {BORDER.name()}; border-radius: 5px; padding: 3px 6px; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: transparent; padding: 8px 18px; color: {DIM.name()}; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {TEXT.name()}; border-bottom: 2px solid {ACCENT.name()}; }}
QProgressBar {{ background: {PANEL.name()}; border: 1px solid {BORDER.name()}; border-radius: 5px; text-align: center; }}
QProgressBar::chunk {{ background: #2f5fb3; border-radius: 4px; }}
QScrollArea {{ background: {BG.name()}; }}
"""


def dark_palette():
    """Every palette role, so no part of the Fusion style falls back to the
    light defaults (it draws bevels and frames with Light, Mid, Dark and
    Shadow)."""
    pal = QPalette()
    roles = {
        QPalette.Window: BG, QPalette.WindowText: TEXT, QPalette.Base: PANEL, QPalette.AlternateBase: TILE,
        QPalette.ToolTipBase: PANEL, QPalette.ToolTipText: TEXT, QPalette.PlaceholderText: DIM,
        QPalette.Text: TEXT, QPalette.Button: TILE, QPalette.ButtonText: TEXT, QPalette.BrightText: QColor("white"),
        QPalette.Light: QColor("#3a404c"), QPalette.Midlight: QColor("#2f3440"), QPalette.Mid: QColor("#262a33"),
        QPalette.Dark: QColor("#121418"), QPalette.Shadow: QColor("#0b0c0f"),
        QPalette.Highlight: ACCENT, QPalette.HighlightedText: QColor("white"),
        QPalette.Link: QColor("#7aa7ff"), QPalette.LinkVisited: QColor("#a58bff"),
    }
    for role, color in roles.items():
        pal.setColor(role, color)   # all color groups
    for role, color in {QPalette.WindowText: "#6b717c", QPalette.Text: "#6b717c", QPalette.ButtonText: "#6b717c",
                        QPalette.Base: "#1a1c22", QPalette.Button: "#20232a", QPalette.Highlight: "#2a3550",
                        QPalette.HighlightedText: "#8c94a3"}.items():
        pal.setColor(QPalette.Disabled, role, QColor(color))
    return pal


def apply_theme(app):
    """The dark theme, whatever the desktop's own theme is: Fusion style,
    the full dark palette, and a dark color scheme hint, which also makes
    Windows and macOS draw dark title bars."""
    hints = app.styleHints()
    if hasattr(hints, "setColorScheme"):   # Qt 6.8 and newer
        hints.setColorScheme(Qt.ColorScheme.Dark)
    app.setStyle("Fusion")
    app.setPalette(dark_palette())
    app.setStyleSheet(STYLE)


def main(argv):
    # Native file dialogs follow the desktop's colors, which may be light;
    # Qt's own dialogs use the app's dark palette.
    QApplication.setAttribute(Qt.AA_DontUseNativeDialogs)
    app = QApplication(argv)
    app.setApplicationName("Wallpaper Frame Picker")
    apply_theme(app)
    win = MainWindow(argv[1] if len(argv) > 1 else None)
    win.show()
    return app.exec()


def run():
    """Entry point for the wallpaper-frame-picker command."""
    sys.exit(main(sys.argv))
