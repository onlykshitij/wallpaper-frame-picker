# SPDX-License-Identifier: AGPL-3.0-or-later
"""Video decoding with PyAV, exact frame seeking, and YUV to RGB conversion.

Frames are identified by their presentation timestamp (pts, in the stream's
time base), which stays exact for variable frame rate video. Frame numbers
shown to the user are round(time * fps).
"""
from fractions import Fraction

import av
import cv2
import numpy as np

# AVColorSpace value -> (Kr, Kb)
_KRKB = {1: (0.2126, 0.0722),                         # BT.709
         5: (0.299, 0.114), 6: (0.299, 0.114),        # BT.470BG, SMPTE 170M
         9: (0.2627, 0.0593), 10: (0.2627, 0.0593)}   # BT.2020
_FULL_RANGE_FORMATS = {"yuvj420p", "yuvj422p", "yuvj444p"}


def ycbcr_to_rgb_matrix(kr, kb, full_range, order="rgb"):
    """3x4 matrix for cv2.transform that maps 8-bit [Y, Cb, Cr] pixels to
    8-bit RGB (or BGR with order="bgr"). cv2.transform rounds and clips."""
    kg = 1 - kr - kb
    y_off, y_scale, c_scale = (0.0, 1.0, 1.0) if full_range else (16.0, 255 / 219, 255 / 224)
    rows = {"r": (0.0, 2 * (1 - kr)),
            "g": (-2 * kb * (1 - kb) / kg, -2 * kr * (1 - kr) / kg),
            "b": (2 * (1 - kb), 0.0)}
    m = []
    for ch in order:
        cb, cr = rows[ch]
        a, b, c = y_scale, cb * c_scale, cr * c_scale
        m.append([a, b, c, -(a * y_off + (b + c) * 128)])
    return np.array(m, np.float32)


class Planes:
    """Copied Y, Cb and Cr planes of one 4:2:0 frame, plus its color tags."""
    __slots__ = ("pts", "y", "u", "v", "kr_kb", "full_range", "rgb")

    def __init__(self, pts, y, u, v, kr_kb, full_range, rgb=None):
        self.pts, self.y, self.u, self.v = pts, y, u, v
        self.kr_kb, self.full_range, self.rgb = kr_kb, full_range, rgb

    @property
    def nbytes(self):
        if self.rgb is not None:
            return self.rgb.nbytes
        return self.y.nbytes + self.u.nbytes + self.v.nbytes

    def to_rgb(self, width=None, order="rgb"):
        """RGB image, full size or scaled down to `width` pixels wide."""
        if self.rgb is not None:  # source was already RGB
            img = self.rgb if order == "rgb" else self.rgb[:, :, ::-1]
            if width and width < img.shape[1]:
                h = max(1, round(img.shape[0] * width / img.shape[1]))
                img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
            return np.ascontiguousarray(img)
        h, w = self.y.shape
        if width and width < w:
            size = (width, max(1, round(h * width / w)))
            y = cv2.resize(self.y, size, interpolation=cv2.INTER_AREA)
            interp = cv2.INTER_AREA if size[0] <= self.u.shape[1] else cv2.INTER_CUBIC
        else:
            size, y, interp = (w, h), self.y, cv2.INTER_CUBIC
        u = cv2.resize(self.u, size, interpolation=interp)
        v = cv2.resize(self.v, size, interpolation=interp)
        m = ycbcr_to_rgb_matrix(*self.kr_kb, self.full_range, order)
        return cv2.transform(cv2.merge([y, u, v]), m)


def _plane(p):
    a = np.frombuffer(p, np.uint8, count=p.line_size * p.height).reshape(p.height, p.line_size)
    return a[:, :p.width].copy()


class VideoFile:
    """One open video file. Not thread-safe: open one per thread."""

    def __init__(self, path):
        self.path = str(path)
        self.container = av.open(self.path)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        cc = self.stream.codec_context
        self.codec = cc.name
        self.width, self.height = cc.width, cc.height
        self.tb = self.stream.time_base
        rate = self.stream.average_rate or self.stream.guessed_rate or Fraction(24)
        self.fps = float(rate)
        self.start_pts = self.stream.start_time or 0
        if self.stream.duration:
            self.duration = float(self.stream.duration * self.tb)
        elif self.container.duration:
            self.duration = self.container.duration / av.time_base
        else:
            self.duration = 0.0
        self.n_frames = self.stream.frames or max(1, round(self.duration * self.fps))
        self.frame_pts = max(1, round(1 / (self.fps * self.tb)))  # one frame, in pts
        self.kr_kb, self.full_range = self._color_tags(cc.colorspace, cc.color_range, cc.pix_fmt)
        self._iter = None
        self._last = None  # pts of the last decoded frame

    def close(self):
        self.container.close()

    # time conversions ------------------------------------------------------

    def t_of(self, pts):
        return float((pts - self.start_pts) * self.tb)

    def pts_of(self, t):
        return self.start_pts + int(round(t / self.tb))

    def index_of(self, pts):
        return int(round(self.t_of(pts) * self.fps))

    def pts_of_index(self, n):
        return self.pts_of(n / self.fps)

    # decoding ----------------------------------------------------------------

    def _color_tags(self, space, rng, pix_fmt):
        kr_kb = _KRKB.get(int(space or 0))
        if kr_kb is None:  # untagged: BT.709 for HD, BT.601 for SD
            kr_kb = _KRKB[1] if self.height >= 720 else _KRKB[6]
        full = int(rng or 0) == 2 or (pix_fmt or "") in _FULL_RANGE_FORMATS
        return kr_kb, full

    def planes(self, frame):
        """Copies a decoded frame into a Planes object."""
        name = frame.format.name
        if name.startswith(("rgb", "bgr", "gbr", "argb", "abgr")):
            return Planes(frame.pts, None, None, None, None, None,
                          rgb=frame.to_ndarray(format="rgb24"))
        kr_kb, full = self._color_tags(getattr(frame, "colorspace", 0) or 0,
                                       getattr(frame, "color_range", 0) or 0, name)
        if name not in ("yuv420p", "yuvj420p"):
            frame = frame.reformat(format="yuv420p")  # 10-bit, 4:2:2, 4:4:4, nv12 ...
        y, u, v = (_plane(p) for p in frame.planes[:3])
        return Planes(frame.pts, y, u, v, kr_kb, full)

    def _seek(self, pts):
        self.container.seek(max(int(pts), self.start_pts), stream=self.stream,
                            backward=True, any_frame=False)
        self._iter = self.container.decode(self.stream)
        self._last = None

    def _next(self):
        if self._iter is None:
            self._seek(self.start_pts)
        for f in self._iter:
            if f.pts is not None:
                self._last = f.pts
                return f
        self._last = None
        return None

    def _seek_to(self, target, visit=None):
        """Positions the decoder and returns the first frame with
        pts >= target (None past the end). `visit` sees every frame decoded
        on the way, which lets callers cache them."""
        back = 0
        while True:
            self._seek(target - back)
            f = self._next()
            # a seek can land after the target; then go further back
            if f is None or f.pts <= target or target - back <= self.start_pts:
                break
            back = back * 2 + int(1 / self.tb)
        while f is not None and f.pts < target:
            if visit:
                visit(f)
            f = self._next()
        return f

    def frame_at(self, pts, visit=None):
        """The frame showing at `pts`: the first frame with pts at or after
        half a frame before it. Decodes forward without seeking when the
        decoder is already a little before the target."""
        target = pts - self.frame_pts // 2
        if self._last is not None and self._last < target <= self._last + int(2 / self.tb):
            f = self._next()
            while f is not None and f.pts < target:
                if visit:
                    visit(f)
                f = self._next()
            if f is not None:
                return f
        return self._seek_to(target, visit)

    def keyframe_near(self, pts):
        """Fast and approximate: the keyframe at or before `pts`."""
        self._seek(pts)
        return self._next()

    def next_frame(self):
        return self._next()

    def frames(self, t0, t1):
        """Every frame with t0 <= time < t1, in order."""
        half = self.frame_pts // 2
        hi = self.pts_of(t1) - half
        f = self.frame_at(self.pts_of(t0))
        while f is not None and f.pts < hi:
            yield f
            f = self._next()
