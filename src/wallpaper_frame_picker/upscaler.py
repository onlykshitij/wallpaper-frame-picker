# SPDX-License-Identifier: AGPL-3.0-or-later
"""Talks to upscale_server.py running in its own process.

With the `upscale` extra installed, the helper runs on this environment's
Python. Without it, `uv run --script upscale_server.py` builds a separate
environment from the script's inline metadata, so PyTorch (several GB) is
only downloaded once someone uses upscaling. Requests and replies are JSON
lines; images go through .npy files in a temporary folder.
"""
import importlib.util
import itertools
import json
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import deque
from pathlib import Path

import numpy as np

from .paths import data_dir, models_dir
from .uvtools import SetupError, ensure_uv, find_uv, install_uv  # noqa: F401 (re-exported)
from .uvtools import download as download_model  # noqa: F401 (re-exported)

SCRIPT = Path(__file__).resolve().with_name("upscale_server.py")
MODEL_SUFFIXES = (".pth", ".pt", ".ckpt", ".safetensors")

# Official Real-ESRGAN releases (BSD-3-Clause), offered in "Get models…".
SUGGESTED = [
    ("RealESRGAN_x4plus_anime_6B.pth", 17.9, "4x, for anime and illustration",
     "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"),
    ("realesr-animevideov3.pth", 2.5, "4x, small and fast, for anime video",
     "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"),
    ("RealESRGAN_x4plus.pth", 67.0, "4x, for photos and live action",
     "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"),
    ("RealESRGAN_x2plus.pth", 67.1, "2x, for photos and live action",
     "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"),
    ("realesr-general-x4v3.pth", 4.9, "4x, small and fast, general purpose",
     "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"),
]


def torch_here():
    """True when PyTorch and spandrel are installed in this environment
    (the `upscale` extra), so the helper can run without uv."""
    if getattr(sys, "frozen", False):
        return False
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "spandrel"))


def _ready_marker():
    return data_dir() / "upscaler-ready"


def upscaler_installed():
    """True once PyTorch is in place: the upscale extra is installed, or
    the upscaler has started successfully before."""
    return torch_here() or _ready_marker().exists()


def list_models():
    return sorted(p for p in models_dir().iterdir() if p.suffix.lower() in MODEL_SUFFIXES)


class UpscalerError(SetupError):
    pass


class UpscaleClient:
    """Thread-safe: one request at a time, from any thread."""

    def __init__(self, on_status=None):
        self.on_status = on_status or (lambda text: None)
        self.proc = None
        self.info = None
        self._lock = threading.Lock()        # one request at a time
        self._start_lock = threading.Lock()
        self._ids = itertools.count(1)
        self._replies = {}
        self._cond = threading.Condition()
        self._log = deque(maxlen=40)
        self.tmp = Path(tempfile.mkdtemp(prefix="wallpaper-frame-picker-"))

    # process -----------------------------------------------------------------

    def _command(self):
        if torch_here():
            return [sys.executable, "-m", "wallpaper_frame_picker.upscale_server", "--serve"]
        uv = ensure_uv(self.on_status)
        # PyTorch goes into its own environment, made from the script's inline metadata
        return [uv, "run", "--script", str(SCRIPT), "--serve"]

    def start(self, timeout=1800):
        """Starts the helper if needed and waits until it is ready."""
        with self._start_lock:
            if self.proc and self.proc.poll() is None and self.info:
                return self.info
            cmd = self._command()
            self.on_status("Starting the upscaler…" if upscaler_installed() else
                           "Starting the upscaler. The first start downloads PyTorch (about 3 GB)…")
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True, bufsize=1)
            self.info = None
            threading.Thread(target=self._read_stdout, daemon=True).start()
            threading.Thread(target=self._read_stderr, daemon=True).start()
            with self._cond:
                ok = self._cond.wait_for(lambda: self.info is not None or self.proc.poll() is not None, timeout)
            if not ok or self.info is None:
                self.stop()
                raise UpscalerError("The upscaler did not start:\n" + "\n".join(self._log))
            _ready_marker().touch()
            dev = self.info["device"] + (f", {self.info['vram_gb']} GB" if self.info["cuda"] else ", no CUDA GPU found")
            self.on_status(f"Upscaler ready on {dev}.")
            return self.info

    def _read_stdout(self):
        for line in self.proc.stdout:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                self._log.append(line.rstrip())
                continue
            with self._cond:
                if msg.get("event") == "ready":
                    self.info = msg
                elif msg.get("event") == "progress":
                    cb = self._replies.get(("progress", msg["id"]))
                    if cb:
                        cb(msg["done"], msg["total"])
                    continue
                else:
                    self._replies[msg.get("id")] = msg
                self._cond.notify_all()
        with self._cond:
            self._cond.notify_all()

    def _read_stderr(self):
        for line in self.proc.stderr:
            line = line.rstrip()
            if line:
                self._log.append(line)
                if self.info is None:   # uv's download and install lines during the first start
                    self.on_status(f"Starting the upscaler: {line[-120:]}")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        self.proc = None
        self.info = None

    def close(self):
        self.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    # requests --------------------------------------------------------------

    def call(self, cmd, on_progress=None, timeout=3600, **args):
        self.start()
        with self._lock:
            rid = next(self._ids)
            if on_progress:
                self._replies[("progress", rid)] = on_progress
            try:
                self.proc.stdin.write(json.dumps({"id": rid, "cmd": cmd, **args}) + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError):
                raise UpscalerError("The upscaler stopped:\n" + "\n".join(self._log))
            with self._cond:
                done = self._cond.wait_for(lambda: rid in self._replies or self.proc.poll() is not None, timeout)
                reply = self._replies.pop(rid, None)
                self._replies.pop(("progress", rid), None)
            if reply is None:
                raise UpscalerError("The upscaler stopped:\n" + "\n".join(self._log) if done else "timed out")
            if not reply.get("ok"):
                raise UpscalerError(reply.get("error", "unknown error"))
            return reply

    def load(self, model):
        return self.call("load", model=str(model))

    def upscale_array(self, rgb, model, scale=None):
        """RGB array in, RGB array out (used for the loupe preview)."""
        src = self.tmp / f"in_{threading.get_ident()}.npy"
        dst = self.tmp / f"out_{threading.get_ident()}.npy"
        np.save(src, np.ascontiguousarray(rgb))
        self.call("upscale", model=str(model), input=str(src), output=str(dst), scale=scale)
        return np.load(dst)

    def upscale_to_file(self, rgb, model, out_path, scale=None, on_progress=None):
        src = self.tmp / f"in_{threading.get_ident()}.npy"
        np.save(src, np.ascontiguousarray(rgb))
        return self.call("upscale", on_progress=on_progress, model=str(model), input=str(src),
                         output=str(out_path), scale=scale)
