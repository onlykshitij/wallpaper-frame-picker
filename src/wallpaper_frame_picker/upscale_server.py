#!/usr/bin/env -S uv run --script
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "spandrel", "numpy", "opencv-python-headless"]
# ///
"""AI upscaling helper: runs spandrel models on PyTorch for Wallpaper Frame Picker.

Wallpaper Frame Picker starts this as a separate process and talks to it with JSON
lines (see serve()). It only imports torch, spandrel, numpy and OpenCV, so
`uv run --script` can run it from the inline metadata above without the
rest of the package.

  python -m wallpaper_frame_picker.upscale_server --info [--model FILE]
  uv run --script upscale_server.py --info

spandrel reads most single-image models from OpenModelDB and the official
Real-ESRGAN releases (.pth, .pt, .ckpt, .safetensors).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from spandrel import ImageModelDescriptor, ModelLoader

TILE_PAD = 32  # context pixels around each tile, cropped off afterwards


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_info():
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return {"cuda": True, "device": p.name, "vram_gb": round(p.total_memory / 2 ** 30, 1)}
    if pick_device().type == "mps":
        return {"cuda": False, "device": "Apple GPU (MPS)", "vram_gb": 0}
    return {"cuda": False, "device": "CPU", "vram_gb": 0}


class Upscaler:
    def __init__(self, path):
        self.path = str(path)
        self.device = pick_device()
        desc = ModelLoader(device=self.device).load_from_file(self.path)
        if not isinstance(desc, ImageModelDescriptor):
            raise ValueError(f"{Path(path).name} is not an image-to-image model")
        if desc.input_channels != 3 or desc.output_channels != 3:
            raise ValueError(f"{Path(path).name} takes {desc.input_channels}-channel images; only RGB models work here")
        self.desc = desc
        self.scale = desc.scale
        self.arch = desc.architecture.name
        self.dtype = torch.float32
        if self.device.type == "cuda":
            if desc.supports_half:
                self.dtype = torch.float16
            elif desc.supports_bfloat16 and torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
        desc.to(self.dtype)
        desc.eval()
        tiling = str(getattr(desc, "tiling", "")).upper()
        self.tile = None if "INTERNAL" in tiling else (768 if self.device.type == "cuda" else 384)

    def describe(self):
        return {"scale": self.scale, "arch": self.arch, "dtype": str(self.dtype).replace("torch.", ""),
                **device_info()}

    @torch.inference_mode()
    def run(self, rgb, progress=None):
        """HxWx3 uint8 RGB -> (H*scale)x(W*scale)x3 uint8 RGB."""
        x = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.dtype).div_(255)
        while True:
            try:
                return self._tiled(x, progress)
            except torch.OutOfMemoryError:
                if self.tile is None:
                    self.tile = 512
                elif self.tile <= 64:
                    raise
                else:
                    self.tile //= 2
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

    def _tiled(self, x, progress):
        _, _, h, w = x.shape
        s = self.scale
        tile = self.tile or max(h, w)
        out = np.empty((h * s, w * s, 3), np.uint8)
        boxes = [(y0, x0) for y0 in range(0, h, tile) for x0 in range(0, w, tile)]
        for k, (y0, x0) in enumerate(boxes):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            py0, px0 = max(y0 - TILE_PAD, 0), max(x0 - TILE_PAD, 0)
            py1, px1 = min(y1 + TILE_PAD, h), min(x1 + TILE_PAD, w)
            o = self.desc(x[:, :, py0:py1, px0:px1])
            o = o[:, :, (y0 - py0) * s:(y1 - py0) * s, (x0 - px0) * s:(x1 - px0) * s]
            o = o.float().clamp_(0, 1).mul_(255).round_().to(torch.uint8)
            out[y0 * s:y1 * s, x0 * s:x1 * s] = o[0].permute(1, 2, 0).cpu().numpy()
            if progress:
                progress(k + 1, len(boxes))
        return out


def resize_to_scale(img, src_shape, scale):
    """Resizes a model output to `scale` times the source size."""
    if scale is None:
        return img
    h, w = src_shape[:2]
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    if size == (img.shape[1], img.shape[0]):
        return img
    shrink = size[0] < img.shape[1]
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LANCZOS4)


def read_rgb(path):
    if str(path).endswith(".npy"):
        return np.load(path)
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_rgb(path, rgb):
    path = Path(path)
    if path.suffix == ".npy":
        np.save(path, rgb)
        return
    ok, data = cv2.imencode(path.suffix or ".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError(f"cannot encode {path}")
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data.tobytes())
    tmp.replace(path)


# server mode -------------------------------------------------------------------

def serve():
    """Line-delimited JSON over stdin and stdout. Every reply carries the
    request's id; long jobs also send {"id", "event": "progress"} lines."""
    out = sys.stdout
    sys.stdout = sys.stderr   # keep stray prints off the reply channel

    def send(msg):
        out.write(json.dumps(msg) + "\n")
        out.flush()

    models = {}

    def get(path):
        path = str(Path(path).resolve())
        if path not in models:
            while len(models) >= 2:   # keep at most two models in memory
                models.pop(next(iter(models)))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            models[path] = Upscaler(path)
        return models[path]

    send({"event": "ready", "torch": torch.__version__, **device_info()})
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        rid = req.get("id")
        cmd = req.get("cmd")
        if cmd == "quit":
            break
        try:
            if cmd == "load":
                send({"id": rid, "ok": True, **get(req["model"]).describe()})
            elif cmd == "upscale":
                up = get(req["model"])
                t0 = time.time()
                src = read_rgb(req["input"])
                res = up.run(src, lambda d, n: send({"id": rid, "event": "progress", "done": d, "total": n}))
                res = resize_to_scale(res, src.shape, req.get("scale"))
                write_rgb(req["output"], res)
                send({"id": rid, "ok": True, "width": res.shape[1], "height": res.shape[0],
                      "seconds": round(time.time() - t0, 3)})
            else:
                send({"id": rid, "ok": False, "error": f"unknown command {cmd!r}"})
        except Exception as e:
            send({"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}"})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serve", action="store_true", help="answer JSON requests on stdin (Wallpaper Frame Picker uses this)")
    ap.add_argument("--info", action="store_true", help="print PyTorch and GPU details")
    ap.add_argument("--model", type=Path, help="with --info, also load this model and describe it")
    a = ap.parse_args()
    if a.serve:
        return serve()
    print(json.dumps({"torch": torch.__version__, **device_info()}, indent=1))
    if a.model:
        print(json.dumps(Upscaler(a.model).describe(), indent=1))


if __name__ == "__main__":
    main()
