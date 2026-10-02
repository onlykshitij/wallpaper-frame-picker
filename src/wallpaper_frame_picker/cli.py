# SPDX-License-Identifier: AGPL-3.0-or-later
"""wallpaper-frame-picker-cli: the app's steps without a window.

  wallpaper-frame-picker-cli analyze VIDEO [--from T] [--to T]
  wallpaper-frame-picker-cli shots VIDEO [--from T --to T] [--min-frames N]
  wallpaper-frame-picker-cli sheets VIDEO OUT [--from T --to T] [--per-sheet N] [--min-frames N]
  wallpaper-frame-picker-cli export VIDEO OUT (--frames N... | --picks FILE | --selected)
                          [--upscale MODEL] [--scale S] [--no-original]
  wallpaper-frame-picker-cli upscale --model MODEL IMAGE... [--scale S] [--out DIR]
  wallpaper-frame-picker-cli models [--download NAME]

Frame numbers count from 0 at the start of the video (time x fps). Times
look like 83.5, 1:23.5 or 0:01:23.5. Analyses, selections and models are
shared with the app.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from . import NOTICE
from . import analysis as A
from . import upscaler as U
from .export import export_frames, frame_filename, upscale_tag
from .timefmt import fmt_time, parse_time
from .video import VideoFile


def err(*args, **kw):
    print(*args, file=sys.stderr, flush=True, **kw)


def time_arg(text):
    t = parse_time(text)
    if t is None:
        raise argparse.ArgumentTypeError(f"{text!r} is not a time like 83.5, 1:23.5 or 0:01:23.5")
    return t


def progress_line(done, total, rate):
    err(f"\r  {done} / {total} frames, {rate:.0f} fps", end="")


def load_analysis(video, t0=None, t1=None, run=True):
    """The analysis of [t0, t1]. Without a range: the whole-video analysis
    if there is one, else the newest saved one, else a new whole-video
    analysis. Runs the analysis when it is missing and `run` is true."""
    info = VideoFile(video)
    info.close()
    if t0 is None and t1 is None:
        saved = A.saved_ranges(video)
        whole = [r for r in saved if r[0] == 0 and r[1] >= round(info.duration, 3) - 0.001]
        if whole or saved:
            return A.Analysis((whole or saved)[0][2])
    t0 = round(t0 or 0.0, 3)
    t1 = round(min(t1 if t1 is not None else info.duration, info.duration), 3)
    folder = A.range_dir(video, t0, t1)
    if not (folder / "meta.json").exists():
        if not run:
            return None
        err(f"Analyzing {fmt_time(t0)} to {fmt_time(t1)} of {Path(video).name}")
        folder = A.analyze(video, t0, t1, progress=progress_line)
        err()
    return A.Analysis(folder)


def visible_shots(an, min_frames):
    return [s for s, (a, b) in enumerate(an.shots) if b - a + 1 >= min_frames]


# sheets ----------------------------------------------------------------------

def _tile(img, label, width):
    h = round(img.shape[0] * width / img.shape[1])
    img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
    bar = np.full((22, width, 3), 37, np.uint8)
    cv2.putText(bar, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([img, bar])


def _grid(tiles, cols, gap=6):
    h, w = tiles[0].shape[:2]
    rows = -(-len(tiles) // cols)
    out = np.full((rows * (h + gap) + gap, min(cols, len(tiles)) * (w + gap) + gap, 3), 24, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        out[gap + r * (h + gap):gap + r * (h + gap) + h, gap + c * (w + gap):gap + c * (w + gap) + w] = t
    return out


def _thumb(an, i):
    return cv2.imdecode(np.fromfile(str(an.thumb_path(i)), np.uint8), cv2.IMREAD_COLOR)


def _save_jpeg(path, img):
    ok, data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    Path(path).write_bytes(data.tobytes())


def read_picks(path):
    picks = []
    for no, line in enumerate(Path(path).read_text().splitlines(), 1):
        words = line.split("#", 1)[0].split()
        if not words:
            continue
        if len(words) != 2 or not words[1].isdigit():
            sys.exit(f"{path} line {no}: expected '<name> <frame>', got {line!r}")
        picks.append((words[0], int(words[1])))
    return picks


# commands ----------------------------------------------------------------------

def cmd_analyze(a):
    an = load_analysis(a.video, a.start if a.start is not None else 0.0, a.end)
    m = an.meta
    print(f"{m['frames']} frames from {fmt_time(m['t0'])} to {fmt_time(m['t1'])}, {len(an.shots)} shots")
    print(f"saved in {an.folder}")


def cmd_shots(a):
    an = load_analysis(a.video, a.start, a.end)
    info = VideoFile(a.video)
    print(f"{'shot':6} {'first':>7} {'last':>7} {'frames':>6}  {'start':>10}  sharpest")
    for s in visible_shots(an, a.min_frames):
        lo, hi = an.shots[s]
        first, last = info.index_of(int(an.pts[lo])), info.index_of(int(an.pts[hi]))
        best = info.index_of(int(an.pts[an.best(s)]))
        print(f"S{s:03d}   {first:7d} {last:7d} {hi - lo + 1:6d}  {fmt_time(info.t_of(int(an.pts[lo]))):>10}  {best}")
    info.close()


def cmd_sheets(a):
    an = load_analysis(a.video, a.start, a.end)
    info = VideoFile(a.video)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shots = visible_shots(an, a.min_frames)
    label = lambda i: (info.index_of(int(an.pts[i])), fmt_time(info.t_of(int(an.pts[i]))))

    for p in range(0, len(shots), 40):
        tiles = []
        for s in shots[p:p + 40]:
            lo, hi = an.shots[s]
            tiles.append(_tile(_thumb(an, an.best(s)), f"S{s:03d}  {label(lo)[1]}  ({hi - lo + 1}f)", 384))
        _save_jpeg(out / f"overview_{p // 40}.jpg", _grid(tiles, 5))

    picks_path = out / "picks.txt"
    have = {n for n, _ in read_picks(picks_path)} if picks_path.exists() else set()
    added = []
    for s in shots:
        cands = an.candidates(s, a.per_sheet)
        best = max(cands, key=lambda c: c[1])[0]
        tiles = [_tile(_thumb(an, i), f"{'* ' if i == best else ''}f{label(i)[0]} {label(i)[1]}  "
                                      f"s{pct} v{v:.1f}", 320) for i, pct, v in cands]
        _save_jpeg(out / f"S{s:03d}.jpg", _grid(tiles, 4))
        if f"S{s:03d}" not in have:
            added.append(f"S{s:03d} {label(best)[0]}  # sharpest frame; compare with S{s:03d}.jpg")
    if added:
        with open(picks_path, "a") as fh:
            fh.write("\n".join(added) + "\n")
    info.close()
    print(f"wrote {-(-len(shots) // 40)} overview sheets and {len(shots)} shot sheets to {out}")
    print(f"{picks_path} has a line per shot set to its sharpest frame; edit the numbers, then run export --picks")


def resolve_model(name):
    p = Path(name)
    if p.exists():
        return p
    for m in U.list_models():
        if name in (m.name, m.stem):
            return m
    sys.exit(f"no model {name!r}; installed models: {', '.join(m.stem for m in U.list_models()) or 'none'}")


def cmd_export(a):
    info = VideoFile(a.video)
    an = load_analysis(a.video, run=False)
    if a.selected:
        f = A.video_dir(a.video) / "selections.json"
        items = [(None, int(x["pts"])) for x in json.loads(f.read_text())] if f.exists() else []
        if not items:
            sys.exit("this video has no selected frames yet; select some in the app first")
    elif a.picks:
        items = [(name, info.pts_of_index(n)) for name, n in read_picks(a.picks)]
    elif a.frames:
        items = [(None, info.pts_of_index(n)) for n in a.frames]
    else:
        sys.exit("give --frames, --picks or --selected")

    def shot_label(pts):
        i = an.local(pts) if an else None
        return f"S{an.shot_of(i):03d}" if i is not None else ""

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.video).stem
    jobs = [(pts, str(out / frame_filename(stem, info.index_of(pts), name or shot_label(pts)))) for name, pts in items]
    up = None
    client = None
    if a.upscale:
        model = resolve_model(a.upscale)
        client = U.UpscaleClient(on_status=err)
        loaded = client.load(model)
        scale = a.scale if a.scale is not None else float(loaded["scale"])
        up = {"client": client, "model": str(model), "scale": a.scale, "keep": not a.no_original,
              "tag": upscale_tag(model, scale)}
    try:
        written = export_frames(a.video, jobs, up,
                                progress=lambda d, n, stage: err(f"\r  {stage} {min(int(d) + 1, n)} / {n}", end=""))
    finally:
        if client:
            client.close()
    err()
    for p in written:
        print(p)


def cmd_upscale(a):
    model = resolve_model(a.model)
    client = U.UpscaleClient(on_status=err)
    try:
        loaded = client.load(model)
        err(f"{model.name}: {loaded['scale']}x {loaded['arch']} on {loaded['device']} ({loaded['dtype']})")
        scale = a.scale if a.scale is not None else float(loaded["scale"])
        for img_path in a.images:
            img = cv2.imdecode(np.fromfile(str(img_path), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                err(f"cannot read {img_path}, skipped")
                continue
            folder = a.out or img_path.parent
            folder.mkdir(parents=True, exist_ok=True)
            dest = folder / f"{img_path.stem}_{upscale_tag(model, scale)}.png"
            r = client.upscale_to_file(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), model, dest, a.scale)
            print(f"{dest}  {r['width']}x{r['height']}  {r['seconds']:.1f}s")
    finally:
        client.close()


def cmd_models(a):
    if a.download:
        for name, mb, desc, url in U.SUGGESTED:
            if a.download in (name, Path(name).stem):
                dest = U.models_dir() / name
                U.download_model(url, dest, lambda d, t: err(f"\r  {d >> 20} / {t >> 20} MB", end=""))
                err()
                print(dest)
                return
        sys.exit(f"{a.download!r} is not in the list below; download other models yourself and put them in "
                 f"{U.models_dir()}")
    print(f"models folder: {U.models_dir()}")
    installed = U.list_models()
    for m in installed:
        print(f"  {m.stem}  ({m.stat().st_size / 2 ** 20:.0f} MB)")
    if not installed:
        print("  (none)")
    print("\nofficial Real-ESRGAN models, get them with --download NAME:")
    for name, mb, desc, url in U.SUGGESTED:
        mark = "installed" if (U.models_dir() / name).exists() else f"{mb:g} MB"
        print(f"  {Path(name).stem:28} {desc} ({mark})")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wallpaper-frame-picker-cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=NOTICE)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_range(p):
        p.add_argument("--from", dest="start", type=time_arg, help="start of the part to use")
        p.add_argument("--to", dest="end", type=time_arg, help="end of the part to use")

    p = sub.add_parser("analyze", help="score every frame and split the video into shots")
    p.add_argument("video", type=Path)
    add_range(p)
    p.set_defaults(fn=cmd_analyze)

    p = sub.add_parser("shots", help="list the shots")
    p.add_argument("video", type=Path)
    add_range(p)
    p.add_argument("--min-frames", type=int, default=6, help="leave out shorter shots (default 6)")
    p.set_defaults(fn=cmd_shots)

    p = sub.add_parser("sheets", help="contact sheets for every shot, plus a draft picks file")
    p.add_argument("video", type=Path)
    p.add_argument("out", type=Path)
    add_range(p)
    p.add_argument("--min-frames", type=int, default=6, help="leave out shorter shots (default 6)")
    p.add_argument("--per-sheet", type=int, default=12, help="frames per shot sheet (default 12)")
    p.set_defaults(fn=cmd_sheets)

    p = sub.add_parser("export", help="write frames as full-resolution PNGs")
    p.add_argument("video", type=Path)
    p.add_argument("out", type=Path)
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--frames", type=int, nargs="+", metavar="N", help="frame numbers")
    which.add_argument("--picks", type=Path, help="file of '<name> <frame>' lines, as sheets writes")
    which.add_argument("--selected", action="store_true", help="the frames selected in the app")
    p.add_argument("--upscale", metavar="MODEL", help="also upscale with this model (name or file)")
    p.add_argument("--scale", type=float, help="upscaled size relative to the video (default: the model's scale)")
    p.add_argument("--no-original", action="store_true", help="with --upscale, skip the plain frame")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("upscale", help="upscale image files")
    p.add_argument("images", type=Path, nargs="+")
    p.add_argument("--model", required=True, help="model name or file")
    p.add_argument("--scale", type=float, help="output size relative to the input (default: the model's scale)")
    p.add_argument("--out", type=Path, help="output folder (default: next to each image)")
    p.set_defaults(fn=cmd_upscale)

    p = sub.add_parser("models", help="list installed models, or download an official one")
    p.add_argument("--download", metavar="NAME", help="e.g. realesr-animevideov3")
    p.set_defaults(fn=cmd_models)

    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except U.UpscalerError as e:
        sys.exit(f"upscaling failed: {e}")


if __name__ == "__main__":
    main()
