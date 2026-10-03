# SPDX-License-Identifier: AGPL-3.0-or-later
"""Builds the standalone app, a small launcher, with PyInstaller for the
current platform.

    uv run --with pyinstaller python packaging/build.py

The launcher installs Qt, PyAV, OpenCV and NumPy on first launch, so it
only carries Python, Tk, the app's wheel, and the exact package versions
from uv.lock. Writes to dist/:
  Linux    WallpaperFramePicker-linux-<arch>             one file, app and `cli`
  Windows  WallpaperFramePicker-windows-<arch>.exe       one file, app
           WallpaperFramePicker-cli-windows-<arch>.exe   one file, command line
  macOS    WallpaperFramePicker-macos-<arch>.zip         WallpaperFramePicker.app
"""
import json
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
BUNDLE = ROOT / "build" / "bundle"
ARCH = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "arm64", "aarch64": "arm64"}.get(
    platform.machine().lower(), platform.machine().lower())


def wheel_size(name, version):
    """Download size of the wheel this platform will get, from PyPI."""
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=30) as r:
            files = json.load(r)["urls"]
    except Exception:
        return None
    if sys.platform.startswith("linux"):
        fits = lambda f: "manylinux" in f and ("x86_64" if ARCH == "x86_64" else "aarch64") in f
    elif sys.platform == "win32":
        fits = lambda f: "win_amd64" in f if ARCH == "x86_64" else "win_arm64" in f
    else:
        fits = lambda f: "macosx" in f and (ARCH in f or "universal2" in f)
    pyver = f"cp{sys.version_info.major}{sys.version_info.minor}"
    sizes = [f["size"] for f in files if f["packagetype"] == "bdist_wheel" and fits(f["filename"])
             and (pyver in f["filename"] or "abi3" in f["filename"] or "-py3-" in f["filename"])]
    return max(sizes) if sizes else None


def make_bundle():
    uv = shutil.which("uv") or sys.exit("the build needs uv on PATH")
    shutil.rmtree(BUNDLE, ignore_errors=True)
    BUNDLE.mkdir(parents=True)
    subprocess.run([uv, "build", "--wheel", "--out-dir", str(BUNDLE), str(ROOT)], check=True)
    req = BUNDLE / "requirements.txt"
    subprocess.run([uv, "export", "--project", str(ROOT), "--no-dev", "--no-emit-project", "--no-hashes",
                    "--format", "requirements-txt", "-o", str(req)], check=True)
    sizes = {}
    for line in req.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if m:
            sizes[m[1].lower().replace("_", "-")] = wheel_size(m[1], m[2])
    (BUNDLE / "sizes.json").write_text(json.dumps(sizes, indent=1))
    print("bundle:", ", ".join(p.name for p in BUNDLE.iterdir()))


def main():
    if DIST.exists():
        shutil.rmtree(DIST)
    make_bundle()
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
                    "--distpath", str(DIST), "--workpath", str(ROOT / "build" / "pyinstaller"),
                    str(ROOT / "packaging" / "wallpaper_frame_picker.spec")], check=True)
    if sys.platform.startswith("linux"):
        (DIST / "WallpaperFramePicker").rename(DIST / f"WallpaperFramePicker-linux-{ARCH}")
    elif sys.platform == "win32":
        (DIST / "WallpaperFramePicker.exe").rename(DIST / f"WallpaperFramePicker-windows-{ARCH}.exe")
        (DIST / "WallpaperFramePicker-cli.exe").rename(DIST / f"WallpaperFramePicker-cli-windows-{ARCH}.exe")
    elif sys.platform == "darwin":
        subprocess.run(["ditto", "-c", "-k", "--keepParent", str(DIST / "WallpaperFramePicker.app"),
                        str(DIST / f"WallpaperFramePicker-macos-{ARCH}.zip")], check=True)
    else:
        sys.exit(f"no build recipe for {sys.platform}")
    for p in sorted(DIST.iterdir()):
        if p.is_file():
            print(f"{p.name}  {p.stat().st_size / 2 ** 20:.0f} MB")


if __name__ == "__main__":
    main()
