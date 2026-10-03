# SPDX-License-Identifier: AGPL-3.0-or-later
"""Builds the standalone app, a small launcher, with PyInstaller for the
current platform.

    uv run --with pyinstaller --with pillow python packaging/build.py

The launcher installs Qt, PyAV, OpenCV and NumPy on first launch, so it
only carries Python, Tk, the app's wheel, and the exact package versions
from uv.lock. Writes to dist/:
  Linux    WallpaperFramePicker-<arch>.AppImage          app and `cli`
  Windows  WallpaperFramePicker-windows-<arch>.exe       one file, app
           WallpaperFramePicker-cli-windows-<arch>.exe   one file, command line
  macOS    WallpaperFramePicker-macos-<arch>.zip         WallpaperFramePicker.app
"""
import json
import os
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


# PySide6 6.10 and newer need glibc 2.34; 6.9.3 is the last release that runs
# on glibc 2.28. So the launcher lets Qt float between the two, and uv picks
# the newest one the user's system supports. Everything else stays pinned.
QT_FOR_OLD_GLIBC = "6.9.3"


def float_qt(requirements):
    out = []
    for line in requirements.splitlines():
        m = re.match(r"^pyside6-essentials==(\d+)\.(\d+)\S*", line)
        if m:
            line = f"pyside6-essentials>={QT_FOR_OLD_GLIBC},<{m[1]}.{int(m[2]) + 1}"
        elif line.startswith("shiboken6=="):
            continue   # follows the PySide6 version
        out.append(line)
    return "\n".join(out) + "\n"


def make_bundle():
    uv = shutil.which("uv") or sys.exit("the build needs uv on PATH")
    shutil.rmtree(BUNDLE, ignore_errors=True)
    BUNDLE.mkdir(parents=True)
    subprocess.run([uv, "build", "--wheel", "--out-dir", str(BUNDLE), str(ROOT)], check=True)
    req = BUNDLE / "requirements.txt"
    subprocess.run([uv, "export", "--project", str(ROOT), "--no-dev", "--no-emit-project", "--no-hashes",
                    "--format", "requirements-txt", "-o", str(req)], check=True)
    sizes = {}   # from the exact versions, before Qt's pin becomes a range
    for line in req.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if m:
            sizes[m[1].lower().replace("_", "-")] = wheel_size(m[1], m[2])
    (BUNDLE / "sizes.json").write_text(json.dumps(sizes, indent=1))
    req.write_text(float_qt(req.read_text()))
    print("bundle:", ", ".join(p.name for p in BUNDLE.iterdir()))


APPIMAGETOOL = ("https://github.com/AppImage/appimagetool/releases/download/continuous/"
                "appimagetool-{arch}.AppImage")
APPRUN = """#!/bin/sh
# Starts Wallpaper Frame Picker from inside the AppImage.
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/lib/wallpaper-frame-picker/WallpaperFramePicker" "$@"
"""
DESKTOP = """[Desktop Entry]
Type=Application
Name=Wallpaper Frame Picker
GenericName=Video frame picker
Comment=Find the sharpest frame of every shot in a video and save it as a wallpaper
Exec=WallpaperFramePicker %f
Icon=wallpaper-frame-picker
Terminal=false
Categories=AudioVideo;Video;
MimeType=video/mp4;video/x-matroska;video/webm;video/quicktime;video/x-msvideo;
Keywords=wallpaper;video;frame;screenshot;sharp;
"""


def appimagetool():
    found = shutil.which("appimagetool")
    if found:
        return found
    tool = ROOT / "build" / "tools" / "appimagetool"
    if not tool.exists():
        tool.parent.mkdir(parents=True, exist_ok=True)
        arch = "x86_64" if ARCH == "x86_64" else "aarch64"
        print("downloading appimagetool")
        urllib.request.urlretrieve(APPIMAGETOOL.format(arch=arch), tool)
        tool.chmod(0o755)
    return str(tool)


def make_appimage():
    """Packs the Linux build folder into dist/WallpaperFramePicker-<arch>.AppImage.
    The AppDir stays in build/AppDir for packaging/check_bundle.py."""
    appdir = ROOT / "build" / "AppDir"
    shutil.rmtree(appdir, ignore_errors=True)
    shutil.copytree(DIST / "WallpaperFramePicker", appdir / "usr" / "lib" / "wallpaper-frame-picker", symlinks=True)
    shutil.rmtree(DIST / "WallpaperFramePicker")
    (appdir / "AppRun").write_text(APPRUN)
    (appdir / "AppRun").chmod(0o755)
    (appdir / "wallpaper-frame-picker.desktop").write_text(DESKTOP)
    apps = appdir / "usr" / "share" / "applications"
    apps.mkdir(parents=True)
    shutil.copy(appdir / "wallpaper-frame-picker.desktop", apps)
    icon = ROOT / "src" / "wallpaper_frame_picker" / "assets" / "icon.png"
    shutil.copy(icon, appdir / "wallpaper-frame-picker.png")
    icons = appdir / "usr" / "share" / "icons" / "hicolor" / "512x512" / "apps"
    icons.mkdir(parents=True)
    shutil.copy(icon, icons / "wallpaper-frame-picker.png")
    (appdir / ".DirIcon").symlink_to("wallpaper-frame-picker.png")
    out = DIST / f"WallpaperFramePicker-{ARCH}.AppImage"
    env = {**os.environ, "ARCH": "x86_64" if ARCH == "x86_64" else "aarch64",
           "APPIMAGE_EXTRACT_AND_RUN": "1"}   # appimagetool is an AppImage too; this runs it without FUSE
    subprocess.run([appimagetool(), str(appdir), str(out)], check=True, env=env)


def main():
    if DIST.exists():
        shutil.rmtree(DIST)
    make_bundle()
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
                    "--distpath", str(DIST), "--workpath", str(ROOT / "build" / "pyinstaller"),
                    str(ROOT / "packaging" / "wallpaper_frame_picker.spec")], check=True)
    if sys.platform.startswith("linux"):
        make_appimage()
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
