# SPDX-License-Identifier: AGPL-3.0-or-later
"""Builds the standalone app with PyInstaller for the current platform.

    uv run --with pyinstaller python packaging/build.py

Writes to dist/:
  Linux    WallpaperFramePicker-linux-<arch>             one file, GUI and `cli`
  Windows  WallpaperFramePicker-windows-<arch>.exe       one file, GUI
           WallpaperFramePicker-cli-windows-<arch>.exe   one file, command line
  macOS    WallpaperFramePicker-macos-<arch>.zip         WallpaperFramePicker.app
"""
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
SEP = ";" if sys.platform == "win32" else ":"
ARCH = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "arm64", "aarch64": "arm64"}.get(
    platform.machine().lower(), platform.machine().lower())
# Bundling the package metadata puts each library's license files in the app.
METADATA = ["wallpaper-frame-picker", "pyside6", "shiboken6", "av", "opencv-python-headless", "numpy", "platformdirs"]


def pyinstaller(entry, name, *extra):
    args = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--name", name,
            "--distpath", str(DIST), "--workpath", str(ROOT / "build" / name), "--specpath", str(ROOT / "build"),
            "--paths", str(ROOT / "src"),
            "--add-data", f"{ROOT / 'src/wallpaper_frame_picker/upscale_server.py'}{SEP}wallpaper_frame_picker",
            "--add-data", f"{ROOT / 'LICENSE'}{SEP}.",
            *[a for m in METADATA for a in ("--copy-metadata", m)],
            *extra, str(ROOT / "packaging" / entry)]
    subprocess.run(args, check=True)


def main():
    if DIST.exists():
        shutil.rmtree(DIST)
    if sys.platform.startswith("linux"):
        pyinstaller("entry.py", "WallpaperFramePicker", "--onefile")
        (DIST / "WallpaperFramePicker").rename(DIST / f"WallpaperFramePicker-linux-{ARCH}")
    elif sys.platform == "win32":
        pyinstaller("entry.py", "WallpaperFramePicker", "--onefile", "--windowed")
        pyinstaller("entry_cli.py", "WallpaperFramePicker-cli", "--onefile", "--console")
        (DIST / "WallpaperFramePicker.exe").rename(DIST / f"WallpaperFramePicker-windows-{ARCH}.exe")
        (DIST / "WallpaperFramePicker-cli.exe").rename(DIST / f"WallpaperFramePicker-cli-windows-{ARCH}.exe")
    elif sys.platform == "darwin":
        pyinstaller("entry.py", "WallpaperFramePicker", "--windowed", "--osx-bundle-identifier", "io.github.wallpaper-frame-picker")
        subprocess.run(["ditto", "-c", "-k", "--keepParent", str(DIST / "WallpaperFramePicker.app"),
                        str(DIST / f"WallpaperFramePicker-macos-{ARCH}.zip")], check=True)
    else:
        sys.exit(f"no build recipe for {sys.platform}")
    for p in sorted(DIST.iterdir()):
        if p.is_file():
            print(f"{p.name}  {p.stat().st_size / 2 ** 20:.0f} MB")


if __name__ == "__main__":
    main()
