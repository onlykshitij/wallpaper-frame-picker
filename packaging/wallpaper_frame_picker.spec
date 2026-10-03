# SPDX-License-Identifier: AGPL-3.0-or-later
# PyInstaller spec for the standalone app. Run through packaging/build.py,
# which first fills build/bundle with the app's wheel, the pinned package
# versions and their download sizes.
#
# The app is a small launcher (launcher.py): Python, Tk for the "Install
# dependencies" pop-up, and the standard library. Qt, PyAV, OpenCV and
# NumPy are installed on first launch, so they are kept out of the bundle.
# On Linux, system libraries are also kept out and taken from the user's
# system, so the launcher runs on any glibc distribution;
# packaging/check_bundle.py verifies that.
import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import copy_metadata

ROOT = Path(SPECPATH).parent
SRC = ROOT / "src"
APP = "WallpaperFramePicker"
LINUX = sys.platform.startswith("linux")
WINDOWS = sys.platform == "win32"
MACOS = sys.platform == "darwin"

datas = [(str(ROOT / "build" / "bundle"), "bundle"), (str(ROOT / "LICENSE"), ".")]
datas += copy_metadata("platformdirs")
EXCLUDES = ["PySide6", "shiboken6", "numpy", "cv2", "av", "torch", "spandrel", "PIL", "pytest"]

HOST_LIBS = re.compile(
    r"^(libc|libm|libmvec|libdl|libpthread|librt|libresolv|libutil|ld-linux.*"   # glibc
    r"|libstdc\+\+|libgcc_s"                                                      # C++ runtime
    r"|libfontconfig|libfreetype|libz|libexpat|libuuid|libdbus-1|libbsd|libmd"
    r"|libX11|libX11-xcb|libXext|libXrender|libXft|libXss|libXau|libXdmcp|libgbm"
    r"|libglib-2\.0|libgobject-2\.0|libgio-2\.0|libgmodule-2\.0|libgthread-2\.0|libffi|libpcre2?(-8)?"
    r"|libselinux|libmount|libblkid|libxml2)\.so")


def keep(entry):
    return not (LINUX and HOST_LIBS.match(Path(entry[0]).name))


def build(entry, name, console):
    a = Analysis([str(ROOT / "packaging" / entry)], pathex=[str(SRC)], datas=datas, excludes=EXCLUDES)
    a.binaries = [b for b in a.binaries if keep(b)]
    pyz = PYZ(a.pure)
    if MACOS:
        exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name=name, console=False, upx=False)
        coll = COLLECT(exe, a.binaries, a.datas, name=name, upx=False)
        BUNDLE(coll, name=f"{name}.app", bundle_identifier="io.github.onlykshitij.wallpaper-frame-picker")
    else:   # one file
        EXE(pyz, a.scripts, a.binaries, a.datas, [], name=name, console=console, upx=False)


# On Linux one file does both: the app, and the command line with `cli`.
build("entry.py", APP, console=not WINDOWS)
if WINDOWS:   # Windows GUI programs have no console, so the CLI is its own program
    build("entry_cli.py", APP + "-cli", console=True)
