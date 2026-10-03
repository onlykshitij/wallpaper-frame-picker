# SPDX-License-Identifier: AGPL-3.0-or-later
"""The system libraries Qt needs on Linux, and installing missing ones with
the distribution's package manager.

Standard library only: this runs before Qt is imported, and in the
standalone launcher before anything else is installed.
"""
import ctypes
import os
import platform
import shutil
import subprocess
import sys

from .uvtools import SetupError

# library -> (what it is for, apt package candidates (first available wins), pacman package)
CORE = {
    "libGL.so.1": ("OpenGL", ["libgl1"], "libglvnd"),
    "libEGL.so.1": ("OpenGL", ["libegl1"], "libglvnd"),
    "libfontconfig.so.1": ("fonts", ["libfontconfig1"], "fontconfig"),
    "libfreetype.so.6": ("fonts", ["libfreetype6"], "freetype2"),
    "libxkbcommon.so.0": ("keyboard", ["libxkbcommon0"], "libxkbcommon"),
    "libX11.so.6": ("X11", ["libx11-6"], "libx11"),
    "libdbus-1.so.3": ("D-Bus", ["libdbus-1-3"], "dbus"),
    "libglib-2.0.so.0": ("GLib", ["libglib2.0-0t64", "libglib2.0-0"], "glib2"),
    "libgthread-2.0.so.0": ("GLib", ["libglib2.0-0t64", "libglib2.0-0"], "glib2"),
    "libz.so.1": ("zlib", ["zlib1g"], "zlib"),
    "libzstd.so.1": ("zstd", ["libzstd1"], "zstd"),
}
WAYLAND = {
    "libwayland-client.so.0": ("Wayland", ["libwayland-client0"], "wayland"),
    "libwayland-cursor.so.0": ("Wayland", ["libwayland-cursor0"], "wayland"),
    "libwayland-egl.so.1": ("Wayland", ["libwayland-egl1"], "wayland"),
}
X11 = {
    "libxcb.so.1": ("X11", ["libxcb1"], "libxcb"),
    "libX11-xcb.so.1": ("X11", ["libx11-xcb1"], "libx11"),
    "libxkbcommon-x11.so.0": ("X11 keyboard", ["libxkbcommon-x11-0"], "libxkbcommon-x11"),
    "libxcb-cursor.so.0": ("X11 cursors", ["libxcb-cursor0"], "xcb-util-cursor"),
    "libxcb-icccm.so.4": ("X11", ["libxcb-icccm4"], "xcb-util-wm"),
    "libxcb-util.so.1": ("X11", ["libxcb-util1"], "xcb-util"),
    "libxcb-image.so.0": ("X11", ["libxcb-image0"], "xcb-util-image"),
    "libxcb-keysyms.so.1": ("X11", ["libxcb-keysyms1"], "xcb-util-keysyms"),
    "libxcb-randr.so.0": ("X11", ["libxcb-randr0"], "libxcb"),
    "libxcb-render.so.0": ("X11", ["libxcb-render0"], "libxcb"),
    "libxcb-render-util.so.0": ("X11", ["libxcb-render-util0"], "xcb-util-renderutil"),
    "libxcb-shape.so.0": ("X11", ["libxcb-shape0"], "libxcb"),
    "libxcb-shm.so.0": ("X11", ["libxcb-shm0"], "libxcb"),
    "libxcb-sync.so.1": ("X11", ["libxcb-sync1"], "libxcb"),
    "libxcb-xfixes.so.0": ("X11", ["libxcb-xfixes0"], "libxcb"),
    "libxcb-xkb.so.1": ("X11", ["libxcb-xkb1"], "libxcb"),
}
ALL = {**CORE, **WAYLAND, **X11}


def needed():
    """The libraries Qt needs in this session: the core set, plus the
    Wayland or the X11 set depending on which one Qt will use."""
    if not sys.platform.startswith("linux"):
        return []
    plat = os.environ.get("QT_QPA_PLATFORM", "").split(";")[0]
    libs = list(CORE)
    if plat in ("offscreen", "minimal", "vnc", "eglfs", "linuxfb"):
        return libs
    wayland = plat == "wayland" or (plat == "" and os.environ.get("WAYLAND_DISPLAY"))
    return libs + list(WAYLAND if wayland else X11)


def loadable(soname):
    try:
        ctypes.CDLL(soname)
        return True
    except OSError:
        return False


def missing():
    return [lib for lib in needed() if not loadable(lib)]


def package_manager():
    for pm in ("apt-get", "dnf", "yum", "zypper", "pacman"):
        if shutil.which(pm):
            return pm
    return None


def _capability(lib):
    """RPM capability for a 64-bit library, which dnf and zypper install by."""
    return f"{lib}()(64bit)" if platform.architecture()[0] == "64bit" else lib


def install_plan(libs):
    """(packages to show, shell script that installs them as root), or
    (sonames, None) when there is no supported package manager."""
    pm = package_manager()
    if pm == "apt-get":
        groups = list(dict.fromkeys(tuple(ALL[lib][1]) for lib in libs))
        pick = " ".join(
            "for p in " + " ".join(g) + '; do if apt-cache show "$p" >/dev/null 2>&1; then pkgs="$pkgs $p"; break; fi; done;'
            for g in groups)
        script = ("set -e; export DEBIAN_FRONTEND=noninteractive; apt-get update -qq; pkgs=; " + pick +
                  " apt-get install -y --no-install-recommends $pkgs")
        return [g[-1] if len(g) == 1 else f"{g[0]} or {g[1]}" for g in groups], script
    if pm in ("dnf", "yum"):
        caps = [_capability(lib) for lib in libs]
        # strict=0: install what is available even if one capability is not
        return caps, f"{pm} install -y --setopt=strict=0 " + " ".join(f"'{c}'" for c in caps)
    if pm == "zypper":
        caps = [_capability(lib) for lib in libs]
        return caps, "zypper --non-interactive install " + " ".join(f"'{c}'" for c in caps)
    if pm == "pacman":
        pkgs = list(dict.fromkeys(ALL[lib][2] for lib in libs))
        sync = 'if [ -z "$(ls -A /var/lib/pacman/sync 2>/dev/null)" ]; then pacman -Sy --noconfirm; fi; '
        return pkgs, sync + "pacman -S --needed --noconfirm " + " ".join(pkgs)
    return list(libs), None


def _elevate(script, gui):
    if os.geteuid() == 0:
        return ["sh", "-c", script]
    if gui and shutil.which("pkexec") and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return ["pkexec", "sh", "-c", script]
    if not gui and shutil.which("sudo"):
        return ["sudo", "sh", "-c", script]
    return None


def install(libs, progress, gui=True):
    """Installs the packages that provide `libs`, streaming the package
    manager's output to progress(str). Raises SetupError on failure."""
    packages, script = install_plan(libs)
    if script is None:
        raise SetupError("Install the packages that provide these libraries with your package manager, "
                         "then start the app again: " + ", ".join(libs))
    argv = _elevate(script, gui)
    if argv is None:
        raise SetupError("Run this in a terminal, then start the app again:\n"
                         f"sudo sh -c \"{script}\"")
    progress("Installing system packages: " + ", ".join(packages))
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    tail = []
    for line in proc.stdout:
        line = line.strip()
        if line:
            tail = (tail + [line])[-8:]
            progress(line)
    if proc.wait() != 0:
        raise SetupError("Installing the system packages failed:\n" + "\n".join(tail))
    still = [lib for lib in libs if not loadable(lib)]
    if still:
        hint = ""
        if "libxcb-cursor.so.0" in still and package_manager() in ("dnf", "yum"):
            hint = ("\nOn RHEL, AlmaLinux and Rocky Linux, libxcb-cursor comes from EPEL. "
                    "Enable it with: sudo dnf install epel-release")
        raise SetupError("These libraries are still missing: " + ", ".join(still) + hint)
