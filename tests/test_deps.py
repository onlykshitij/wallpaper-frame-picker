# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Linux system-library check and the package manager install plans."""
import sys

import pytest

from wallpaper_frame_picker import deps


def test_needed_follows_the_session(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    wayland = set(deps.needed())
    assert set(deps.WAYLAND) <= wayland and not set(deps.X11) & wayland
    monkeypatch.delenv("WAYLAND_DISPLAY")
    assert set(deps.X11) <= set(deps.needed())
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    assert deps.needed() == list(deps.CORE)


def test_nothing_is_needed_off_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert deps.needed() == [] and deps.missing() == []


@pytest.mark.parametrize("pm", ["apt-get", "dnf", "zypper", "pacman"])
def test_install_plans(monkeypatch, pm):
    monkeypatch.setattr(deps, "package_manager", lambda: pm)
    packages, script = deps.install_plan(["libxcb-cursor.so.0", "libglib-2.0.so.0", "libgthread-2.0.so.0"])
    if pm == "apt-get":
        # GLib's two libraries come from one package, which Ubuntu 24.04 renamed
        assert packages == ["libxcb-cursor0", "libglib2.0-0t64 or libglib2.0-0"]
        assert "for p in libglib2.0-0t64 libglib2.0-0;" in script and "apt-get install -y" in script
    elif pm in ("dnf", "zypper"):
        assert "'libxcb-cursor.so.0()(64bit)'" in script
    else:
        assert packages == ["xcb-util-cursor", "glib2"]
        assert script.endswith("pacman -S --needed --noconfirm xcb-util-cursor glib2")


def test_unknown_package_manager_explains_what_to_install(monkeypatch):
    monkeypatch.setattr(deps, "package_manager", lambda: None)
    assert deps.install_plan(["libxcb-cursor.so.0"]) == (["libxcb-cursor.so.0"], None)
    with pytest.raises(deps.SetupError, match="libxcb-cursor.so.0"):
        deps.install(["libxcb-cursor.so.0"], print)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux only")
def test_qt_core_libraries_are_found_here(monkeypatch):
    # every machine that runs the GUI tests has them, so this checks the detection itself
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    assert deps.missing() == []
