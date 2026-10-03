# SPDX-License-Identifier: AGPL-3.0-or-later
"""The standalone app's Python environment.

The downloadable app is a small launcher. On first launch it uses uv to
install Python, Qt, PyAV, OpenCV and NumPy into the user data folder, then
runs the app from there. The launcher carries the app's own wheel and the
exact package versions CI tested (bundle/requirements.txt), so every
install gets the same versions.

Standard library only, like the rest of the launcher.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__
from .paths import data_dir
from .uvtools import (UV_CERT_ERROR, SetupError, ensure_uv, find_uv, use_system_certs_for_uv,
                      uv_uses_system_certs)

PYTHON = "3.12"
# what each downloaded package is, for the install dialog
PACKAGES = {
    "pyside6-essentials": ("Qt 6 (PySide6)", "the user interface"),
    "shiboken6": ("Shiboken", "connects Qt to Python"),
    "av": ("PyAV with FFmpeg", "video decoding"),
    "opencv-python-headless": ("OpenCV", "sharpness and motion analysis"),
    "numpy": ("NumPy", "number crunching"),
    "platformdirs": ("platformdirs", "finds the settings folders"),
    "truststore": ("truststore", "checks certificates the way the system does"),
}
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def bundle_dir():
    """The wheel, requirements and sizes the launcher carries, or None when
    not running as the standalone launcher."""
    base = getattr(sys, "_MEIPASS", None)
    d = Path(base) / "bundle" if base else None
    return d if d and (d / "requirements.txt").exists() else None


def _runtime_dir():
    req = (bundle_dir() / "requirements.txt").read_bytes()
    return data_dir() / "runtime" / f"{__version__}-{hashlib.sha1(req).hexdigest()[:8]}"


def _python(venv, gui=False):
    if os.name == "nt":
        return venv / "Scripts" / ("pythonw.exe" if gui else "python.exe")
    return venv / "bin" / "python"


def is_ready():
    rt = _runtime_dir()
    return (rt / "ready").exists() and _python(rt / "venv").exists()


def download_list():
    """[(name, purpose, size in MB or None)] of what install() will fetch."""
    sizes = {}
    f = bundle_dir() / "sizes.json"
    if f.exists():
        sizes = json.loads(f.read_text())
    rows = []
    if find_uv() is None:
        rows.append(("uv", "installs everything else", 20))
    if not _managed_python_present():
        rows.append((f"Python {PYTHON}", "runs the app", 30))
    for pkg, (name, purpose) in PACKAGES.items():
        mb = sizes.get(pkg)
        rows.append((name, purpose, round(mb / 2 ** 20) if mb else None))
    return rows


def _managed_python_present():
    uv = find_uv()
    if not uv:
        return False
    r = subprocess.run([uv, "python", "find", "--python-preference", "only-managed", PYTHON],
                       capture_output=True, text=True, env=clean_env())
    return r.returncode == 0


def clean_env():
    """The environment without what PyInstaller sets for the launcher
    itself, so child processes load their own libraries."""
    env = dict(os.environ)
    frozen = getattr(sys, "frozen", False)
    for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        orig = env.pop(var + "_ORIG", None)
        if orig is not None:
            env[var] = orig
        elif frozen:
            env.pop(var, None)
    if frozen:
        for var in ("TCL_LIBRARY", "TK_LIBRARY", "PYTHONHOME", "PYTHONPATH"):
            env.pop(var, None)
    env["NO_COLOR"] = "1"
    if uv_uses_system_certs():
        env.setdefault("UV_SYSTEM_CERTS", "1")   # also reaches the app, for the upscaler's install
    return env


def _run(argv, progress, what):
    tail = _run_once(argv, progress)
    if tail and UV_CERT_ERROR.search("\n".join(tail)) and not uv_uses_system_certs():
        progress("uv does not trust the server's certificate. Trying again with the system's certificate check…")
        use_system_certs_for_uv()
        tail = _run_once(argv, progress)
    if tail:
        raise SetupError(f"{what} failed:\n" + "\n".join(tail))


def _run_once(argv, progress):
    """Runs argv, passing its output lines to progress. Returns None on
    success, or the last lines of output on failure."""
    proc = subprocess.Popen([str(a) for a in argv], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace", env=clean_env())
    tail = []
    for line in proc.stdout:
        line = ANSI.sub("", line).strip()
        if line:
            tail = (tail + [line])[-8:]
            progress(line)
    if proc.wait() == 0:
        return None
    return tail or ["(no output)"]


def install(progress):
    """Installs the environment, reporting progress(str) as it goes."""
    bundle = bundle_dir()
    rt = _runtime_dir()
    venv = rt / "venv"
    uv = ensure_uv(progress)
    if rt.exists():
        shutil.rmtree(rt)   # a half-finished earlier attempt
    rt.mkdir(parents=True)
    progress(f"Setting up Python {PYTHON}…")
    _run([uv, "venv", "--python", PYTHON, "--python-preference", "only-managed", venv], progress,
         f"Setting up Python {PYTHON}")
    wheel = next(bundle.glob("*.whl"))
    progress("Downloading Qt, PyAV, OpenCV and NumPy…")
    try:
        _run([uv, "pip", "install", "--python", _python(venv), "-r", bundle / "requirements.txt", wheel], progress,
             "Installing the Python packages")
    except SetupError as e:
        if "matching platform tag" not in str(e):
            raise
        # an older system the tested versions have no builds for
        progress("The tested versions have no builds for this system. Installing the newest versions that do…")
        _run([uv, "pip", "install", "--python", _python(venv), wheel], progress, "Installing the Python packages")
    progress("Checking the installation…")
    _run([_python(venv), "-c", "import PySide6.QtWidgets, av, cv2, numpy, wallpaper_frame_picker"], progress,
         "Checking the installation")
    (rt / "ready").write_text(__version__)
    for old in rt.parent.iterdir():   # environments of earlier versions
        if old != rt and old.is_dir():
            shutil.rmtree(old, ignore_errors=True)
    progress("Done.")


def launch(args, gui):
    """Runs the app from the environment with these arguments."""
    py = _python(_runtime_dir() / "venv", gui=gui)
    argv = [str(py), "-m", "wallpaper_frame_picker", *args]
    if os.name == "posix":
        os.execve(argv[0], argv, clean_env())   # replaces the launcher process
    sys.exit(subprocess.call(argv, env=clean_env()))
