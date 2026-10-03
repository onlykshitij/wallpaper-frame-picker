# SPDX-License-Identifier: AGPL-3.0-or-later
"""Checks a standalone build on a generated test video: the version notice,
the first-launch dependency install, command-line analysis, exact-frame
export, and that the app starts.

    uv run python packaging/smoke_test.py APP [CLI]

APP is the launcher. CLI is the separate command-line program (Windows
only); without it, APP's `cli` mode is used. Dependencies install into a
temporary folder, so this never touches the real ones.
"""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from conftest import H, W, make_video, read_barcode   # noqa: E402


def main():
    app = [sys.argv[1]]
    cli = [sys.argv[2]] if len(sys.argv) > 2 else [sys.argv[1], "cli"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        env = {**os.environ, "FRAME_PICKER_CACHE": str(tmp / "cache"), "FRAME_PICKER_DATA": str(tmp / "data"),
               "XDG_CONFIG_HOME": str(tmp / "config")}
        env["FRAME_PICKER_OWN_UV"] = "1"   # download uv, as on most computers, though CI has it on PATH
        video = tmp / "smoke.webm"
        make_video(video)

        def run(argv, what):
            t = time.time()
            r = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=1800)
            if r.returncode:
                sys.exit(f"FAILED: {what}\n{r.stdout}\n{r.stderr}")
            print(f"ok  {what}  ({time.time() - t:.1f} s)")
            return r.stdout + r.stderr

        assert "GNU Affero General Public License" in run([*cli, "--version"], "version notice")
        # the Windows app has no console, so its separate CLI program installs there
        installer = cli[:1] if len(sys.argv) > 2 else app
        run([*installer, "--install-deps"], "first-launch install of the dependencies")
        assert "4 shots" in run([*cli, "analyze", str(video)], "analyze")
        run([*cli, "export", str(video), str(tmp / "out"), "--frames", "5", "230"], "export")
        for n in (5, 230):
            path = next((tmp / "out").glob(f"*_f{n:05d}.png"))
            img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
            assert img.shape == (H, W, 3) and read_barcode(img[..., ::-1]) == n, path
        print("ok  exported frames are full size and exact")

        env["QT_QPA_PLATFORM"] = "offscreen"
        # One-file builds run as a launcher plus the app, so stop the whole group.
        popen = {"start_new_session": True} if os.name == "posix" else \
                {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        proc = subprocess.Popen([*app, str(video)], env=env, **popen)
        time.sleep(20)
        alive = proc.poll() is None
        stop(proc)
        if not alive:
            sys.exit(f"FAILED: the app exited with code {proc.returncode}")
        print("ok  the app starts and keeps running")


def stop(proc):
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
        return
    import signal
    proc.terminate()   # the launcher passes SIGTERM on to the app
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


if __name__ == "__main__":
    main()
