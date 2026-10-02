# SPDX-License-Identifier: AGPL-3.0-or-later
"""Checks a standalone build on a generated test video: the version notice,
command-line analysis, exact-frame export, and that the GUI starts.

    uv run python packaging/smoke_test.py GUI_BINARY [CLI_BINARY]

Without CLI_BINARY the GUI binary's `cli` mode is used (Linux and macOS).
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
    gui = [sys.argv[1]]
    cli = [sys.argv[2]] if len(sys.argv) > 2 else [sys.argv[1], "cli"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        env = {**os.environ, "FRAME_PICKER_CACHE": str(tmp / "cache"), "FRAME_PICKER_DATA": str(tmp / "data"),
               "XDG_CONFIG_HOME": str(tmp / "config"), "QT_QPA_PLATFORM": "offscreen"}
        video = tmp / "smoke.webm"
        make_video(video)

        def run(*args):
            t = time.time()
            r = subprocess.run([*cli, *args], env=env, capture_output=True, text=True, timeout=600)
            if r.returncode:
                sys.exit(f"FAILED: {' '.join(args)}\n{r.stdout}\n{r.stderr}")
            print(f"ok  {' '.join(args[:2])}  ({time.time() - t:.1f} s)")
            return r.stdout

        assert "GNU Affero General Public License" in run("--version")
        assert "4 shots" in run("analyze", str(video))
        run("export", str(video), str(tmp / "out"), "--frames", "5", "230")
        for n in (5, 230):
            path = next((tmp / "out").glob(f"*_f{n:05d}.png"))
            img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
            assert img.shape == (H, W, 3) and read_barcode(img[..., ::-1]) == n, path
        print("ok  exported frames are full size and exact")

        # One-file builds run as a launcher plus the app, so stop the whole group.
        popen = {"start_new_session": True} if os.name == "posix" else \
                {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        proc = subprocess.Popen([*gui, str(video)], env=env, **popen)
        time.sleep(20)
        alive = proc.poll() is None
        stop(proc)
        if not alive:
            sys.exit(f"FAILED: the GUI exited with code {proc.returncode}")
        print("ok  GUI starts and keeps running")


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
