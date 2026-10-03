# SPDX-License-Identifier: AGPL-3.0-or-later
"""Checks that a one-file Linux build runs on any glibc distribution from
the target glibc version on: no bundled library may need a newer glibc or
C++ runtime, and none of the libraries the spec leaves to the user's system
may be bundled.

    uv run --with pyinstaller python packaging/check_bundle.py dist/WallpaperFramePicker-linux-x86_64

Needs objdump (binutils).
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PyInstaller.archive.readers import CArchiveReader

MAX_GLIBC = (2, 28)
MAX_GLIBCXX = (3, 4, 25)   # the C++ runtime of glibc 2.28 systems (GCC 8)
HOST_LIBS = re.compile(r"^(libc|libm|libmvec|libstdc\+\+|libgcc_s|libfontconfig|libfreetype|libX11|"
                       r"libglib-2\.0|libgtk-3|libGL|libEGL|libdrm|libwayland-client)\.so")


def versions(path, prefix):
    out = subprocess.run(["objdump", "-T", str(path)], capture_output=True, text=True).stdout
    return {tuple(int(x) for x in v.split(".")) for v in re.findall(rf"\b{prefix}_([0-9.]+[0-9])\b", out)}


def main():
    archive = CArchiveReader(sys.argv[1])
    problems = []
    newest_glibc, newest_cxx = (0,), (0,)
    with tempfile.TemporaryDirectory() as tmp:
        names = [n for n in archive.toc if re.search(r"\.so(\.|$)", n)]
        for name in names:
            path = Path(tmp) / name.replace("/", "_")
            path.write_bytes(archive.extract(name))
            base = Path(name).name
            if HOST_LIBS.match(base):
                problems.append(f"{name} should come from the user's system, not the bundle")
            g, c = versions(path, "GLIBC"), versions(path, "GLIBCXX")
            if g and max(g) > MAX_GLIBC:
                problems.append(f"{name} needs glibc {'.'.join(map(str, max(g)))}")
            if c and max(c) > MAX_GLIBCXX:
                problems.append(f"{name} needs GLIBCXX {'.'.join(map(str, max(c)))}")
            newest_glibc, newest_cxx = max([newest_glibc, *g]), max([newest_cxx, *c])
    print(f"{len(names)} shared libraries; newest glibc needed {'.'.join(map(str, newest_glibc))}, "
          f"newest GLIBCXX {'.'.join(map(str, newest_cxx))}")
    if problems:
        print("\n".join(problems))
        sys.exit(f"{len(problems)} problems: this build would not run on every distribution")
    print("ok: runs on glibc %d.%d and newer" % MAX_GLIBC)


if __name__ == "__main__":
    main()
