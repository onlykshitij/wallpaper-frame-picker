# SPDX-License-Identifier: AGPL-3.0-or-later
"""Finding or downloading uv, and plain file downloads.

Standard library only (plus platformdirs, through paths, and truststore),
because the standalone launcher uses this before any other dependency is
installed.
"""
import os
import platform
import re
import shutil
import ssl
import sys
import tarfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from .paths import data_dir

# uv release archives, by (operating system, machine)
UV_ASSETS = {
    ("linux", "x86_64"): "uv-x86_64-unknown-linux-gnu.tar.gz",
    ("linux", "aarch64"): "uv-aarch64-unknown-linux-gnu.tar.gz",
    ("darwin", "x86_64"): "uv-x86_64-apple-darwin.tar.gz",
    ("darwin", "arm64"): "uv-aarch64-apple-darwin.tar.gz",
    ("windows", "amd64"): "uv-x86_64-pc-windows-msvc.zip",
    ("windows", "arm64"): "uv-aarch64-pc-windows-msvc.zip",
}
UV_RELEASES = "https://github.com/astral-sh/uv/releases/latest/download/"


class SetupError(RuntimeError):
    """A dependency could not be found, downloaded or installed."""


def ssl_context():
    """Certificate checks done by the operating system, as in a browser.
    Python's own check only knows the root certificates already on the
    computer, but Windows adds most of them the first time a connection
    needs one, so a fresh Windows rejects github.com."""
    try:
        import truststore
    except ImportError:   # running from source without it
        return ssl.create_default_context()
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def download(url, dest, progress=None):
    """Downloads url to dest, via dest.part so a failed download leaves
    nothing behind. progress(done_bytes, total_bytes)."""
    dest = Path(dest)
    part = dest.with_name(dest.name + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "wallpaper-frame-picker"})
        with urllib.request.urlopen(req, timeout=60, context=ssl_context()) as r, open(part, "wb") as fh:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while chunk := r.read(1 << 16):
                fh.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
        part.replace(dest)
    except BaseException as e:
        part.unlink(missing_ok=True)
        reason = getattr(e, "reason", None)
        if isinstance(reason, ssl.SSLCertVerificationError):
            raise SetupError(
                f"This computer does not trust the certificate of {urllib.parse.urlsplit(url).hostname} "
                f"or a server it redirected to ({getattr(reason, 'verify_message', None) or reason}). "
                "If your network inspects secure connections, its certificate has to be installed "
                "on this computer.") from e
        raise
    return dest


# uv checks certificates against its own copy of the public root certificates.
# Networks that inspect secure connections sign them with a certificate only
# the operating system trusts, so after a certificate error the launcher
# switches uv to the system's check, and remembers that for later runs.
UV_CERT_ERROR = re.compile(r"certificate|UnknownIssuer", re.I)


def _system_certs_mark():
    return data_dir() / "uv-system-certs"


def uv_uses_system_certs():
    return _system_certs_mark().exists()


def use_system_certs_for_uv():
    _system_certs_mark().parent.mkdir(parents=True, exist_ok=True)
    _system_certs_mark().touch()


def uv_name():
    return "uv.exe" if sys.platform == "win32" else "uv"


def private_uv():
    return data_dir() / "bin" / uv_name()


def find_uv():
    """uv on PATH, or the private copy in the data folder, or None.
    FRAME_PICKER_OWN_UV=1 skips the one on PATH."""
    own = private_uv()
    on_path = None if os.environ.get("FRAME_PICKER_OWN_UV") == "1" else shutil.which("uv")
    return on_path or (str(own) if own.exists() else None)


def install_uv(progress=None):
    """Downloads uv from its GitHub releases into the data folder."""
    key = (platform.system().lower(), platform.machine().lower())
    asset = UV_ASSETS.get(key)
    if asset is None:
        raise SetupError(f"There is no uv download for {key[0]} on {key[1]}. "
                         "Install uv yourself: https://docs.astral.sh/uv/")
    url = os.environ.get("FRAME_PICKER_UV_URL", UV_RELEASES + asset)
    bin_dir = private_uv().parent
    bin_dir.mkdir(parents=True, exist_ok=True)
    archive = download(url, bin_dir / asset, progress)
    want = uv_name()
    try:
        if asset.endswith(".zip"):
            with zipfile.ZipFile(archive) as z:
                member = next(m for m in z.namelist() if Path(m).name == want)
                data = z.read(member)
        else:
            with tarfile.open(archive) as t:
                member = next(m for m in t.getmembers() if Path(m.name).name == want and m.isfile())
                data = t.extractfile(member).read()
    except StopIteration:
        raise SetupError(f"{asset} has no {want} in it")
    finally:
        archive.unlink(missing_ok=True)
    dest = bin_dir / want
    dest.write_bytes(data)
    dest.chmod(0o755)
    return str(dest)


def ensure_uv(progress_text=None):
    """uv's path, downloading it first if needed. progress_text(str)."""
    uv = find_uv()
    if uv:
        return uv
    say = progress_text or (lambda s: None)
    say("Downloading uv (about 20 MB)…")
    shown = [-1]

    def progress(done, total):
        if done >> 20 != shown[0]:   # once per MB
            shown[0] = done >> 20
            say(f"Downloading uv: {done >> 20} of {total >> 20} MB")
    return install_uv(progress)
