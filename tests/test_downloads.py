# SPDX-License-Identifier: AGPL-3.0-or-later
"""Downloads over HTTPS, and switching uv to the system's certificate check."""
import functools
import http.server
import shutil
import ssl
import subprocess
import sys
import threading

import pytest

from wallpaper_frame_picker import runtime, uvtools


@pytest.fixture
def https_server(tmp_path):
    """A local HTTPS server with a self-signed certificate, serving
    tmp_path/www. Yields (base url, certificate file)."""
    if not shutil.which("openssl"):
        pytest.skip("needs the openssl command")
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    made = subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                           "-keyout", str(key), "-out", str(cert), "-subj", "/CN=localhost",
                           "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1"], capture_output=True)
    if made.returncode:
        pytest.skip("this openssl cannot make the test certificate")
    www = tmp_path / "www"
    www.mkdir()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(www))
    handler.log_message = lambda *a: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    server.socket = ctx.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"https://localhost:{server.server_address[1]}/", cert, www
    server.shutdown()
    server.server_close()


def test_downloads_use_the_system_certificate_check():
    import truststore
    assert isinstance(uvtools.ssl_context(), truststore.SSLContext)


def test_untrusted_certificate_is_explained(https_server, tmp_path):
    url, _, www = https_server
    (www / "file.bin").write_bytes(b"x" * 1000)
    dest = tmp_path / "file.bin"
    with pytest.raises(uvtools.SetupError, match="does not trust the certificate of localhost"):
        uvtools.download(url + "file.bin", dest)
    assert list(tmp_path.glob("file.bin*")) == []


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="trusting a test certificate needs SSL_CERT_FILE")
def test_trusted_certificate_downloads(https_server, tmp_path, monkeypatch):
    url, cert, www = https_server
    data = bytes(range(256)) * 1000
    (www / "file.bin").write_bytes(data)
    monkeypatch.setenv("SSL_CERT_FILE", str(cert))
    seen = []
    dest = uvtools.download(url + "file.bin", tmp_path / "file.bin", lambda d, t: seen.append((d, t)))
    assert dest.read_bytes() == data and seen[-1] == (len(data), len(data))


FAKE_UV = ("import os, sys\n"
           "if os.environ.get('UV_SYSTEM_CERTS') != '1':\n"
           "    print('error: invalid peer certificate: UnknownIssuer')\n"
           "    sys.exit(2)\n")


def test_uv_retries_with_the_system_certificates(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAME_PICKER_DATA", str(tmp_path))
    monkeypatch.delenv("UV_SYSTEM_CERTS", raising=False)
    lines = []
    runtime._run([sys.executable, "-c", FAKE_UV], lines.append, "Installing")
    assert any("system's certificate check" in line for line in lines)
    assert uvtools.uv_uses_system_certs() and runtime.clean_env()["UV_SYSTEM_CERTS"] == "1"


def test_other_uv_failures_are_not_retried(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAME_PICKER_DATA", str(tmp_path))
    lines = []
    with pytest.raises(uvtools.SetupError, match="Installing failed:\nno space left"):
        runtime._run([sys.executable, "-c", "print('no space left'); raise SystemExit(1)"], lines.append, "Installing")
    assert lines == ["no space left"] and not uvtools.uv_uses_system_certs()
