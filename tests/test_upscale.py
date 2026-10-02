# SPDX-License-Identifier: AGPL-3.0-or-later
"""Upscaling tests. They need the upscale extra (uv run --extra upscale pytest)
and use a random-weight model with the Real-ESRGAN Compact layout, so they
check the plumbing, not picture quality."""
import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("spandrel")
safetensors_torch = pytest.importorskip("safetensors.torch")

from conftest import H, W, read_barcode   # noqa: E402
from wallpaper_frame_picker import cli, export, upscale_server   # noqa: E402
from wallpaper_frame_picker import upscaler as U   # noqa: E402
from wallpaper_frame_picker.video import VideoFile   # noqa: E402


@pytest.fixture(scope="module")
def model():
    from torch import nn
    torch.manual_seed(0)
    layers = [nn.Conv2d(3, 32, 3, 1, 1), nn.PReLU(32)]
    for _ in range(4):
        layers += [nn.Conv2d(32, 32, 3, 1, 1), nn.PReLU(32)]
    layers.append(nn.Conv2d(32, 3 * 4 * 4, 3, 1, 1))
    body = nn.ModuleList(layers)
    sd = {f"body.{n}": (t * 0.05 if t.dim() > 1 else t).contiguous() for n, t in body.state_dict().items()}
    path = U.models_dir() / "test-compact-x4.safetensors"
    safetensors_torch.save_file(sd, path)
    yield path
    path.unlink()


@pytest.fixture(scope="module")
def client():
    c = U.UpscaleClient()
    yield c
    c.close()


def test_model_loads(client, model):
    info = client.load(model)
    assert info["scale"] == 4 and info["arch"] == "RealESRGAN Compact"
    assert model in U.list_models()


def test_upscale_array(client, model):
    img = np.random.default_rng(0).integers(0, 255, (50, 70, 3), dtype=np.uint8)
    assert client.upscale_array(img, model).shape == (200, 280, 3)
    assert client.upscale_array(img, model, scale=1.0).shape == (50, 70, 3)


def test_tiles_leave_no_seams(model):
    up = upscale_server.Upscaler(model)
    img = np.random.default_rng(1).integers(0, 255, (300, 420, 3), dtype=np.uint8)
    up.tile = None
    whole = up.run(img)
    up.tile = 128
    assert np.array_equal(whole, up.run(img))


def test_errors_come_back_as_messages(client, tmp_path):
    bad = tmp_path / "broken.pth"
    bad.write_bytes(b"not a model")
    with pytest.raises(U.UpscalerError):
        client.load(bad)


def test_export_with_upscaling(client, model, video, tmp_path):
    v = VideoFile(video)
    jobs = [(v.pts_of_index(30), str(tmp_path / "a.png"))]
    up = {"client": client, "model": str(model), "scale": 2.0, "keep": True, "tag": export.upscale_tag(model, 2)}
    written = export.export_frames(video, jobs, up)
    assert [p.name for p in written] == ["a.png", "a_test-compact-x4_2x.png"]
    assert cv2.imread(str(written[1])).shape == (2 * H, 2 * W, 3)
    assert read_barcode(cv2.imread(str(written[0]))[..., ::-1]) == 30


def test_cli_upscale(model, video, tmp_path, capsys):
    cli.main(["export", str(video), str(tmp_path), "--frames", "40", "--upscale", model.stem, "--no-original"])
    files = [p.name for p in tmp_path.iterdir()]
    assert files == ["synthetic_S000_f00040_test-compact-x4_4x.png"]
    cli.main(["upscale", str(tmp_path / files[0]), "--model", str(model), "--scale", "0.5", "--out", str(tmp_path / "x")])
    out = next((tmp_path / "x").iterdir())
    assert cv2.imread(str(out)).shape == (2 * H, 2 * W, 3)
