# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where Wallpaper Frame Picker keeps analyses and upscaling models.

Analyses go to the user cache folder (~/.cache/wallpaper-frame-picker on Linux) and
models to the user data folder (~/.local/share/wallpaper-frame-picker/models).
FRAME_PICKER_CACHE and FRAME_PICKER_MODELS override them.
"""
import os
from pathlib import Path

import platformdirs

APP = "wallpaper-frame-picker"


def cache_dir():
    env = os.environ.get("FRAME_PICKER_CACHE")
    return Path(env) if env else Path(platformdirs.user_cache_dir(APP))


def data_dir():
    """User data folder: models, plus the private copy of uv that the
    standalone builds download for AI upscaling."""
    env = os.environ.get("FRAME_PICKER_DATA")
    d = Path(env) if env else Path(platformdirs.user_data_dir(APP))
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir():
    env = os.environ.get("FRAME_PICKER_MODELS")
    d = Path(env) if env else data_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d
