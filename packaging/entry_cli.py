# SPDX-License-Identifier: AGPL-3.0-or-later
"""Entry point of the Windows command-line build. Windows GUI programs have
no console, so the CLI ships there as its own console program."""
from wallpaper_frame_picker.cli import main

main()
