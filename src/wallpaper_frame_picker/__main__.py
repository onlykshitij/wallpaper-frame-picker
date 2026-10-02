# SPDX-License-Identifier: AGPL-3.0-or-later
"""Starts the app, or the command line when the first argument is `cli`:

    python -m wallpaper_frame_picker [VIDEO]
    python -m wallpaper_frame_picker cli shots VIDEO

The standalone builds use this as their entry point too.
"""
import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        from .cli import main as cli_main
        cli_main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] in ("--version", "-V"):
        from . import NOTICE
        print(NOTICE)
    else:
        from .app import run
        run()


if __name__ == "__main__":
    main()
