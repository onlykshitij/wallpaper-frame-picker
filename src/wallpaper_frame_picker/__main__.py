# SPDX-License-Identifier: AGPL-3.0-or-later
"""Starts the app, or the command line when the first argument is `cli`:

    python -m wallpaper_frame_picker [VIDEO]
    python -m wallpaper_frame_picker cli shots VIDEO

On Linux, the app first checks that the system libraries Qt needs are
there, and offers to install the missing ones.
"""
import os
import sys


def system_libraries_ok():
    """True when Qt can load; otherwise shows the install pop-up."""
    if os.environ.get("FRAME_PICKER_SKIP_DEPS"):
        return True
    from . import deps
    libs = deps.missing()
    if not libs:
        return True
    from . import depdialog
    return depdialog.ask_and_install([], (deps.install_plan(libs)[0], None),
                                     lambda progress: deps.install(libs, progress, gui=True))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        from .cli import main as cli_main
        cli_main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] in ("--version", "-V"):
        from . import NOTICE
        print(NOTICE)
    else:
        if not system_libraries_ok():
            sys.exit(1)
        from .app import run
        run()


if __name__ == "__main__":
    main()
