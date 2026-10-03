# SPDX-License-Identifier: AGPL-3.0-or-later
"""Entry point of the standalone app.

Checks for missing dependencies, shows the "Install dependencies" pop-up
when something is missing, then runs the app (or its command line) from
the environment that runtime.py manages.

    WallpaperFramePicker [VIDEO]          the app
    WallpaperFramePicker cli ...          the command line
    WallpaperFramePicker --install-deps   install everything without asking, then exit

Standard library only: nothing else is installed yet when this runs.
"""
import sys

from . import NOTICE, depdialog, deps, runtime


def main(cli=False):
    args = sys.argv[1:]
    if args[:1] == ["cli"]:
        cli, args = True, args[1:]
    if args[:1] in (["--version"], ["-V"]):
        print(NOTICE)
        return
    install_only = "--install-deps" in args
    assume_yes = install_only or "--yes" in args
    args = [a for a in args if a not in ("--install-deps", "--yes")]
    gui = not (cli or install_only)

    libs = [] if cli else deps.missing()   # the command line never loads Qt
    need_env = not runtime.is_ready()
    if libs or need_env:
        rows = runtime.download_list() if need_env else []
        system = (deps.install_plan(libs)[0], None) if libs else None

        def install(progress):
            if libs:
                deps.install(libs, progress, gui=gui)
            if need_env:
                runtime.install(progress)

        if not depdialog.ask_and_install(rows, system, install, gui=gui, assume_yes=assume_yes):
            sys.exit(1)
    if install_only:
        print("All dependencies are installed.")
        return
    runtime.launch((["cli"] if cli else []) + args, gui=gui)
