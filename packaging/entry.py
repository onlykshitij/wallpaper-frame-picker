# SPDX-License-Identifier: AGPL-3.0-or-later
"""Entry point of the standalone app: the launcher, which installs missing
dependencies on first launch and then starts the app or, with `cli`, the
command line."""
from wallpaper_frame_picker.launcher import main

main()
