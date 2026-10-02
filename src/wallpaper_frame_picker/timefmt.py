# SPDX-License-Identifier: AGPL-3.0-or-later
"""Times as the app shows and accepts them."""


def fmt_time(t):
    """Seconds -> 'm:ss.sss', or 'h:mm:ss.sss' from an hour up."""
    t = max(0.0, t)
    m, s = divmod(t, 60)
    h, m = divmod(int(m), 60)
    return f"{h}:{m:02d}:{s:06.3f}" if h else f"{m}:{s:06.3f}"


def parse_time(text):
    """'83.5', '1:23.5' or '0:01:23.5' -> seconds, or None."""
    try:
        parts = [float(p) for p in str(text).strip().split(":")]
    except ValueError:
        return None
    if not 1 <= len(parts) <= 3 or any(p < 0 for p in parts):
        return None
    t = 0.0
    for p in parts:
        t = t * 60 + p
    return t
