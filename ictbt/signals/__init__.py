"""Signal primitives.

Each primitive is a pure function over a canonical bar frame (see
`ictbt.schema`). None of them read anything but the bars they are given, and
none of them look forward in time at detection.
"""

from .fvg import FVG_COLUMNS, find_fvgs, fvg_signal

__all__ = ["FVG_COLUMNS", "find_fvgs", "fvg_signal"]
