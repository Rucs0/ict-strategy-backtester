"""Signal primitives.

Each primitive is a pure function over a canonical bar frame (see
`ictbt.schema`). None of them read anything but the bars they are given, and
none of them look forward in time at detection.
"""

from .fvg import FVG_COLUMNS, find_fvgs, fvg_signal
from .swings import DEFAULT_N, find_swings, swing_signal
from .sweeps import find_sweeps, sweep_signal
from .structure import find_structure_events, mss_signal
from .ote import find_ote_zones, ote_signal
from .killzone import (
    KILLZONES,
    Killzone,
    describe_killzones,
    filter_to_killzone,
    get_killzone,
    in_killzone,
    killzone_mask,
)

__all__ = [
    "DEFAULT_N",
    "FVG_COLUMNS",
    "KILLZONES",
    "Killzone",
    "find_ote_zones",
    "find_structure_events",
    "find_sweeps",
    "find_swings",
    "mss_signal",
    "ote_signal",
    "sweep_signal",
    "swing_signal",
    "describe_killzones",
    "filter_to_killzone",
    "find_fvgs",
    "fvg_signal",
    "get_killzone",
    "in_killzone",
    "killzone_mask",
]
