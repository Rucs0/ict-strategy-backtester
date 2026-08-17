"""Killzone filters: time-of-day windows in New York wall-clock time.

**These boundaries are quoted, not derived.** Every window below is taken from
how ICT material commonly states it, and the sources do not fully agree —
the New York AM killzone appears as 07:00-10:00, 08:30-11:00 and 09:30-11:00
depending on who is writing. None of them come with a justification that
survives asking "why not fifteen minutes earlier?"

That makes every window a free parameter, and free parameters are how a
backtest gets tuned into looking profitable. Two consequences, both enforced
by the design here rather than by discipline:

1. Windows are data, not constants baked into logic. `KILLZONES` is a plain
   registry and `Killzone` instances can be constructed freely, so a
   sensitivity sweep is cheap to write.
2. Phase 3 must sweep them. If an edge appears at 09:30-11:00 and vanishes at
   09:45-11:00, that is noise wearing a schedule.

**Session overlap.** These windows come from a 24-hour futures and forex
context. US equities trade 09:30-16:00, so the Asian and London windows fall
almost entirely outside the hours this project can test at all. They are
included for completeness and flagged by `overlaps_rth`; a killzone with no
regular-hours overlap will silently select nothing on equity data, which is
worth knowing before wondering why a strategy produced no trades.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from ..calendar import RTH_CLOSE, RTH_OPEN, _require_ny


@dataclass(frozen=True)
class Killzone:
    """A time-of-day window, New York wall clock.

    The interval is half-open, ``[start, end)``, to match left-labelled bars:
    a window ending at 11:00 includes the 10:45 bar and excludes the 11:00
    bar, which opens as the window closes.
    """

    name: str
    start: dt.time
    end: dt.time

    @property
    def wraps_midnight(self) -> bool:
        return self.start >= self.end

    @property
    def overlaps_rth(self) -> bool:
        """Does any part of this window fall inside US equity regular hours?"""
        if self.wraps_midnight:
            return self.start < RTH_CLOSE or self.end > RTH_OPEN
        return self.start < RTH_CLOSE and self.end > RTH_OPEN

    def duration_minutes(self) -> int:
        start = self.start.hour * 60 + self.start.minute
        end = self.end.hour * 60 + self.end.minute
        return (end - start) % (24 * 60) or 24 * 60


def _t(hhmm: str) -> dt.time:
    hour, minute = hhmm.split(":")
    return dt.time(int(hour), int(minute))


#: Commonly quoted killzone windows. Values are conventional, not derived —
#: see the module docstring. Treat this as a starting point for a sweep.
KILLZONES: dict[str, Killzone] = {
    "asian": Killzone("asian", _t("20:00"), _t("00:00")),
    "london": Killzone("london", _t("02:00"), _t("05:00")),
    "ny_am": Killzone("ny_am", _t("07:00"), _t("10:00")),
    "london_close": Killzone("london_close", _t("10:00"), _t("12:00")),
    "silver_bullet_am": Killzone("silver_bullet_am", _t("10:00"), _t("11:00")),
    "ny_pm": Killzone("ny_pm", _t("13:30"), _t("16:00")),
}


def get_killzone(name: str) -> Killzone:
    """Look up a named killzone, with a useful error when it is missing."""
    try:
        return KILLZONES[name]
    except KeyError:
        raise KeyError(
            f"unknown killzone {name!r}; known: {sorted(KILLZONES)}"
        ) from None


def in_killzone(
    index: pd.DatetimeIndex, killzone: Killzone | str
) -> pd.Series:
    """Boolean mask: does each bar open inside the window?

    Because the index carries New York wall-clock time, this is automatically
    correct across daylight-saving transitions — 10:00 New York is 10:00 New
    York in both regimes, even though the UTC offset differs.
    """
    _require_ny(index)
    kz = get_killzone(killzone) if isinstance(killzone, str) else killzone

    times = index.time
    if kz.wraps_midnight:
        mask = (times >= kz.start) | (times < kz.end)
    else:
        mask = (times >= kz.start) & (times < kz.end)
    return pd.Series(mask, index=index, name=f"in_{kz.name}")


def killzone_mask(
    index: pd.DatetimeIndex, names: list[str] | list[Killzone]
) -> pd.Series:
    """Union of several killzones — true if the bar is in any of them."""
    if not names:
        return pd.Series(False, index=index, name="in_killzone")
    mask = in_killzone(index, names[0])
    for other in names[1:]:
        mask = mask | in_killzone(index, other)
    return mask.rename("in_killzone")


def filter_to_killzone(
    df: pd.DataFrame, killzone: Killzone | str
) -> pd.DataFrame:
    """Keep only bars inside the window."""
    return df.loc[in_killzone(df.index, killzone).to_numpy()]


def describe_killzones() -> pd.DataFrame:
    """The registry as a table, including which windows equities can test."""
    return pd.DataFrame(
        [
            {
                "name": kz.name,
                "start": kz.start.strftime("%H:%M"),
                "end": kz.end.strftime("%H:%M"),
                "minutes": kz.duration_minutes(),
                "overlaps_rth": kz.overlaps_rth,
            }
            for kz in KILLZONES.values()
        ]
    ).set_index("name")
