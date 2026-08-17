"""Swing point (fractal) detection.

A swing high is a bar whose high exceeds the highs of the `n` bars on either
side of it. A swing low is the mirror image. This is the standard fractal
definition and it is mechanically clean — except for `n`, which is the first
genuinely free parameter in this project.

**`n` is chosen, not derived.** No ICT source specifies it. Values of 2, 3 and
5 all appear in circulation with no stated justification. The default here is
2 because it is the smallest value that requires a bar to beat more than its
immediate neighbours, and for no better reason than that. Everything built on
swing points — liquidity sweeps, market structure shifts — inherits this
parameter, so a result that holds at n=2 and collapses at n=3 is noise. Sweep
it before believing anything.

**Confirmation lag is the correctness trap.** A swing high at bar `i` cannot
be known until bar `i + n` has closed, because until then the bars that would
disqualify it have not printed. Marking the swing at bar `i` and letting a
backtest act there is lookahead bias, and it is especially seductive here
because a chart drawn after the fact shows the swing sitting at `i`. Every row
this module emits carries `confirmed_at`, and anything acting on swings must
use that column rather than the index.

**Ties are not swings.** Comparison is strict, so a bar that merely equals its
neighbour's high is not a swing high. Equal highs are common in real data —
they are exactly the resting-liquidity clusters that sweeps are supposed to
target — and calling both bars swings would double-count them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..calendar import rth_only, session_date
from ..schema import validate_bars

#: Default bars either side. Chosen, not derived — see the module docstring.
DEFAULT_N = 2

SWING_COLUMNS = ("kind", "price", "confirmed_at", "bars_to_confirm")

_SCOPES = ("session", "continuous")


def find_swings(
    df: pd.DataFrame,
    *,
    n: int = DEFAULT_N,
    scope: str = "session",
    rth: bool = True,
    validate: bool = True,
) -> pd.DataFrame:
    """Locate swing highs and lows.

    Returns one row per swing, indexed by the timestamp of the swing bar
    itself, with `confirmed_at` giving the first bar at which the swing was
    knowable.

    Args:
        df: canonical bar frame.
        n: bars required on each side. Free parameter — sweep it.
        scope: "session" restarts detection each trading day; "continuous"
            lets a swing be confirmed against bars from the following session.
            Session scoping is the default for consistency with the rest of
            the package, and it is a choice: ICT structure is often read
            across days, so a prior-day high is a meaningful level. Continuous
            scoping admits those, at the cost of letting the overnight gap
            participate in the comparison.
        rth: restrict to regular trading hours first.
        validate: check the input against the bar contract.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if scope not in _SCOPES:
        raise ValueError(f"scope must be one of {_SCOPES}, got {scope!r}")
    if validate:
        validate_bars(df, name="swing input")

    frame = rth_only(df) if rth else df
    if len(frame) < 2 * n + 1:
        return _empty_swings()

    if scope == "continuous":
        pieces = [_scan(frame, n)]
    else:
        pieces = [
            _scan(session, n)
            for _, session in frame.groupby(session_date(frame.index).to_numpy())
        ]

    pieces = [p for p in pieces if len(p)]
    if not pieces:
        return _empty_swings()

    out = pd.concat(pieces).sort_index()
    out.index.name = "swing_at"
    return out


def _scan(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    """Find fractals in one contiguous stretch of bars."""
    if len(frame) < 2 * n + 1:
        return _empty_swings()

    high = frame["high"].to_numpy(dtype="float64")
    low = frame["low"].to_numpy(dtype="float64")
    index = frame.index
    size = len(frame)

    is_high = np.ones(size, dtype=bool)
    is_low = np.ones(size, dtype=bool)
    # Strict comparison against each of the n neighbours on both sides.
    for k in range(1, n + 1):
        left_h = np.full(size, -np.inf)
        right_h = np.full(size, -np.inf)
        left_h[k:] = high[:-k]
        right_h[:-k] = high[k:]
        is_high &= (high > left_h) & (high > right_h)

        left_l = np.full(size, np.inf)
        right_l = np.full(size, np.inf)
        left_l[k:] = low[:-k]
        right_l[:-k] = low[k:]
        is_low &= (low < left_l) & (low < right_l)

    # Bars without n full neighbours on both sides cannot be evaluated.
    edge = np.zeros(size, dtype=bool)
    edge[:n] = True
    edge[size - n :] = True
    is_high &= ~edge
    is_low &= ~edge

    positions = np.flatnonzero(is_high | is_low)
    if not len(positions):
        return _empty_swings()

    kinds = np.where(is_high[positions], "high", "low")
    prices = np.where(is_high[positions], high[positions], low[positions])

    # Confirmation: the swing is knowable once bar i + n has closed.
    confirm_pos = positions + n
    confirmed_at = pd.DatetimeIndex(
        [index[p] if p < size else pd.NaT for p in confirm_pos], tz=index.tz
    )

    out = pd.DataFrame(
        {
            "kind": kinds,
            "price": prices,
            "confirmed_at": confirmed_at,
            "bars_to_confirm": np.full(len(positions), float(n)),
        },
        index=index[positions],
    )
    out.index.name = "swing_at"
    return out


def swing_signal(
    df: pd.DataFrame,
    *,
    n: int = DEFAULT_N,
    kind: str | None = None,
    scope: str = "session",
    rth: bool = True,
) -> pd.Series:
    """Boolean series aligned to `df`: was a swing confirmed on this bar?

    Marks `confirmed_at`, never the swing bar itself. A strategy reading the
    swing bar would be trading on a shape that had not finished forming.
    """
    # Argument validation precedes any work: a bad argument is a bug whether
    # or not the data happens to contain a swing.
    if kind is not None and kind not in ("high", "low"):
        raise ValueError(f"kind must be 'high' or 'low', got {kind!r}")

    swings = find_swings(df, n=n, scope=scope, rth=rth)
    signal = pd.Series(False, index=df.index, name="swing_signal")
    if swings.empty:
        return signal

    if kind is not None:
        swings = swings[swings["kind"] == kind]

    marks = swings["confirmed_at"].dropna()
    signal.loc[signal.index.isin(marks)] = True
    return signal


def _empty_swings() -> pd.DataFrame:
    from ..schema import NY

    return pd.DataFrame(
        {
            "kind": pd.Series(dtype="object"),
            "price": pd.Series(dtype="float64"),
            "confirmed_at": pd.Series(dtype=f"datetime64[ns, {NY}]"),
            "bars_to_confirm": pd.Series(dtype="float64"),
        },
        index=pd.DatetimeIndex([], tz=NY, name="swing_at"),
    )
