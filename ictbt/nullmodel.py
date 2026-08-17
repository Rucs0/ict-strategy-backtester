"""Null models: what would these statistics look like if the pattern meant nothing?

A statistic about a pattern is only evidence if it differs from what the same
statistic would be without the pattern. "74% of fair value gaps get filled"
sounds like a finding, but price at intraday timeframes oscillates constantly,
so a great many arbitrary price levels get revisited within a session. Without
a baseline the number is uninterpretable.

Two independent baselines are implemented here, because they fail in different
ways and agreeing answers are worth more than either alone.

**Sequence shuffle** (`shuffled_bars`). Rebuilds each session from its own
bars in a random order, preserving every bar's shape and every bar-to-bar
jump, and destroying only the order they arrived in. If the real sequence
carries information — if fair value gaps mark something about order flow —
then shuffled sessions should produce fewer of them, or ones that behave
differently. If the counts and fill rates match, the pattern is an artifact of
the return distribution rather than of sequence.

**Matched-position touch rate** (`matched_touch_rate`). Exploits a fact about
the fill definition: in touch mode a bullish gap is filled when some later bar
trades at or below candle 3's low. The gap's width never enters that
condition. So the fill rate is really measuring how often price revisits a
recent extreme before the session ends. This baseline asks exactly that of an
arbitrary bar — holding time-of-day and remaining-bars fixed, since a level
set at 09:45 has far more session left to be revisited than one set at 15:30.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar import rth_only, session_date
from .schema import OHLCV_COLUMNS, to_canonical


def shuffle_session(session: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Rebuild one session from its own bars in random order.

    Each bar is reduced to multiplicative geometry — the jump from the
    previous close to this open, and the high/low/close relative to this open
    — then the bars are reordered and re-chained. Preserved exactly: the
    multiset of bar shapes, the multiset of inter-bar jumps, and therefore the
    marginal return distribution. Destroyed: the order, and every serial
    dependence with it.
    """
    if len(session) < 2:
        return session.copy()

    o = session["open"].to_numpy(dtype="float64")
    h = session["high"].to_numpy(dtype="float64")
    low = session["low"].to_numpy(dtype="float64")
    c = session["close"].to_numpy(dtype="float64")

    # Relative geometry, all strictly positive for real price data.
    ho = h / o
    lo = low / o
    co = c / o
    jump = np.ones_like(o)
    jump[1:] = o[1:] / c[:-1]

    order = rng.permutation(len(session))

    new_o = np.empty_like(o)
    new_h = np.empty_like(o)
    new_l = np.empty_like(o)
    new_c = np.empty_like(o)

    price = o[0]
    for k, src in enumerate(order):
        open_k = price if k == 0 else price * jump[src]
        new_o[k] = open_k
        new_h[k] = open_k * ho[src]
        new_l[k] = open_k * lo[src]
        new_c[k] = open_k * co[src]
        price = new_c[k]

    out = pd.DataFrame(
        {
            "open": new_o,
            "high": new_h,
            "low": new_l,
            "close": new_c,
            "volume": session["volume"].to_numpy()[order],
            "trade_count": session["trade_count"].to_numpy()[order],
            "vwap": (new_h + new_l) / 2.0,
        },
        index=session.index,
    )[list(OHLCV_COLUMNS)]
    return out


def shuffled_bars(
    df: pd.DataFrame, rng: np.random.Generator, *, rth: bool = True
) -> pd.DataFrame:
    """Apply `shuffle_session` independently to every session in `df`.

    Shuffling is within-session on purpose. Mixing bars across days would
    destroy the session structure too, and then a difference in results would
    not say which of the two caused it.
    """
    frame = rth_only(df) if rth else df
    if frame.empty:
        return frame.copy()

    pieces = [
        shuffle_session(session, rng)
        for _, session in frame.groupby(session_date(frame.index).to_numpy())
    ]
    return to_canonical(pd.concat(pieces))


def session_positions(df: pd.DataFrame, timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Position of each timestamp within its own session, 0-based."""
    frame = df
    dates = session_date(frame.index).to_numpy()
    pos_within = np.concatenate(
        [np.arange(count) for count in pd.Series(dates).value_counts(sort=False).sort_index()]
    )
    lookup = pd.Series(pos_within, index=frame.index)
    return lookup.reindex(timestamps).to_numpy()


def matched_touch_rate(
    df: pd.DataFrame,
    gaps: pd.DataFrame,
    rng: np.random.Generator,
    *,
    rth: bool = True,
) -> float:
    """Baseline fill rate for arbitrary levels at matched times of day.

    For every real gap, takes its position within the session, moves to a
    randomly chosen *different* session, and asks the fill question of that
    session's bar at the same position: does any later bar in that session
    trade back through its low (bullish) or its high (bearish)?

    Holding the within-session position fixed is what makes this comparable.
    A level set on the third bar of the day has 23 chances to be revisited; a
    level set on the second-to-last bar has one.
    """
    frame = rth_only(df) if rth else df
    if frame.empty or gaps.empty:
        return float("nan")

    by_session = {
        date: session
        for date, session in frame.groupby(session_date(frame.index).to_numpy())
    }
    dates = list(by_session)
    positions = session_positions(frame, gaps.index)
    directions = gaps["direction"].to_numpy()

    hits = 0
    counted = 0
    for pos, direction in zip(positions, directions):
        if np.isnan(pos):
            continue
        pos = int(pos)
        # Draw a session long enough to contain both the level and a bar after it.
        for _ in range(20):
            candidate = by_session[dates[rng.integers(len(dates))]]
            if len(candidate) > pos + 1:
                break
        else:
            continue

        highs = candidate["high"].to_numpy()
        lows = candidate["low"].to_numpy()
        if direction == "bullish":
            hit = bool((lows[pos + 1 :] <= lows[pos]).any())
        else:
            hit = bool((highs[pos + 1 :] >= highs[pos]).any())
        hits += hit
        counted += 1

    return hits / counted if counted else float("nan")
