"""The canonical bar frame, and the contract every other module relies on.

One representation, defined once. Everything downstream — signal primitives,
the backtest loop — assumes a frame that has passed `validate_bars`.

Two conventions are fixed here because getting either wrong is silent and
expensive:

**Timezone.** The index is tz-aware and localized to America/New_York, never
naive and never UTC. ICT concepts are defined in New York wall-clock time
(a killzone is "09:30-11:00 New York", not "13:30-15:00 UTC"), and the UTC
offset changes twice a year. Storing naive timestamps, or UTC timestamps that
get formatted for display later, is the standard way this class of bug gets in.

**Bar labelling.** A bar's timestamp is the START of its interval, matching
Alpaca. The bar labelled 09:30 covers 09:30:00-09:30:59.999. So the last
regular-hours minute bar of the day is labelled 15:59, and a bar labelled
16:00 is already after the close. Off-by-one here shifts every time-of-day
filter by one bar, which is small enough to look plausible and large enough
to change results.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd
from pandas.api import types as pdt

NY = ZoneInfo("America/New_York")

#: Bar timestamps mark the opening instant of the interval they cover.
BAR_LABEL = "left"

#: Required columns, in canonical order.
OHLCV_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
)

#: Columns that must never be null — a bar without a price is not a bar.
PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")


class SchemaError(ValueError):
    """A bar frame violated the contract in `validate_bars`."""


def validate_bars(df: pd.DataFrame, *, name: str = "bars") -> None:
    """Raise `SchemaError` unless `df` satisfies the canonical bar contract.

    Call this at every boundary where bars enter the system — after a fetch,
    after a cache load, after any reshaping. It is cheap relative to a
    backtest and it converts silent corruption into a loud failure.
    """
    if not isinstance(df, pd.DataFrame):
        raise SchemaError(f"{name}: expected DataFrame, got {type(df).__name__}")

    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise SchemaError(f"{name}: index must be a DatetimeIndex, got {type(idx).__name__}")
    if idx.tz is None:
        raise SchemaError(
            f"{name}: index is timezone-naive. Bars must carry an explicit "
            "timezone; see module docstring."
        )
    # Compare zone identity by current offset rules rather than object equality,
    # since ZoneInfo("America/New_York") and pytz's equivalent are both valid.
    if str(idx.tz) != "America/New_York":
        raise SchemaError(
            f"{name}: index timezone is {idx.tz!r}, expected 'America/New_York'. "
            "Convert with .tz_convert(NY) before validating."
        )
    if idx.name != "timestamp":
        raise SchemaError(f"{name}: index must be named 'timestamp', got {idx.name!r}")
    if not idx.is_monotonic_increasing:
        raise SchemaError(f"{name}: index must be sorted ascending")
    if idx.has_duplicates:
        dupes = idx[idx.duplicated()].unique()
        raise SchemaError(f"{name}: index has {len(dupes)} duplicate timestamps, e.g. {dupes[:3].tolist()}")

    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"{name}: missing required columns {missing}")

    for col in OHLCV_COLUMNS:
        if not pdt.is_numeric_dtype(df[col]):
            raise SchemaError(f"{name}: column {col!r} must be numeric, got {df[col].dtype}")

    if df.empty:
        return

    for col in PRICE_COLUMNS:
        if df[col].isna().any():
            n = int(df[col].isna().sum())
            raise SchemaError(f"{name}: column {col!r} has {n} null values")

    _validate_ohlc_consistency(df, name=name)


def _validate_ohlc_consistency(df: pd.DataFrame, *, name: str) -> None:
    """Check that each bar's four prices describe a possible bar.

    Catches vendor bad ticks and, more often, our own reshaping bugs — a
    misaligned join produces frames where `high` is below `open` almost
    immediately.
    """
    body_max = df[["open", "close"]].max(axis=1)
    body_min = df[["open", "close"]].min(axis=1)

    checks = {
        "high < low": df["high"] < df["low"],
        "high < max(open, close)": df["high"] < body_max,
        "low > min(open, close)": df["low"] > body_min,
        "negative volume": df["volume"] < 0,
    }
    for label, violated in checks.items():
        if violated.any():
            n = int(violated.sum())
            first = df.index[violated][0]
            raise SchemaError(
                f"{name}: {n} bars violate '{label}', first at {first}"
            )


def to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a raw bar frame into canonical form.

    Handles the mechanical part — timezone conversion, index naming, column
    ordering, sorting, de-duplication. Does not invent data: anything that
    cannot be fixed by relabelling is left for `validate_bars` to reject.
    """
    out = df.copy()

    if not isinstance(out.index, pd.DatetimeIndex):
        raise SchemaError(f"cannot canonicalize: index is {type(out.index).__name__}")

    if out.index.tz is None:
        raise SchemaError(
            "cannot canonicalize a timezone-naive index: the correct source "
            "timezone is not inferable and guessing it would corrupt every "
            "time-of-day filter downstream. Localize at the source instead."
        )
    out.index = out.index.tz_convert(NY)
    out.index.name = "timestamp"

    # Keep the last observation on duplicate timestamps: when a vendor restates
    # a bar, the later record is the correction.
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()

    for col in OHLCV_COLUMNS:
        if col not in out.columns:
            out[col] = float("nan")

    return out[list(OHLCV_COLUMNS)]


def empty_bars() -> pd.DataFrame:
    """An empty frame that satisfies the contract — useful as a base case."""
    idx = pd.DatetimeIndex([], tz=NY, name="timestamp")
    return pd.DataFrame(
        {c: pd.Series(dtype="float64") for c in OHLCV_COLUMNS},
        index=idx,
    )
