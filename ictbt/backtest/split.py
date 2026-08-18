"""The in-sample / out-of-sample split.

**This split was fixed before a single backtest was run.** That ordering is
the entire point. A holdout chosen after seeing results is not a holdout — if
the boundary moves once the numbers are known, whatever it was protecting
against has already happened.

Split date: 2023-01-01. Everything before is in-sample and may be used freely
for development, parameter sweeps, and as many looks as necessary. Everything
from that date on is out-of-sample and is intended to be evaluated **once**,
after the strategy and its parameters are final.

Chosen for the split ratio it produces on the available history (2016 to
2026): roughly seven years in, three and a half years out, about 66/34. The
alternative of a random or interleaved split is wrong for time series —
adjacent bars are correlated, so shuffling leaks the answer across the
boundary.

The out-of-sample period is not clean of prior knowledge and pretending
otherwise would be dishonest. It covers a market I have general familiarity
with, and its broad shape is not a secret. What the holdout does protect
against is the specific and much larger risk here: tuning `n`, killzone
boundaries, penetration thresholds and ratio bands until a particular set of
trades looks profitable.

`require_in_sample` exists so that touching the holdout is an explicit act
that shows up in a diff, rather than something that happens by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..schema import NY, validate_bars

#: Fixed before any backtest existed. Do not move this to improve a result.
SPLIT_DATE = pd.Timestamp("2023-01-01", tz=NY)


class HoldoutViolation(RuntimeError):
    """Out-of-sample data was used where in-sample was required."""


@dataclass
class DataSplit:
    """In-sample and out-of-sample halves of a bar series."""

    in_sample: pd.DataFrame
    out_of_sample: pd.DataFrame
    split_date: pd.Timestamp = SPLIT_DATE

    def describe(self) -> str:
        def span(df: pd.DataFrame) -> str:
            if df.empty:
                return "empty"
            return f"{df.index[0].date()} to {df.index[-1].date()}  {len(df):,} bars"

        return (
            f"split at {self.split_date.date()}\n"
            f"  in-sample:     {span(self.in_sample)}\n"
            f"  out-of-sample: {span(self.out_of_sample)}"
        )

    @property
    def ratio(self) -> float:
        total = len(self.in_sample) + len(self.out_of_sample)
        return len(self.in_sample) / total if total else float("nan")


def split_bars(
    df: pd.DataFrame, *, split_date: pd.Timestamp | str = SPLIT_DATE
) -> DataSplit:
    """Cut `df` chronologically at `split_date`.

    The boundary belongs to the out-of-sample side, so a bar exactly at
    midnight on the split date is held out rather than trained on.
    """
    validate_bars(df, name="split input")
    boundary = pd.Timestamp(split_date)
    if boundary.tz is None:
        boundary = boundary.tz_localize(NY)
    else:
        boundary = boundary.tz_convert(NY)

    return DataSplit(
        in_sample=df.loc[df.index < boundary],
        out_of_sample=df.loc[df.index >= boundary],
        split_date=boundary,
    )


def require_in_sample(
    df: pd.DataFrame, *, split_date: pd.Timestamp = SPLIT_DATE, allow: bool = False
) -> None:
    """Raise if `df` contains out-of-sample bars, unless explicitly allowed.

    Call at the top of anything that reports results during development. The
    `allow=True` path is for the single final evaluation, and passing it
    should be a deliberate, reviewable change.
    """
    if allow or df.empty:
        return
    if df.index[-1] >= split_date:
        raise HoldoutViolation(
            f"data extends to {df.index[-1].date()}, past the holdout boundary "
            f"{split_date.date()}. Pass allow=True only for the final "
            "out-of-sample evaluation, and only once the parameters are fixed."
        )
