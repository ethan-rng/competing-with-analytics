"""Seasonal naive: same weekday last week (m=7)."""
import numpy as np

from .common import M, band


def seasonal_naive(y, h, m=M, **_):
    last = y.iloc[-m:].to_numpy()
    mean = np.tile(last, h // m + 1)[:h]
    lo, hi = band(mean, (y - y.shift(m)).dropna())
    return mean, lo, hi
