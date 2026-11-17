"""Naive forecast: tomorrow = today."""
import numpy as np

from .common import band


def naive(y, h, **_):
    mean = np.repeat(y.iloc[-1], h)
    lo, hi = band(mean, y.diff().dropna())
    return mean, lo, hi
