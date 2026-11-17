"""Simple moving average of the last N days (course definition: forecasts roll forward)."""
import numpy as np

from .common import band


def moving_average(y, h, N=7, **_):
    buf = list(y.iloc[-N:])
    out = []
    for _ in range(h):
        f = float(np.mean(buf[-N:]))
        out.append(f)
        buf.append(f)
    mean = np.array(out)
    lo, hi = band(mean, (y - y.rolling(N).mean().shift(1)).dropna())
    return mean, lo, hi
